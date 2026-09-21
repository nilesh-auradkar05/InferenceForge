"""
Paged KV Cache

The continous cache reserved `max_batch x max_len` slots up front.
Measured result was 0.090 -- 4096 slots reserved per row to hold ~370 tokens.
Page Blocks: a sequence allocates a 16-token block only when it needs one, and
returns every block the instant it retires.

DISCLAIMER:
GATHER COST. SDPA cannot read a block table, so each decode step copies each
row's scattered blocks into a contiguous buffer. That is new traffic the
contiguous cache never paid -- roughly one extra KV read+write per step. It is
instrumented here (`profile_gather`).
"""

from __future__ import annotations

from collections import deque

import torch

from src.model_src.qwen3 import ModelDims

_DTYPE_BYTES = {torch.float16: 2, torch.bfloat16: 2, torch.float32: 4}

class OutOfBlocks(RuntimeError):
    pass

class BlockAllocator:
    """ Fixed pool of fixed-size blocks. No Coalescing needed -- every block is
    the same size, so external fragmentation is structurally impossible. That
    is the whole trick: uniform blocks trade a little internal waste (the last
    partly-filled block per sequence) for zero external fragmentation.
    """

    def __init__(self, num_blocks: int, block_size: int):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._free: deque[int] = deque(range(num_blocks))
        self.peak_used = 0
        self.alloc_calls = 0
        self.failed_allocs = 0

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_used(self) -> int:
        return self.num_blocks - len(self._free)

    def blocks_needed(self, n_tokens: int) -> int:
        return (n_tokens + self.block_size - 1) // self.block_size

    def can_allocate(self, n_blocks: int) -> bool:
        return len(self._free) >= n_blocks

    def allocate(self, n_blocks: int) -> list[int]:
        self.alloc_calls += 1
        if len(self._free) < n_blocks:
            self.failed_allocs += 1
            raise OutOfBlocks(f"need {n_blocks} blocks, have {len(self._free)}")
        out = [self._free.popleft() for _ in range(n_blocks)]
        self.peak_used = max(self.peak_used, self.num_used)
        return out

    def free(self, block_ids: list[int]) -> None:
        self._free.extend(block_ids)

    def stats(self) -> dict:
        return {
            "blocks_total": self.num_blocks,
            "blocks_used": self.num_used,
            "blocks_peak_used": self.peak_used,
            "block_utilization": self.num_used / max(1, self.num_blocks),
            "peak_block_utilization": self.peak_used / max(1, self.num_blocks),
            "failed_allocs": self.failed_allocs,
        }

class PagedKVCache:
    """Storage is [num_blocks, block_size, kv_heads, head_dim] per layer.

    Slot s of block b lives at flat at index b * block_size + s, so both the
    scatter (write one token per row) and the gather (collect a row's history)
    reduce to a single index op on a flattened view.
    """

    def __init__(self, dims: ModelDims, device: str, dtype: torch.dtype,
                 block_size: int = 16, budget_bytes: int | None = None,
                 num_blocks: int | None = None, profile_gather_every: int = 0):
        self.dims = dims
        self.device = device
        self.dtype = dtype
        self.block_size = block_size
        eb = _DTYPE_BYTES[dtype]

        bytes_per_block = 2 * dims.n_layers * block_size * dims.n_kv_heads * dims.head_dim * eb
        if num_blocks is None:
            if budget_bytes is None:
                raise ValueError("give num_blocks or budget_bytes")
            num_blocks = int(budget_bytes // bytes_per_block)

        if num_blocks < 1:
            raise ValueError(f"budget too small: fits {num_blocks} blocks")

        self.num_blocks = num_blocks
        self.bytes_per_block = bytes_per_block
        self.allocated_bytes = num_blocks * bytes_per_block
        self.allocator = BlockAllocator(num_blocks, block_size)

        shape = (num_blocks * block_size, dims.n_kv_heads, dims.head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(dims.n_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(dims.n_layers)]

        # Per-step index tensors, built ONCE and reused across all 36 layers.
        # Rebuilding them per layer would cost 36x the index arithemetic for
        self._write_idx: torch.Tensor | None = None
        self._gather_idx: torch.Tensor | None = None
        self._pos_block: torch.Tensor | None = None
        self._pos_slot: torch.Tensor | None = None
        self._pos_cache_T: int = -1

        # A silently-ignored --block-size produces plausible-looking numbers,
        # which is the worst failure mode a benchmark harness has. Assert the
        # geometry the flag is supposed to have produced.
        expected = block_size * dims.kv_bytes_per_token(eb)
        if bytes_per_block != expected:
            raise AssertionError(
                f"block geometry mismatch: bytes_per_block={bytes_per_block} "
                f"but block_size={block_size} x kv_bytes_per_token={expected // block_size} "
                f"= {expected}"
            )

        self.profile_gather_every = profile_gather_every
        self._step = 0
        self._gather_samples: list[float] = []
        self._gather_elem_samples: list[int] = []

    # --------------- index construction ---------------------------------------------------------------
    def build_write_index(self, block_tables: list[list[int]], positions: list[int]) -> None:
        """Flat slot for each row's next token. positions are absolute"""
        slots = [
            block_tables[b][p // self.block_size] * self.block_size + (p % self.block_size)
            for b, p in enumerate(positions)
        ]

        self._write_idx = torch.tensor(slots, dtype=torch.long, device=self.device)

    def build_gather_index(self, block_tables: list[list[int]], lengths: list[int]) -> int:
        """
        [B, T_max] of flat slots. Padded entries point at slot 0, which is
        harmless ONLY because the caller masks them out. Returns T_max.

        The previous form looped over B in Python, issuing a blocking H2D copy
        plus ~6 kernel launches per row, every decode step. At B=26 that is ~26
        syncs and ~150 launches of scheduling overhead in front of a step that
        should be one dense forward. Launch count here is independent of B: one
        padded host->device transfer of the block tables, then four elementwise
        ops. Position arithmetic is cached because it depends only on T.
        """
        B = len(lengths)
        T = max(lengths)
        bs = self.block_size
        n_blocks_max = (T + bs - 1) // bs

        # Admission allocates prompt+output blocks per sequence, so tables are
        # longer than n_blocks_max and differ across the batch. `[0] * (n - len)`
        # is a no-op when len > n, and torch.tensor then raises on the ragged
        # rows. Pad then slice so every row is exactly the columns we index.
        padded = [(t + [0] * n_blocks_max)[:n_blocks_max] for t in block_tables]
        tables = torch.tensor(padded, dtype=torch.long, device=self.device)  # [B, NB]

        if self._pos_cache_T != T:
            pos = torch.arange(T, device=self.device)
            self._pos_block = pos // bs          # [T]
            self._pos_slot = pos % bs            # [T]
            self._pos_cache_T = T

        # gather block id per (row, position), then convert to a flat slot.
        idx = tables[:, self._pos_block] * bs + self._pos_slot  # [B, T]

        self._gather_idx = idx
        return T

    def gather_elems(self, B: int, T: int) -> int:
        """K+V elements touched by one layer's gather. Used to sanity-check the
        profiled time against the achievable bandwidth floor."""
        return 2 * B * T * self.dims.n_kv_heads * self.dims.head_dim

    # --------------- HOT path ---------------------------------------------------------------------------
    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, start_pos: int):
        """k, v: [B, kv_heads, S, head_dim]. Writes, then returns the gathered
        contiguous history [B, kv_heads, T_max, head_dim]."""
        B, H, S, D = k.shape
        if S == 1:
            self.k[layer_idx].index_copy_(0, self._write_idx, k[:, :, 0, :])
            self.v[layer_idx].index_copy_(0, self._write_idx, v[:, :, 0, :])

        else:
            # Prefill: S contiguous slots per row, so the write is a
            # [B, S] block laid out the same way as the gather index.
            flat = self._write_idx.view(B, S).reshape(-1)
            self.k[layer_idx].index_copy_(0, flat, k.permute(0, 2, 1, 3).reshape(-1, H, D))
            self.v[layer_idx].index_copy_(0, flat, v.permute(0, 2, 1, 3).reshape(-1, H, D))

        # S == 1 is decode. Prefill forwards do NOT advance _step (end_step is
        # only called from the decode path), so if _step happened to sit on a
        # multiple of the sampling period, every prefill layer-0 gather got
        # sampled -- a B=1, T=prompt_len gather roughly two orders of magnitude
        # cheaper than the decode gather we are trying to measure. That is why
        # gather_samples ran ~10x above the expected decode sample count and the
        # reported means were both too small and unstable across runs.
        profile = (
            self.profile_gather_every
            and S == 1
            and layer_idx == 0
            and self._step % self.profile_gather_every == 0
            and self.device.startswith("cuda")
        )
        if profile:
            torch.cuda.synchronize()
            ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
            ev0.record()

        gk = self.k[layer_idx][self._gather_idx]           # [B, T, H, D]
        gv = self.v[layer_idx][self._gather_idx]

        if profile:
            ev1.record()
            torch.cuda.synchronize()
            self._gather_samples.append(ev0.elapsed_time(ev1))
            self._gather_elem_samples.append(
                self.gather_elems(B, int(self._gather_idx.shape[1]))
            )


        return gk.permute(0, 2, 1, 3), gv.permute(0, 2, 1, 3)


    def end_step(self) -> None:
        self._step += 1

    # --------------- Accounting -------------------------------------------------------------------------
    def stats(self) -> dict:
        s = {
            "kv_allocated_gb": self.allocated_bytes / 2**30,
            "kv_block_size": self.block_size,
            "kv_bytes_per_block": self.bytes_per_block,
            "kv_bytes_per_token": self.dims.kv_bytes_per_token(_DTYPE_BYTES[self.dtype]),
        }
        s.update(self.allocator.stats())
        if self._gather_samples:
            n = len(self._gather_samples)
            mean = sum(self._gather_samples) / n
            s["gather_ms_per_layer"] = mean
            # One gather per layer per decode step, so this is the per-step cost
            # paging adds over a contiguous cache. NOT per generated token: at
            # batch B the step emits B tokens, so divide by mean_decode_batch
            # before comparing against TPOT.
            s["gather_ms_per_step"] = mean * self.dims.n_layers
            s["gather_samples"] = n
            if self._gather_elem_samples:
                # Read + write of the gathered slice, so 2x the elements. Lets a
                # reader check the timing against the card's bandwidth instead
                # of taking an unfalsifiable millisecond count on faith.
                eb = _DTYPE_BYTES[self.dtype]
                mean_elems = sum(self._gather_elem_samples) / len(self._gather_elem_samples)
                s["gather_bytes_per_layer"] = 2 * mean_elems * eb
                s["gather_effective_gbps"] = (
                    (2 * mean_elems * eb) / (mean * 1e-3) / 1e9 if mean > 0 else None
                )
        return s
