"""The KV cache

Phase 1: Static Preallocation of the KV cache.
 
Two design decisions:
 
1. PREALLOCATED, not concatenated. `torch.cat([past_k, new_k], dim=2)` every
   step allocates a fresh tensor and copies the whole history -- O(n^2) memory
   traffic hiding inside an O(n) algorithm. On a memory-bound decode loop that
   is the difference between ~80 tok/s and ~25 tok/s. Preallocate, then write a
   single row per step.
 
2. AN EXPLICIT BUDGET. On a 48 GB RTX 6000 Ada with a 4B model you have room
   for ~276k cached tokens -- roughly 720 concurrent short chats. 
   Under those conditions paging, eviction, and fragmentation are all unobservable no-ops.
   Hence, the `budget_bytes` caps the cache below the hardware limit on purpose,
   so the allocator has a job and the phase-4 experiments have a signal. Sweeping the
   budget sweeps you across the memory-bound / compute-bound roofline crossover.
"""

from __future__ import annotations

import torch

from src.model_src.qwen3 import ModelDims

_DTYPE_BYTES = {torch.float16: 2, torch.bfloat16: 2, torch.float32: 4}

class KVCacheBudgetExceeded(RuntimeError):
    ...


class PreallocatedKVCache:
    """Contiguous per-layer storage: [B, kv_heads, max_len, head_dim].
 
    Phase 1 is single-sequence (B=1); the batch dim is present so phases 2-3
    can slot in without reshaping the world. Phase 4 replaces the contiguous
    `max_len` axis with block tables -- that is the only thing that changes.
    """

    def __init__(self, dims: ModelDims, max_len: int, device: str, dtype: torch.dtype,
                 batch_size: int = 1, budget_bytes: int | None = None):

        self.dims = dims
        self.max_len = max_len
        self.batch_size = batch_size
        self.dtype = dtype
        self.device = device

        nbytes = self.required_bytes(dims, max_len, batch_size, dtype)
        if budget_bytes is not None and nbytes > budget_bytes:
            max_fit = self.max_tokens_for_budget(dims, budget_bytes, batch_size, dtype)
            raise KVCacheBudgetExceeded(
                f"""cache needs {nbytes / 2**30:.2f} GiB but budget is {budget_bytes / 2**30:.2f} GiB;
                budget fits {max_fit} tokens (requested {max_len} x batch {batch_size})"""
            )

        self.budget_bytes = budget_bytes
        self.allocated_bytes = nbytes
        
        shape = (batch_size, dims.n_kv_heads, max_len, dims.head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(dims.n_layers)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(dims.n_layers)]
        self.seq_len = 0

    @staticmethod
    def required_bytes(dims: ModelDims, max_len: int, batch_size: int, dtype) -> int:
        eb = _DTYPE_BYTES[dtype]
        return 2 * dims.n_layers * batch_size * dims.n_kv_heads * max_len * dims.head_dim * eb

    @staticmethod
    def max_tokens_for_budget(dims: ModelDims, budget_bytes: int, batch_size: int, dtype) -> int:
        per_token = dims.kv_bytes_per_token(_DTYPE_BYTES[dtype]) * batch_size
        return budget_bytes // per_token

    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor, start_pos: int):
        """Write k/v at [start_pos, start_pos+S), return the full valid span.
 
        The returned slice is a VIEW, not a copy. Returning `self.k[i]` whole
        and letting attention see zero-padded positions would be a correctness
        bug that a causal mask hides during prefill but not during decode.
        """

        S = k.shape[2]
        end = start_pos + S
        if end > self.max_len:
            raise KVCacheBudgetExceeded(
                f"sequence length {end} exceeds cache max_len {self.max_len}"
            )

        self.k[layer_idx][:, :, start_pos:end, :] = k
        self.v[layer_idx][:, :, start_pos:end, :] = v

        if layer_idx == self.dims.n_layers - 1:
            self.seq_len = end

        return self.k[layer_idx][:, :, :end, :], self.v[layer_idx][:, :, :end, :]

    def reset(self) -> None:
        """Logical clear only. Zeroing the storage would cost a full pass over
        the cache on every request, which is pure bandwidth you cannot spare."""

        self.seq_len = 0

    def stats(self) -> dict:
        return {
            "kv_allocated_gb": self.allocated_bytes / 2**30,
            "kv_budget_db": (self.budget_bytes / 2**30) if self.budget_bytes else None,
            "kv_bytes_per_token": self.dims.kv_bytes_per_token(_DTYPE_BYTES[self.dtype]),
            "kv_seq_len": self.max_len,
            "kv_occupancy": self.seq_len / self.max_len if self.max_len else 0.0,
        }
