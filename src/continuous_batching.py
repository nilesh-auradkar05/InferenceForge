"""
Continous Batching over a Paged KV Cache.

Paged Attention:
1. Divide Large memory segments into smaller pages.
2. Store logical Page to Physical Memory mapping in a Page Table.
3. Helps to avoid memory fragmentation and improve memory utilization.

Continous Batching:
1. When a prompt meets it EOS token. The page block is freed and the next prompt is added to on going batch and the process repeats.
2. scheduling decisions happen per ITERATION, not per batch. A finished sequence retires and frees its
blocks on the step it finishes; a waiting request takes the freed slot on the very next step.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Iterator

import torch

from configs.config import EngineConfig
from src.utils.engine import Event
from src.utils.trace import Request

from src.paged_kv import PagedKVCache
from src.model_src.qwen3 import Qwen3Scratch, dims_from_hf_config

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

@dataclass
class Seq:
    req: Request
    out_q: queue.Queue
    block_table: list[int] = field(default_factory=list)
    length: int = 0                                        # tokens currently in the cache
    emitted: int = 0
    next_token: int = 0
    admitted_t: float = 0.0

    @property
    def done(self) -> bool:
        return self.emitted >= self.req.output_len

class ContinuousBatchEngine:
    name = "p3_continuous_batching"

    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.model: Qwen3Scratch | None = None
        self.cache: PagedKVCache | None = None
        self._vocab = 0
        self.max_running = int(cfg.max_batch_size)
        self.max_len = int(cfg.extra.get("max_len", 4096))
        self.block_size = int(cfg.extra.get("block_size", 16))
        gb = cfg.extra.get("kv_budget_gb", 4.0)
        self.budget_bytes = int(float(gb) * 2**30)
        self.sync_per_token = bool(cfg.extra.get("sync_per_token", True))
        self.profile_gather_every = int(cfg.extra.get("profile_gather_every", 50))
        self.attn_impl = str(cfg.extra.get("attn_impl", "broadcast"))

        self._inbox: queue.Queue = queue.Queue()
        self._waiting: list[Seq] = []
        self._running: list[Seq] = []
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()
        self._reset_counters()

    def _reset_counters(self) -> None:
        self._decode_steps = 0
        self._wasted_steps = 0
        self._prefill_tokens = 0
        self._iterations = 0
        self._decode_iters = 0
        self._admission_iters = 0
        self._prefill_forwards = 0
        self._batch_size_sum = 0
        self._admission_stalls = 0
        self._oom_count = 0
        self._max_running_seen = 0

    def reset_stats(self) -> None:
        self._reset_counters()
        if self.cache is not None:
            self.cache._gather_samples.clear()
            self.cache._gather_elem_samples.clear()
            self.cache._step = 0

    def setup(self) -> None:
        from transformers import AutoConfig, AutoModelForCausalLM

        dtype = _DTYPES[self.cfg.dtype.value]
        conf = AutoConfig.from_pretrained(self.cfg.model_id)
        dims = dims_from_hf_config(conf)
        self._vocab = dims.vocab

        hf = AutoModelForCausalLM.from_pretrained(self.cfg.model_id, dtype=dtype)
        self.model = Qwen3Scratch(hf.state_dict(), dims, self.cfg.device, dtype,
                                  max_position=self.max_len, attn_impl=self.attn_impl)

        del hf
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.cache = PagedKVCache(
            dims, self.cfg.device, dtype, block_size=self.block_size,
            budget_bytes=self.budget_bytes,
            profile_gather_every=self.profile_gather_every,
        )
        st = self.cache.stats()
        print(f"[p3] max_running={self.max_running} block_size={st['kv_block_size']} "
              f"blocks={st['blocks_total']} kv={st['kv_allocated_gb']:.2f} GiB "
              f"attn={self.attn_impl}")

        self._stop.clear()
        self._worker = threading.Thread(target=self._loop, daemon=True)
        self._worker.start()

    @property
    def vocab_size(self) -> int:
        return self._vocab

    # ------------------ Scheduling -------------------------------------------------------------------------
    def _drain_inbox(self) -> None:
        while True:
            try:
                self._waiting.append(self._inbox.get_nowait())
            except queue.Empty:
                return

    def _try_admit(self) -> list[Seq]:
        """Admit as many waiting requests as blocks and slots allow."""
        admitted = []
        while self._waiting and len(self._running) + len(admitted) < self.max_running:
            seq = self._waiting[0]
            need = self.cache.allocator.blocks_needed(
                len(seq.req.prompt_token_ids) + seq.req.output_len
            )
            if not self.cache.allocator.can_allocate(need):
                self._admission_stalls += 1
                break
            seq.block_table = self.cache.allocator.allocate(need)
            self._waiting.pop(0)
            admitted.append(seq)
        return admitted

    def _retire(self, seq: Seq) -> None:
        self.cache.allocator.free(seq.block_table)
        seq.block_table = []
        seq.out_q.put(Event(kind="done", t=time.perf_counter()))
        seq.out_q.put(None)

    def _sync(self) -> float:
        if self.sync_per_token and self.cfg.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

        return time.perf_counter()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._drain_inbox()
            if not self._waiting and not self._running:
                time.sleep(0.001)
                continue
            try:
                admitted = self._try_admit()
                if admitted:
                    self._prefill(admitted)
                    self._running.extend(admitted)
                    self._admission_iters += 1
                elif self._running:
                    self._decode()
                    self._decode_iters += 1
                self._iterations += 1
                self._max_running_seen = max(self._max_running_seen, len(self._running))
            except torch.cuda.OutOfMemoryError as exc:
                self._oom_count += 1
                torch.cuda.empty_cache()
                for s in list(self._running):
                    s.out_q.put(Event(kind="error", t=time.perf_counter(), message=repr(exc)))
                    self._retire(s)
                self._running.clear()
            except Exception as exc:
                for s in list(self._running):
                    s.out_q.put(Event(kind="error", t=time.perf_counter(), message=repr(exc)))
                    self._retire(s)
                self._running.clear()

    # ------------------ Execution -------------------------------------------------------------------------
    @torch.inference_mode()
    def _prefill(self, seqs: list[Seq]) -> None:
        """One request at a time. Batched prefill needs padding + a mask, which
        is exactly the cost this phase is trying to stop paying"""
        dev = self.cfg.device
        for seq in seqs:
            ids_list = seq.req.prompt_token_ids
            L = len(ids_list)
            seq.admitted_t = time.perf_counter()
            seq.out_q.put(Event(kind="scheduled", t=seq.admitted_t))

            ids = torch.tensor([ids_list], dtype=torch.long, device=dev)
            pos = torch.arange(L, device=dev).unsqueeze(0)
            self.cache.build_write_index([seq.block_table] * L, list(range(L)))
            self.cache._write_idx = self.cache._write_idx.view(1, L).reshape(-1)
            self.cache.build_gather_index([seq.block_table], [L])

            logits = self.model.forward(ids, self.cache, start_pos=0, position_ids=pos, valid_mask=None)
            self._prefill_forwards += 1

            seq.length = L
            self._prefill_tokens += L
            tok = int(torch.argmax(logits, dim=-1).item())
            seq.next_token = tok
            seq.emitted = 1
            self._decode_steps += 1
            seq.out_q.put(Event(kind="token", t=self._sync(), token_id=tok, index=0))
            seq.length += 1
            if seq.done:
                self._running_remove_later = True

    @torch.inference_mode()
    def _decode(self) -> None:
        dev = self.cfg.device
        active = [s for s in self._running if not s.done]
        if not active:
            for s in self._running:
                self._retire(s)
            self._running.clear()
            return

        B = len(active)
        self._batch_size_sum += B
        tokens = torch.tensor([[s.next_token] for s in active], dtype=torch.long, device=dev)
        positions = torch.tensor([[s.length - 1] for s in active], dtype=torch.long, device=dev)

        tables = [s.block_table for s in active]
        lengths = [s.length for s in active]
        self.cache.build_write_index(tables, [L - 1 for L in lengths])
        T = self.cache.build_gather_index(tables, lengths)

        # Rows have different lengths, so a mask is unavoidable in a dense SDPA call.
        valid = torch.zeros((B, T), dtype=torch.bool, device=dev)
        for b, L in enumerate(lengths):
            valid[b, :L] = True

        logits = self.model.forward(tokens, self.cache, start_pos=0,
                                    position_ids=positions, valid_mask=valid)
        self.cache.end_step()
        nxt = torch.argmax(logits, dim=-1).tolist()
        t = self._sync()

        for b, seq in enumerate(active):
            seq.next_token = nxt[b]
            seq.out_q.put(Event(kind="token", t=t, token_id=nxt[b], index=seq.emitted))
            seq.emitted += 1
            seq.length += 1
            self._decode_steps += 1

        # Retire immediately. The main purpose of Continuous Batching
        # To retire the completed sequence and append next seq in batch

        finished = [s for s in self._running if s.done]
        for s in finished:
            self._retire(s)
            self._running.remove(s)

    # ------------------ Harness-facing ----------------------------------------------------------------------
    def stream(self, req: Request) -> Iterator[Event]:
        q: queue.Queue = queue.Queue()
        self._inbox.put(Seq(req=req, out_q=q))

        while True:
            ev = q.get()
            if ev is None:
                return
            yield ev

    def stats(self) -> dict:
        total = self._decode_steps + self._wasted_steps
        s = {
            "max_running": self.max_running,
            "max_running_seen": self._max_running_seen,
            "iterations": self._iterations,
            # Admission groups can shrink as more requests arrive together,
            # while prefill still executes one B=1 model forward per request.
            "admission_iters": self._admission_iters,
            "prefill_forwards": self._prefill_forwards,
            "decode_iters": self._decode_iters,
            "mean_decode_batch": self._batch_size_sum / max(1, self._decode_iters),
            "useful_decode_steps": self._decode_steps,
            "padded_decode_steps": self._wasted_steps,
            "wasted_decode_ratio": self._wasted_steps / max(1, total),
            "prefill_tokens": self._prefill_tokens,
            "admission_stalls": self._admission_stalls,
            "oom_count": self._oom_count,
            "attn_impl": self.attn_impl,
        }
        if self.cache is not None:
            s.update(self.cache.stats())
        return s
 
    def teardown(self) -> None:
        self._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=3.0)
            self._worker = None
        cache, self.cache = self.cache, None
        model, self.model = self.model, None
        if model is not None:
            model.layers.clear()
            model.rope = None
            model.embed = model.lm_head = model.final_norm = None
        if cache is not None:
            cache.k.clear()
            cache.v.clear()
        del cache, model
        import gc
 
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
 