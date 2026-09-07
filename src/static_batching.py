"""
P2 - Static Batching

A thread collects arrivals into a batch, runs the whole batch to
completion, then collects the next one.

The two costs this deliberately incurs, both measured:

    1. PADDING WASTE:
        Prompts are left-padded to the longest in the batch. Those pad
        positions consume real FLOPs and real KV slots and produce nothing.
        Reported as `pad_token_ratio`.

    2. HEAD-OF-LINE BLOCKING:
        The batch runs until its LONGEST sequence completes.
        A request wanting 8 tokens batched with one wanting 512 pays for 512 steps.
        Meanwhile new arrivals wait for the entire batch to drain.
        Reported as `wasted_decode_ratio` and visible as TTFT p99.

Both problems are what CONTINOUS BATCHING solves.

Why left Padding and not right padding?
Left padding makes every row's write position identical, so the
cache advances by one slot per step for the whole batch. Right padding would
need per-row write offsets, which is exactly what block tables provide,
and why continuous batching wants paged memory.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Iterator

import torch

from configs.config import EngineConfig
from src.utils.engine import Engine, Event
from src.utils.trace import Request

from src.kv_cache_src.kv_cache import PreallocatedKVCache
from src.model_src.qwen3 import Qwen3Scratch, dims_from_hf_config

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

class _Slot:
    """One request's mailbox. The scheduler pushes, `stream()` drains."""

    __slots__ = ("req", "q")

    def __init__(self, req: Request):
        self.req = req
        self.q: queue.Queue = queue.Queue()

class StaticBatchEngine(Engine):
    name = "p2_static_batch"

    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.model: Qwen3Scratch | None = None
        self.cache: PreallocatedKVCache | None = None
        self._vocab = 0
        self.max_batch = int(cfg.max_batch_size)
        self.max_len = int(cfg.extra.get("max_len", 4096))

        # How long the scheduler waits for a batch to fill before launching a
        # partial one. Pure latency vs throughput knob: raise it and batches get
        # fuller but every request waits longer
        self.batch_timeout_s = float(cfg.extra.get("batch_timeout_s", 0.010))
        gb = cfg.extra.get("kv_budget_gb")
        self.budget_bytes = int(float(gb) * 2**30) if gb else None
        self.sync_per_token = bool(cfg.extra.get("sync_per_token", True))
        self._dims = None
        self._dtype = None

        self._inbox: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._stop = threading.Event()
        self._reset_counters()

    def _reset_counters(self) -> None:
        self._batches_run = 0
        self._batched_requests = 0
        self._useful_decode_steps = 0          # steps that produced a wanted token
        self._padded_decode_steps = 0          # steps burned past a row's output_len
        self._prompt_tokens = 0
        self._pad_prompt_tokens = 0
        self._oom_count = 0

    def reset_stats(self) -> None:
        self._reset_counters()

    def _alloc_cache(self) -> None:
        old, self.cache = self.cache, None
        if old is not None:
            old.k.clear()
            old.v.clear()
            del old
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self.cache = PreallocatedKVCache(
            self._dims, self.max_len, self.cfg.device, self._dtype,
            batch_size=self.max_batch, budget_bytes=self.budget_bytes,
        )

    def reconfigure(self, cfg: EngineConfig) -> None:
        """Reallocate KV for a new max_batch_size. Weights stay loaded."""
        new_batch = int(cfg.max_batch_size)
        self.cfg = cfg
        self.batch_timeout_s = float(cfg.extra.get("batch_timeout_s", self.batch_timeout_s))
        gb = cfg.extra.get("kv_budget_gb")
        self.budget_bytes = int(float(gb) * 2**30) if gb else self.budget_bytes
        if new_batch == self.max_batch and self.cache is not None:
            return
        self.max_batch = new_batch
        if self._dims is None:
            return
        self._alloc_cache()
        print(f"[p2] batch={self.max_batch} {self.cache.stats()}")

    def setup(self) -> None:
        from transformers import AutoConfig, AutoModelForCausalLM

        dtype = _DTYPES[self.cfg.dtype.value]
        conf = AutoConfig.from_pretrained(self.cfg.model_id)
        dims = dims_from_hf_config(conf)
        self._vocab = dims.vocab

        hf = AutoModelForCausalLM.from_pretrained(self.cfg.model_id, dtype=dtype)
        self.model = Qwen3Scratch(hf.state_dict(), dims, self.cfg.device, dtype, max_position=self.max_len)

        del hf
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._dims = dims
        self._dtype = dtype
        # KV is now allocated for max_batch rows, so the budget bites B times
        # harder. This is the first phase where the kv budget can actually runout
        self._alloc_cache()
        print(f"[p2] batch={self.max_batch} {self.cache.stats()}")

        self._stop.clear()
        self._worker = threading.Thread(target=self._scheduler_loop, daemon=True)
        self._worker.start()

    @property
    def vocab_size(self) -> int:
        return self._vocab

    # ---- SCHEDULER -----------------------------------------------------------------------------------------
    def _collect_batch(self) -> list[_Slot]:
        """Block for the first arrival, then briefly greedily fill."""
        try:
            first = self._inbox.get(timeout=0.05)
        except queue.Empty:
            return []

        batch = [first]
        deadline = time.perf_counter() + self.batch_timeout_s
        while len(batch) < self.max_batch:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                batch.append(self._inbox.get(timeout=remaining))
            except queue.Empty:
                break
        
        return batch

    def _scheduler_loop(self) -> None:
        while not self._stop.is_set():
            batch = self._collect_batch()
            if not batch:
                continue
            try:
                self._run_batch(batch)
            except Exception as exc:
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    self._oom_count += 1
                    torch.cuda.empty_cache()
                for slot in batch:
                    slot.q.put(Event(kind="error", t=time.perf_counter(), message=repr(exc)))
                    slot.q.put(None)

    def _sync(self) -> float:
        if self.sync_per_token and self.cfg.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()

    @torch.inference_mode()
    def _run_batch(self, batch: list[_Slot]) -> None:
        dev = self.cfg.device
        B = len(batch)
        prompts = [s.req.prompt_token_ids for s in batch]
        outs = [s.req.output_len for s in batch]
        P = max(len(p) for p in prompts)
        max_out = max(outs)

        if P + max_out > self.max_len:
            raise RuntimeError(f"batch needs {P + max_out} slots, cache max_len={self.max_len}")

        self.cache.reset()
        t_sched = time.perf_counter()
        for s in batch:
            s.q.put(Event(kind="scheduled", t=t_sched))

        # ---------- LEFT PAD -----------------------------------------------------------------------
        ids = torch.zeros((B, P), dtype=torch.long, device=dev)
        pos = torch.zeros((B, P), dtype=torch.long, device=dev)
        valid = torch.zeros((B, self.max_len), dtype=torch.bool, device=dev)
        for b, p in enumerate(prompts):
            L = len(p)
            ids[b, P - L:] = torch.tensor(p, dtype=torch.long, device=dev)
            pos[b, P - L:] = torch.arange(L, device=dev)
            valid[b, P - L:P] = True
            self._prompt_tokens += L
            self._pad_prompt_tokens += P - L

        # ---------- PREFILL -----------------------------------------------------------------------
        logits = self.model.forward(ids, self.cache, start_pos=0,
                                    position_ids=pos, valid_mask=valid[:, :P])
        nxt = torch.argmax(logits, dim=-1, keepdim=True)                      # [B, 1]
        cur = P
        valid[:, cur] = True
        emitted = [0] * B
        toks = nxt.squeeze(1).tolist()
        t = self._sync()
        for b, s in enumerate(batch):
            s.q.put(Event(kind="token", t=t, token_id=int(toks[b]), index=0))
            emitted[b] = 1

        self._useful_decode_steps += B

        # ---------- DECODE: EVERY row steps until the longest row is done --------------------------
        next_pos = torch.full((B, 1), 0, dtype=torch.long, device=dev)
        for step in range(1, max_out):
            next_pos[:, 0] = torch.tensor(
                [len(prompts[b]) + step - 1 for b in range(B)], dtype=torch.long, device=dev
            )
            logits = self.model.forward(nxt, self.cache, start_pos=cur,
                                        position_ids=next_pos, valid_mask=valid[:, :cur + 1])
            cur += 1
            valid[:, cur] = True
            nxt = torch.argmax(logits, dim=-1, keepdim=True)
            toks = nxt.squeeze(1).tolist()
            t = self._sync()
            for b, s in enumerate(batch):
                if emitted[b] < outs[b]:
                    s.q.put(Event(kind="token", t=t, token_id=int(toks[b]), index=emitted[b]))
                    emitted[b] += 1
                    self._useful_decode_steps += 1
                else:
                    # This row finished but the batch has not. Its slot is still
                    # computed, still consumes bandwidth, and is thrown away.
                    self._padded_decode_steps += 1

        t_done = time.perf_counter()
        for s in batch:
            s.q.put(Event(kind="done", t=t_done))
            s.q.put(None)
        self._batches_run += 1
        self._batched_requests += B

    # ---- HARNESS FACING -----------------------------------------------------------------------------
    def stream(self, req: Request) -> Iterator[Event]:
        slot = _Slot(req)
        self._inbox.put(slot)
        try:
            while True:
                ev = slot.q.get()
                if ev is None:
                    return
                yield ev

        finally:
            pass

    def stats(self) -> dict:
        total_decode = self._useful_decode_steps + self._padded_decode_steps
        total_prompt = self._prompt_tokens + self._pad_prompt_tokens
        s = {
            "max_batch_size": self.max_batch,
            "batches_run": self._batches_run,
            "mean_batch_size": self._batched_requests / max(1, self._batches_run),
            "useful_decode_steps": self._useful_decode_steps,
            "padded_decode_steps": self._padded_decode_steps,
            "wasted_decode_ratio": self._padded_decode_steps / max(1, total_decode),
            "pad_token_ratio": self._pad_prompt_tokens / max(1, total_prompt),
            "batch_timeout_s": self.batch_timeout_s,
            "oom_count": self._oom_count,
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
