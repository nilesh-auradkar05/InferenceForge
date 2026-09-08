"""The KV Cache from Scratch"""

from __future__ import annotations

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
 
 
class ScratchKVEngine(Engine):
    name = "p1_scratch_kv"
 
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.model: Qwen3Scratch | None = None
        self.cache: PreallocatedKVCache | None = None
        self._vocab = 0
        self._lock = threading.Lock()
        self.max_len = int(cfg.extra.get("max_len", 4096))
        gb = cfg.extra.get("kv_budget_gb")
        self.budget_bytes = int(float(gb) * 2**30) if gb else None
        self.sync_per_token = bool(cfg.extra.get("sync_per_token", True))
        self._steps = 0
        self._prefill_tokens = 0
        self._oom_count = 0
 
    def setup(self) -> None:
        from transformers import AutoConfig, AutoModelForCausalLM
 
        dtype = _DTYPES[self.cfg.dtype.value]
        conf = AutoConfig.from_pretrained(self.cfg.model_id)
        dims = dims_from_hf_config(conf)
        self._vocab = dims.vocab
 
        hf = AutoModelForCausalLM.from_pretrained(self.cfg.model_id, dtype=dtype)
        sd = hf.state_dict()
        self.model = Qwen3Scratch(sd, dims, self.cfg.device, dtype, max_position=self.max_len)
        del hf, sd
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
 
        self.cache = PreallocatedKVCache(
            dims, self.max_len, self.cfg.device, dtype,
            batch_size=1, budget_bytes=self.budget_bytes,
        )
        print(f"[p1] {self.cache.stats()}")
 
    @property
    def vocab_size(self) -> int:
        return self._vocab
 
    def _sync(self) -> float:
        if self.sync_per_token and self.cfg.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        return time.perf_counter()
 
    def stream(self, req: Request) -> Iterator[Event]:
        # Still serialized: one sequence at a time. Concurrency > 1 will queue,
        # and throughput will NOT improve. That is phase 3's problem, not this
        # phase's failure.
        with self._lock:
            ids = logits = nxt = None
            try:
                yield Event(kind="scheduled", t=time.perf_counter())
                self.cache.reset()
 
                ids = torch.tensor(
                    [req.prompt_token_ids], dtype=torch.long, device=self.cfg.device
                )
                pos = 0
 
                # --- prefill: one forward over the whole prompt ---
                logits = self.model.forward(ids, self.cache, start_pos=pos)
                pos += ids.shape[1]
                self._prefill_tokens += ids.shape[1]
                nxt = torch.argmax(logits, dim=-1, keepdim=True)
                t = self._sync()
                yield Event(kind="token", t=t, token_id=int(nxt.item()), index=0)
 
                # --- decode: one token in, one token out ---
                for i in range(1, req.output_len):
                    logits = self.model.forward(nxt, self.cache, start_pos=pos)
                    pos += 1
                    self._steps += 1
                    nxt = torch.argmax(logits, dim=-1, keepdim=True)
                    t = self._sync()
                    yield Event(kind="token", t=t, token_id=int(nxt.item()), index=i)
 
                yield Event(kind="done", t=time.perf_counter())
            except torch.cuda.OutOfMemoryError:
                # Without this the allocator stays fragmented and EVERY later
                # request in the sweep fails too -- which reads as "the engine
                # collapsed at concurrency 16" when in fact one request did.
                self._oom_count += 1
                del ids, logits, nxt
                ids = logits = nxt = None
                torch.cuda.empty_cache()
                raise
            finally:
                # Runs on normal completion AND on generator .close(). Drops
                # activation tensors before the next request allocates.
                del ids, logits, nxt
                self.cache.reset()
 
    def reset_stats(self) -> None:
        self._steps = 0
        self._prefill_tokens = 0
        self._oom_count = 0

    def stats(self) -> dict:
        s = {"decode_steps": self._steps, "prefill_tokens": self._prefill_tokens,
             "sync_per_token": self.sync_per_token, "oom_count": self._oom_count}
        if self.cache is not None:
            s.update(self.cache.stats())
        return s
 
    def teardown(self) -> None:
        """Idempotent and safe to call after a FAILED setup, where `model` or
        `cache` may be None or only half-built."""
        cache, self.cache = self.cache, None
        model, self.model = self.model, None
        if model is not None:
            model.layers.clear()   # LayerWeights hold direct tensor refs
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
 