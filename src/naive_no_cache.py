"""B0 -- the deliberately dumb version. Post item 2.
 
Every decode step re-runs the forward pass over the ENTIRE sequence and throws
away all of it except the last position's logits. Total work is quadratic in
sequence length.
 
This exists to make one number visible: the gap between B0 and B1 is what KV
caching alone buys. Once you have seen it, B0 never appears in a speedup
chart again -- comparing anything to B0 is comparing to a strawman.
 
Safety valve: `max_total_len` guards against a long-output request turning a
benchmark into an overnight job. Trips are recorded, not silently swallowed.
"""

from __future__ import annotations

import threading
import time
from typing import Iterator

from configs.config import EngineConfig
from src.utils.engine import Engine, Event
from src.utils.trace import Request

_DTYPES = {"float16": "float16", "bfloat16": "bfloat16", "float32": "float32"}

class NaiveNoCacheEngine(Engine):
    name = "b0_naive_no_cache"

    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.model = None
        self._vocab = 0
        self._lock = threading.Lock()
        self._n_forwards = 0
        self._n_positions = 0
        self.max_total_len = int(cfg.extra.get("max_total_len", 4096))

    def setup(self) -> None:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM

        dtype = getattr(torch, _DTYPES[self.cfg.dtype.value])
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_id, dtype=dtype, attn_implementation="sdpa",
        ).to(self.cfg.device).eval()
        conf = AutoConfig.from_pretrained(self.cfg.model_id)
        self._vocab = int(getattr(conf, "vocab_size", self.model.config.vocab_size))

    @property
    def vocab_size(self) -> int:
        return self._vocab

    def _sync(self) -> float:
        import torch

        if self.cfg.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

        return time.perf_counter()

    def stream(self, req: Request) -> Iterator[Event]:
        import torch

        with self._lock:
            yield Event(kind="scheduled", t=time.perf_counter())

            ids = torch.tensor([req.prompt_token_ids], dtype=torch.long, device=self.cfg.device)
            emitted = 0
            with torch.inference_mode():
                for _ in range(req.output_len):
                    if ids.shape[1] >= self.max_total_len:
                        yield Event(
                            kind="error",
                            t=time.perf_counter(),
                            message=f"max_total_len {self.max_total_len} exceeded after {emitted} tokens",
                        )
                        break

                    # Full-sequence forward, use_cache=False, keep 1 row.
                    # Growing `ids` hits a new allocation size every step; the
                    # caching allocator keeps every distinct block. That is why
                    # B0 fragments tens of GB of reserved memory while B1/P1
                    # do not -- keep it, it is the empirical argument for paged KV.
                    logits = self.model(input_ids=ids, use_cache=False).logits[:, -1, :]
                    nxt = torch.argmax(logits, dim=-1, keepdim=True)
                    self._n_forwards += 1
                    self._n_positions += ids.shape[1]
                    ids = torch.cat([ids, nxt], dim=1)
                    t = self._sync()
                    emitted += 1
                    yield Event(kind="token", t=t, token_id=int(nxt.item()), index=emitted-1)
            yield Event(kind="done", t=time.perf_counter())

    def reset_stats(self) -> None:
        self._n_forwards = 0
        self._n_positions = 0

    def stats(self) -> dict:
        return {
            "forward_steps": self._n_forwards,
            "positions_forwarded": self._n_positions,
            "wasted_position_ratio": (
                1.0 - self._n_forwards / self._n_positions if self._n_positions else 0.0
            ),
        }

    def teardown(self) -> None:
        import torch

        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
