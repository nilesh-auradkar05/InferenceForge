"""A zero-dependency engine that fakes plausible timings.

Purpose: test the harness, the trace, the aggregation, and the W&B wiring on a
laptop, so that GPU hours are only ever spent on real measurements. It models
prefill as linear in prompt length and decode as constant per token, plus a
per-request serialization lock -- enough shape to exercise every code path.
"""

from __future__ import annotations

import threading
import time
from typing import Iterator

from configs.config import EngineConfig
from src.utils.engine import Engine, Event
from src.utils.trace import Request


class FakeEngine(Engine):
    name = "fake"

    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.prefill_s_per_tok = float(cfg.extra.get("prefill_s_per_tok", 2e-5))
        self.decode_s_per_tok = float(cfg.extra.get("decode_s_per_tok", 5e-4))
        self.serialize = bool(cfg.extra.get("serialize", True))
        self._vocab = int(cfg.extra.get("vocab_size", 32000))
        self._lock = threading.Lock()
        self._steps = 0

    def setup(self) -> None:
        return None

    @property
    def vocab_size(self) -> int:
        return self._vocab

    def stream(self, req: Request) -> Iterator[Event]:
        ctx = self._lock if self.serialize else _NullCtx()
        with ctx:
            yield Event(kind="scheduled", t=time.perf_counter())
            time.sleep(self.prefill_s_per_tok * req.prompt_len)
            for i in range(req.output_len):
                time.sleep(self.decode_s_per_tok)
                self._steps += 1
                yield Event(kind="token", t=time.perf_counter(), token_id=(i % self._vocab), index=i)
            yield Event(kind="done", t=time.perf_counter())

    def reset_stats(self) -> None:
        self._steps = 0

    def stats(self) -> dict:
        return {"forward_steps": self._steps}


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False
