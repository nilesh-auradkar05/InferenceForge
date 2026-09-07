"""Engine interface.

Every phase of this project implements this one interface. That is what makes
phase N comparable to phase N-1: the harness, trace, and metric code never
change, only the thing behind `stream()`.

Timing contract (the part that is easy to get wrong):
  * The engine emits the `scheduled` event the instant it begins real work for
    a request -- after acquiring whatever lock or slot it needs.
  * Every `token` event carries a timestamp taken AFTER the CUDA work that
    produced it has been synchronized. An unsynchronized timestamp measures
    kernel *launch*, not kernel *completion*, and will make decode loop
    look 50x faster than it is.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass
from typing import Iterator, Literal

from src.utils.trace import Request


@dataclass(slots=True)
class Event:
    kind: Literal["scheduled", "token", "done", "error"]
    t: float
    token_id: int | None = None
    index: int | None = None
    message: str | None = None


def now() -> float:
    return time.perf_counter()


def sync_now(device: str = "cuda") -> float:
    """Timestamp that is honest about async CUDA execution."""
    try:
        import torch

        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
    return time.perf_counter()


class Engine(abc.ABC):
    """Blocking, streaming inference engine."""

    name: str = "engine"

    @abc.abstractmethod
    def setup(self) -> None:
        """Load weights, allocate caches, compile. Not timed."""

    @abc.abstractmethod
    def stream(self, req: Request) -> Iterator[Event]:
        """Yield events for one request. Called from a worker thread."""

    def teardown(self) -> None:
        return None

    @property
    @abc.abstractmethod
    def vocab_size(self) -> int:
        ...

    def stats(self) -> dict:
        """Engine-specific counters (cache hit rate, batch occupancy, ...)."""
        return {}

    def reset_stats(self) -> None:
        """Zero per-run counters.

        The harness reuses one engine across warmup and every sweep point.
        Counters that are not reset become prefix sums, and every ratio built
        on them (wasted_position_ratio, wasted_decode_ratio, ...) is wrong.
        """
        return None

    def reconfigure(self, cfg) -> None:
        """Apply per-sweep-point engine knobs without reloading weights.

        p2 reallocates the KV cache when max_batch_size changes. Default is
        a no-op so B0/B1/P1 can ignore this.
        """
        return None
