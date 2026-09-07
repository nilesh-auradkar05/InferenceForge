"""Metric records and aggregation."""

from __future__ import annotations
 
import threading
import time
from dataclasses import dataclass, field
 
import numpy as np
 
from configs.config import SLOSpec
 
 
@dataclass(slots=True)
class RequestRecord:
    req_id: int
    prompt_len: int
    output_len_requested: int
 
    #: Wall-clock (perf_counter) stamps.
    arrival_t: float = 0.0      # when the driver submitted it
    schedule_t: float = 0.0     # when the engine actually started work
    first_token_t: float = 0.0
    end_t: float = 0.0
 
    output_len_actual: int = 0
    #: Gap between consecutive token emissions, seconds. len == n_tokens - 1.
    itls: list[float] = field(default_factory=list)
    error: str | None = None
 
    # --- derived ---------------------------------------------------------
    @property
    def ttft_ms(self) -> float:
        """Queue wait + prefill. The number a user feels."""
        return (self.first_token_t - self.arrival_t) * 1e3
 
    @property
    def ttft_service_ms(self) -> float:
        """Prefill only, excluding queue. Isolates engine compute."""
        return (self.first_token_t - self.schedule_t) * 1e3
 
    @property
    def queue_ms(self) -> float:
        return (self.schedule_t - self.arrival_t) * 1e3
 
    @property
    def e2e_ms(self) -> float:
        return (self.end_t - self.arrival_t) * 1e3
 
    @property
    def tpot_ms(self) -> float:
        """Mean inter-token latency. Undefined for single-token outputs."""
        return float(np.mean(self.itls)) * 1e3 if self.itls else float("nan")
 
    @property
    def decode_tps(self) -> float:
        if not self.itls:
            return float("nan")
        return 1.0 / float(np.mean(self.itls))
 
 
def _pct(a: np.ndarray, q: float) -> float:
    return float(np.percentile(a, q)) if a.size else float("nan")
 
 
def aggregate(
    records: list[RequestRecord],
    *,
    wall_start: float,
    wall_end: float,
    slo: SLOSpec,
) -> dict:
    ok = [r for r in records if r.error is None and r.output_len_actual > 0]
    failed = len(records) - len(ok)
    wall = max(wall_end - wall_start, 1e-9)
 
    if not ok:
        return {"n_requests": 0, "n_failed": failed, "wall_s": wall}
 
    ttft = np.array([r.ttft_ms for r in ok])
    ttft_svc = np.array([r.ttft_service_ms for r in ok])
    queue = np.array([r.queue_ms for r in ok])
    e2e = np.array([r.e2e_ms for r in ok])
    tpot = np.array([r.tpot_ms for r in ok if r.itls])
    all_itls = np.array([x * 1e3 for r in ok for x in r.itls])
 
    prompt_tok = int(sum(r.prompt_len for r in ok))
    out_tok = int(sum(r.output_len_actual for r in ok))
 
    # Goodput: requests that met BOTH targets, per second of wall clock.
    met = [
        r for r in ok
        if r.ttft_ms <= slo.ttft_p99_ms
        and (not r.itls or r.tpot_ms <= slo.tpot_p99_ms)
    ]
 
    return {
        "n_requests": len(ok),
        "n_failed": failed,
        "wall_s": wall,
        # throughput
        "request_throughput_rps": len(ok) / wall,
        "output_throughput_tps": out_tok / wall,
        "total_throughput_tps": (prompt_tok + out_tok) / wall,
        "goodput_rps": len(met) / wall,
        "slo_attainment": len(met) / len(ok),
        # ttft
        "ttft_mean_ms": float(ttft.mean()),
        "ttft_p50_ms": _pct(ttft, 50),
        "ttft_p90_ms": _pct(ttft, 90),
        "ttft_p99_ms": _pct(ttft, 99),
        "ttft_service_p50_ms": _pct(ttft_svc, 50),
        "queue_p50_ms": _pct(queue, 50),
        "queue_p99_ms": _pct(queue, 99),
        # inter-token
        "tpot_mean_ms": float(tpot.mean()) if tpot.size else float("nan"),
        "tpot_p50_ms": _pct(tpot, 50),
        "tpot_p99_ms": _pct(tpot, 99),
        "itl_p99_ms": _pct(all_itls, 99),
        "per_stream_tps": float(1e3 / tpot.mean()) if tpot.size else float("nan"),
        # e2e
        "e2e_p50_ms": _pct(e2e, 50),
        "e2e_p99_ms": _pct(e2e, 99),
        # volume
        "total_prompt_tokens": prompt_tok,
        "total_output_tokens": out_tok,
    }
 
 
class GpuTelemetry:
    """Background sampler for GPU utilization and memory.
 
    Runs in a thread and does NOT call torch.cuda.synchronize -- sampling must
    never perturb the thing being measured.
    """
 
    def __init__(self, interval_s: float = 0.25, device: int = 0):
        self.interval_s = interval_s
        self.device = device
        self.samples: list[dict] = []
        self.physical_device = device
        self.foreign_procs = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle = None
        self._nvml = None
 
    @staticmethod
    def _physical_index(torch_index: int) -> int:
        """Map a torch device index to a physical NVML index.
 
        CUDA_VISIBLE_DEVICES remaps them. Reading NVML index 0 while torch runs
        on physical GPU 2 reports a NEIGHBOUR'S memory and utilization -- which
        looks like mysterious foreign allocation on your measurement card.
        """
        import os
 
        vis = os.environ.get("CUDA_VISIBLE_DEVICES")
        if not vis:
            return torch_index
        ids = [x.strip() for x in vis.split(",") if x.strip() != ""]
        try:
            return int(ids[torch_index])
        except (ValueError, IndexError):
            return torch_index  # UUID form; caller must verify manually
 
    def _init_nvml(self) -> None:
        try:
            import pynvml
 
            pynvml.nvmlInit()
            self._nvml = pynvml
            self.physical_device = self._physical_index(self.device)
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.physical_device)
            try:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(self._handle)
                self.foreign_procs = max(0, len(procs) - 1)
                if self.foreign_procs:
                    print(
                        f"[warn] {self.foreign_procs} other process(es) on GPU "
                        f"{self.physical_device}. Measurements are contaminated."
                    )
            except Exception:
                pass
        except Exception:
            self._nvml = None
 
    def _loop(self) -> None:
        while not self._stop.is_set():
            s: dict = {"t": time.perf_counter()}
            if self._nvml is not None:
                try:
                    u = self._nvml.nvmlDeviceGetUtilizationRates(self._handle)
                    m = self._nvml.nvmlDeviceGetMemoryInfo(self._handle)
                    s["gpu_util_pct"] = float(u.gpu)
                    s["mem_util_pct"] = float(u.memory)
                    s["mem_used_gb"] = m.used / 1024**3
                except Exception:
                    pass
            try:
                import torch
 
                if torch.cuda.is_available():
                    s["torch_alloc_gb"] = torch.cuda.memory_allocated(self.device) / 1024**3
                    s["torch_reserved_gb"] = torch.cuda.memory_reserved(self.device) / 1024**3
            except Exception:
                pass
            self.samples.append(s)
            self._stop.wait(self.interval_s)
 
    def start(self) -> None:
        self._init_nvml()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
 
    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
 
    def summary(self) -> dict:
        if not self.samples:
            return {}
        out: dict = {}
        for key in ("gpu_util_pct", "mem_util_pct", "mem_used_gb", "torch_alloc_gb", "torch_reserved_gb"):
            vals = np.array([s[key] for s in self.samples if key in s])
            if vals.size:
                out[f"{key}_mean"] = float(vals.mean())
                out[f"{key}_max"] = float(vals.max())
        out["telemetry_samples"] = len(self.samples)
        out["nvml_physical_device"] = self.physical_device
        out["foreign_procs_on_gpu"] = self.foreign_procs
        # NVML mem_used is the caching allocator's reserved pool plus CUDA
        # context / driver overhead (~1 GiB). With foreign_procs_on_gpu==0
        # that gap is not a neighbour process.
        if "mem_used_gb_mean" in out and "torch_reserved_gb_mean" in out:
            out["nvml_minus_reserved_gb_mean"] = (
                out["mem_used_gb_mean"] - out["torch_reserved_gb_mean"]
            )
        return out
 