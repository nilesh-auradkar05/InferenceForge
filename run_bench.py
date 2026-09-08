"""Phase 0 entrypoint.
 
    # smoke test, no GPU, no weights
    python run_bench.py --engine fake --phase p0_smoke --no-wandb
 
    # real baselines
    python run_bench.py --engine b1_hf_generate --phase p0_baseline \
        --model Qwen/Qwen3-4B-Instruct-2507 --concurrency 1
 
    # the curve that actually matters
    python run_bench.py --engine b1_hf_generate --phase p0_baseline \
        --sweep-concurrency 1,2,4,8,16,32

    # p2: sweep max batch at fixed arrival concurrency
    python run_bench.py --engine p2_static_batch --phase p2_static_batch \
        --workload chat_short --concurrency 16 --max-batch-size 1,2,4,8,16
"""

from __future__ import annotations
 
import argparse
import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
 
from configs.config import (
    Dtype, EngineConfig, EnvFingerprint, LoadMode, LoadSpec, RunConfig, SLOSpec, WorkloadSpec,
)
from src.utils.harness import run_benchmark
from src.utils.reporting import log_wandb, save_local
from src.utils.trace import build_trace, trace_summary
 
ENGINES = {
    "fake": ("fake_run", "FakeEngine"),
    "b0_naive_no_cache": ("src.naive_no_cache", "NaiveNoCacheEngine"),
    "b1_hf_generate": ("src.hf_baseline", "HFBaselineEngine"),
    "p1_scratch_kv": ("src.kv_cache_src.kv_scratch", "ScratchKVEngine"),
    "p2_static_batch": ("src.static_batching", "StaticBatchEngine"),
}
 
 
def build_engine(cfg: EngineConfig):
    import importlib
 
    mod_name, cls_name = ENGINES[cfg.engine]
    return getattr(importlib.import_module(mod_name), cls_name)(cfg)
 
 
WORKLOADS = {
    # Short chat turns. Decode-dominated -- this is where KV cache, batching,
    # and quantization show up.
    "chat_short": dict(prompt_len_mean=256, prompt_len_std=96, output_len_mean=128, output_len_std=48),
    # Long context, short answer. Prefill-dominated -- this is where attention
    # kernels and prefix caching show up.
    "rag_long": dict(prompt_len_mean=2048, prompt_len_std=512, output_len_mean=64, output_len_std=16),
    # Tiny, for smoke tests.
    "tiny": dict(prompt_len_mean=32, prompt_len_std=8, output_len_mean=16, output_len_std=4),
}


def csv_ints(s: str) -> list[int]:
    try:
        vals = [int(p.strip()) for p in s.split(",") if p.strip() != ""]
    except ValueError as e:
        raise ValueError(f"expected comma-separated ints, got {s!r}") from e
    if not vals or any(v <= 0 for v in vals):
        raise ValueError("values must be positive integers")
    return vals


@dataclass(frozen=True, slots=True)
class SweepPoint:
    concurrency: int
    rate: float | None
    max_batch: int


def sweep_points(
    *,
    concurrency: int,
    rate: float | None,
    sweep_concurrency: str | None,
    sweep_rate: str | None,
    batch_sizes: list[int],
) -> list[SweepPoint]:
    if sweep_concurrency:
        concs = csv_ints(sweep_concurrency)
        return [SweepPoint(c, None, b) for b in batch_sizes for c in concs]
    if sweep_rate:
        rates = [float(r) for r in sweep_rate.split(",") if r.strip()]
        return [SweepPoint(concurrency, r, b) for b in batch_sizes for r in rates]
    return [SweepPoint(concurrency, rate, b) for b in batch_sizes]


def parse_args(argv: list[str] | None = None):
    p = argparse.ArgumentParser()
    p.add_argument("--engine", required=True, choices=list(ENGINES))
    p.add_argument("--phase", default="p0_baseline")
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--dtype", default="bfloat16", choices=[d.value for d in Dtype])
    p.add_argument("--device", default="cuda")
    p.add_argument("--workload", default="chat_short", choices=list(WORKLOADS))
    p.add_argument("--num-requests", type=int, default=64)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--shared-prefix-frac", type=float, default=0.0)
    p.add_argument("--mode", default="closed", choices=[m.value for m in LoadMode])
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--rate-rps", type=float, default=None)
    p.add_argument("--sweep-concurrency", default=None, help="e.g. 1,2,4,8,16")
    p.add_argument("--sweep-rate", default=None, help="e.g. 0.5,1,2,4 (open loop)")
    p.add_argument("--slo-ttft-ms", type=float, default=1000.0)
    p.add_argument("--slo-tpot-ms", type=float, default=100.0)
    p.add_argument("--wandb-project", default="llm-inference-from-scratch")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--no-wandb", action="store_true")
    p.add_argument("--kv-budget-gb", type=float, default=None,
                   help="Cap KV cache size. Your 48GB card has too much headroom "
                        "for paging/eviction to be observable -- cap it on purpose.")
    p.add_argument("--max-len", type=int, default=4096)
    p.add_argument(
        "--max-batch-size",
        type=csv_ints,
        default=[1],
        help="Single int, or a comma-separated sweep: 1,2,4,8,16",
    )
    p.add_argument("--batch-timeout-ms", type=float, default=10.0,
                   help="How long the scheduler waits to fill a batch. "
                        "Pure latency-vs-throughput knob -- sweep it.")
    p.add_argument("--extra", default="{}", help="JSON dict of engine knobs")
    return p.parse_args(argv)
 
 
def _engine_extra(a) -> dict:
    extra = json.loads(a.extra)
    extra.setdefault("max_len", a.max_len)
    if a.kv_budget_gb is not None:
        extra.setdefault("kv_budget_gb", a.kv_budget_gb)
    extra.setdefault("batch_timeout_s", a.batch_timeout_ms / 1e3)
    return extra
 
 
def make_config(
    a,
    *,
    concurrency: int,
    rate: float | None,
    stamp: str,
    max_batch_size: int,
) -> RunConfig:
    wl = WorkloadSpec(
        name=a.workload,
        num_requests=a.num_requests,
        seed=a.seed,
        shared_prefix_frac=a.shared_prefix_frac,
        **WORKLOADS[a.workload],
    )
    load = LoadSpec(
        mode=LoadMode(a.mode), concurrency=concurrency,
        rate_rps=rate, warmup_requests=a.warmup,
    )
    tag = f"c{concurrency}" if a.mode == "closed" else f"r{rate}"
    if len(a.max_batch_size) > 1 or max_batch_size != 1:
        tag = f"b{max_batch_size}-{tag}"
    return RunConfig(
        run_name=f"{a.engine}-{a.workload}-{tag}-{stamp}",
        phase=a.phase,
        engine=EngineConfig(
            engine=a.engine, model_id=a.model, dtype=Dtype(a.dtype),
            device=a.device, max_batch_size=max_batch_size,
            extra=_engine_extra(a),
        ),
        workload=wl,
        load=load,
        slo=SLOSpec(ttft_p99_ms=a.slo_ttft_ms, tpot_p99_ms=a.slo_tpot_ms),
        wandb_project=a.wandb_project,
        wandb_entity=a.wandb_entity,
    )
 
 
def main() -> None:
    a = parse_args()
    stamp = datetime.now().strftime("%m%d-%H%M%S")
    env = EnvFingerprint.capture()
    print(json.dumps(env.model_dump(), indent=2))
 
    points = sweep_points(
        concurrency=a.concurrency,
        rate=a.rate_rps,
        sweep_concurrency=a.sweep_concurrency,
        sweep_rate=a.sweep_rate,
        batch_sizes=a.max_batch_size,
    )
 
    # Engine is built ONCE and reused across sweep points: reloading weights
    # between points would re-pay warmup and inject allocator noise. Batch-size
    # changes reallocate KV only -- see Engine.reconfigure.
    first = points[0]
    first_cfg = make_config(
        a, concurrency=first.concurrency, rate=first.rate,
        stamp=stamp, max_batch_size=first.max_batch,
    )
    engine = build_engine(first_cfg.engine)
    # setup() is INSIDE the try: loading 8 GB of weights is the single most
    # likely place to OOM, and a half-built engine still holds GPU memory.
    try:
        engine.setup()
        for pt in points:
            cfg = make_config(
                a, concurrency=pt.concurrency, rate=pt.rate,
                stamp=stamp, max_batch_size=pt.max_batch,
            )
            engine.reconfigure(cfg.engine)
            reqs = build_trace(cfg.workload, cfg.load, vocab_size=engine.vocab_size)
            tstats = trace_summary(reqs)
            print(f"\n=== {cfg.run_name} ===")
            print(json.dumps(tstats, indent=2))
 
            summary, records = asyncio.run(run_benchmark(engine, reqs, cfg))
            print(json.dumps({k: v for k, v in summary.items()}, indent=2, default=str))
 
            local = save_local(cfg, env, summary, records)
            print(f"[saved] {local}")
            if not a.no_wandb:
                log_wandb(cfg, env, summary, records, tstats, local_dir=local)
    finally:
        engine.teardown()
 
 
if __name__ == "__main__":
    main()
