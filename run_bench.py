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
    "p1_scratch_kv": ("src.kv_scratch_src.kv_scratch", "ScratchKVEngine"),
    "p2_static_batch": ("src.static_batching", "StaticBatchEngine"),
    "p3_continuous_batch": ("src.continuous_batching", "ContinuousBatchEngine"),
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
 
 
def parse_args():
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
    p.add_argument("--max-batch-size", type=int, default=1,
                   help="p2: fixed batch. p3: max concurrently RUNNING sequences.")
    p.add_argument("--block-size", type=int, default=16,
                   help="KV block granularity. Smaller = less internal waste, "
                        "more index overhead. Sweep it.")
    p.add_argument("--profile-gather-every", type=int, default=50,
                   help="Sample the gather cost every N decode steps. 0 = off.")
    p.add_argument("--batch-timeout-ms", type=float, default=10.0,
                   help="How long the scheduler waits to fill a batch. "
                        "Pure latency-vs-throughput knob -- sweep it.")
    p.add_argument("--extra", default="{}", help="JSON dict of engine knobs")
    return p.parse_args()
 
 
def _engine_extra(a) -> dict:
    extra = json.loads(a.extra)
    extra.setdefault("max_len", a.max_len)
    if a.kv_budget_gb is not None:
        extra.setdefault("kv_budget_gb", a.kv_budget_gb)
    extra.setdefault("batch_timeout_s", a.batch_timeout_ms / 1e3)
    extra.setdefault("block_size", a.block_size)
    extra.setdefault("profile_gather_every", a.profile_gather_every)
    return extra
 
 
def make_config(a, *, concurrency: int, rate: float | None, stamp: str) -> RunConfig:
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
    return RunConfig(
        run_name=f"{a.engine}-{a.workload}-{tag}-{stamp}",
        phase=a.phase,
        engine=EngineConfig(
            engine=a.engine, model_id=a.model, dtype=Dtype(a.dtype),
            device=a.device, max_batch_size=a.max_batch_size,
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
 
    if a.max_batch_size > a.concurrency and not a.sweep_concurrency:
        print(f"[warn] max_batch_size={a.max_batch_size} > concurrency={a.concurrency}: "
              f"the batch can never fill. Raise --concurrency to at least the batch "
              f"size or the engine measures partial batches.")
 
    if a.sweep_concurrency:
        points = [(int(c), None) for c in a.sweep_concurrency.split(",")]
    elif a.sweep_rate:
        points = [(a.concurrency, float(r)) for r in a.sweep_rate.split(",")]
    else:
        points = [(a.concurrency, a.rate_rps)]
 
    # Engine is built ONCE and reused across sweep points: reloading weights
    # between points would re-pay warmup and inject allocator noise.
    first_cfg = make_config(a, concurrency=points[0][0], rate=points[0][1], stamp=stamp)
    engine = build_engine(first_cfg.engine)
    # setup() is INSIDE the try: loading 8 GB of weights is the single most
    # likely place to OOM, and a half-built engine still holds GPU memory.
    try:
        engine.setup()
        for conc, rate in points:
            cfg = make_config(a, concurrency=conc, rate=rate, stamp=stamp)
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
 