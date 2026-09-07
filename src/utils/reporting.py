"""Persistence and Weights & Biases logging.

Rules encoded here:
  * Local JSON is written FIRST and always. W&B is a viewer, not the system of
    record -- a network hiccup must never cost you a GPU run.
  * The workload fingerprint and the env fingerprint ride along with every run.
    A speedup number without both attached cannot be defended.
  * Per-request rows go up as a table so you can inspect the tail, not just the
    mean. The tail is where every serving bug lives.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

from configs.config import EnvFingerprint, RunConfig
from src.utils.metrics import RequestRecord


def _clean(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def records_to_rows(records: list[RequestRecord]) -> list[dict]:
    rows = []
    for r in records:
        rows.append(
            {
                "req_id": r.req_id,
                "prompt_len": r.prompt_len,
                "output_len_requested": r.output_len_requested,
                "output_len_actual": r.output_len_actual,
                "queue_ms": _clean(r.queue_ms),
                "ttft_ms": _clean(r.ttft_ms),
                "ttft_service_ms": _clean(r.ttft_service_ms),
                "tpot_ms": _clean(r.tpot_ms),
                "e2e_ms": _clean(r.e2e_ms),
                "decode_tps": _clean(r.decode_tps),
                "error": r.error,
            }
        )
    return rows


def save_local(
    cfg: RunConfig, env: EnvFingerprint, summary: dict, records: list[RequestRecord]
) -> Path:
    out = Path(cfg.output_dir) / cfg.phase / cfg.run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(cfg.model_dump_json(indent=2))
    (out / "env.json").write_text(env.model_dump_json(indent=2))
    (out / "summary.json").write_text(
        json.dumps({k: _clean(v) for k, v in summary.items()}, indent=2)
    )
    with (out / "requests.jsonl").open("w") as f:
        for row in records_to_rows(records):
            f.write(json.dumps(row) + "\n")
    return out


def log_wandb(
    cfg: RunConfig,
    env: EnvFingerprint,
    summary: dict,
    records: list[RequestRecord],
    trace_stats: dict,
    local_dir: "Path | None" = None,
):
    try:
        import wandb
    except ImportError:
        print("[reporting] wandb not installed; local results only")
        return None

    tags = [
        cfg.phase,
        cfg.engine.engine,
        f"load:{cfg.load.mode.value}",
        f"conc:{cfg.load.concurrency}",
        f"wl:{cfg.workload.name}",
    ]
    run = wandb.init(
        project=cfg.wandb_project,
        entity=cfg.wandb_entity,
        name=cfg.run_name,
        group=cfg.workload.fingerprint(),   # groups everything that IS comparable
        job_type=cfg.phase,
        tags=tags,
        config={
            **cfg.model_dump(mode="json"),
            "env": env.model_dump(mode="json"),
            "trace": trace_stats,
            "workload_fingerprint": cfg.workload.fingerprint(),
            "comparison_key": cfg.comparison_key(),
        },
        reinit=True,
    )

    for k, v in summary.items():
        run.summary[k] = _clean(v)

    rows = records_to_rows(records)
    if rows:
        cols = list(rows[0].keys())
        table = wandb.Table(columns=cols, data=[[r[c] for c in cols] for r in rows])
        run.log({"requests": table})

    if local_dir is not None and os.environ.get("WANDB_LOG_ARTIFACTS", "1") == "1":
        art = wandb.Artifact(f"run-{cfg.run_name}", type="benchmark_run")
        art.add_dir(str(local_dir))
        run.log_artifact(art)

    run.finish()
    return run
