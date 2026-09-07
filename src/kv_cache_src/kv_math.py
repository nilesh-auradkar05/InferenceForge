"""Print the KV arithmetic for a model+GPU before you run anything.

    uv run python3 src/kv_cache_src/kv_math.py --model Qwen/Qwen3-4B-Instruct-2507 --gpu-gb 48
    uv run python3 src/kv_cache_src/kv_math.py --model Qwen/Qwen3-4B-Instruct-2507 --peak-alloc-gb 7.64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Running this file directly puts kv_cache_src/ on sys.path, not the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.model_src.qwen3 import ModelDims  # noqa: E402


def kv_kib_per_token(dims: ModelDims, dtype_bytes: int = 2) -> float:
    return dims.kv_bytes_per_token(dtype_bytes) / 1024.0


def weights_gb(
    *,
    peak_alloc_gb: float | None,
    param_count: int | None,
    dtype_bytes: int = 2,
) -> tuple[float, str]:
    """Prefer a measured peak_alloc_gb. Hub safetensors.total is a param count, not bytes."""
    if peak_alloc_gb is not None:
        return float(peak_alloc_gb), "peak_alloc_gb"
    if param_count:
        return param_count * dtype_bytes / 2**30, "param_count"
    return float("nan"), "unknown"


def decode_floor(weight_gb: float, bandwidth_gbs: float) -> tuple[float, float]:
    """Batch-1 memory-bound decode: ms/token and tok/s from weight traffic."""
    seconds = weight_gb * 2**30 / (bandwidth_gbs * 1e9)
    return seconds * 1e3, 1.0 / seconds


def report_lines(
    dims: ModelDims,
    *,
    gpu_gb: float,
    bandwidth_gbs: float,
    tflops: float,
    seq_len: int,
    peak_alloc_gb: float | None,
    param_count: int | None,
    dtype_bytes: int = 2,
) -> list[str]:
    kib = kv_kib_per_token(dims, dtype_bytes)
    wgb, source = weights_gb(
        peak_alloc_gb=peak_alloc_gb, param_count=param_count, dtype_bytes=dtype_bytes
    )
    per_tok = dims.kv_bytes_per_token(dtype_bytes)
    free_gb = gpu_gb - (wgb if wgb == wgb else 8.0) - 2.0
    cap_tokens = int(free_gb * 2**30 // per_tok) if per_tok else 0
    balance = tflops * 1e12 / (bandwidth_gbs * 1e9)

    lines = [
        f"layers={dims.n_layers} q_heads={dims.n_heads} kv_heads={dims.n_kv_heads} head_dim={dims.head_dim}",
        f"KV per token (bf16): {kib:.1f} KiB",
        f"weights ({source}): {wgb:.2f} GiB",
        (
            f"KV headroom         : {free_gb:.1f} GB -> {cap_tokens:,} tokens = "
            f"~{cap_tokens // seq_len if seq_len else 0} concurrent seqs of {seq_len}"
        ),
        f"machine balance      : {balance:.0f} FLOP/byte -> decode is compute-bound only past batch ~{balance:.0f}",
    ]
    if wgb == wgb:
        ms, tps = decode_floor(wgb, bandwidth_gbs)
        lines.append(f"batch-1 decode floor: {ms:.2f} ms/token = {tps:.0f} tok/s")
    return lines


def _hub_param_count(model_id: str) -> int | None:
    try:
        from huggingface_hub import model_info

        info = getattr(model_info(model_id), "safetensors", None)
        total = getattr(info, "total", None) if info is not None else None
        return int(total) if total else None
    except Exception:
        return None


def main(argv: list[str] | None = None) -> None:
    from transformers import AutoConfig

    from src.model_src.qwen3 import dims_from_hf_config

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    p.add_argument("--gpu-gb", type=float, default=48.0)
    p.add_argument("--bandwidth-gbs", type=float, default=960.0, help="RTX 6000 Ada")
    p.add_argument("--tflops", type=float, default=182.0, help="dense bf16")
    p.add_argument("--seq-len", type=int, default=384)
    p.add_argument(
        "--peak-alloc-gb",
        type=float,
        default=None,
        help="Measured torch peak allocated GiB from a bench run. "
        "Preferred over hub metadata for the decode-floor calc.",
    )
    p.add_argument("--dtype-bytes", type=int, default=2)
    a = p.parse_args(argv)

    d = dims_from_hf_config(AutoConfig.from_pretrained(a.model))
    params = _hub_param_count(a.model)
    for line in report_lines(
        d,
        gpu_gb=a.gpu_gb,
        bandwidth_gbs=a.bandwidth_gbs,
        tflops=a.tflops,
        seq_len=a.seq_len,
        peak_alloc_gb=a.peak_alloc_gb,
        param_count=params,
        dtype_bytes=a.dtype_bytes,
    ):
        print(line)


if __name__ == "__main__":
    main()
