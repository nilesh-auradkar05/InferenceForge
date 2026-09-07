"""Deterministic workload traces.

The same WorkloadSpec + seed must produce byte-identical token ids and identical
arrival offsets on every machine, forever.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from configs.config import LoadMode, LoadSpec, PromptSource, WorkloadSpec


@dataclass(slots=True)
class Request:
    req_id: int
    prompt_token_ids: list[int]
    output_len: int
    #: Seconds after trace start at which this request should be *submitted*.
    #: Only meaningful in open-loop mode.
    arrival_offset_s: float = 0.0
    is_warmup: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)


def _lognormal_ints(rng, mean: float, std: float, lo: int, hi: int, n: int) -> np.ndarray:
    if std <= 0:
        return np.full(n, int(mean), dtype=np.int64)
    # Solve lognormal params so the *arithmetic* mean/std match what was asked.
    var = std**2
    mu = np.log(mean**2 / np.sqrt(var + mean**2))
    sigma = np.sqrt(np.log(1.0 + var / mean**2))
    vals = rng.lognormal(mu, sigma, size=n)
    return np.clip(np.rint(vals), lo, hi).astype(np.int64)


def build_trace(
    workload: WorkloadSpec,
    load: LoadSpec,
    *,
    vocab_size: int,
    corpus_token_ids: list[int] | None = None,
) -> list[Request]:
    """Build the full trace, warmup requests first.

    Args:
        vocab_size: needed for synthetic prompts; ids are drawn below this and
            above a small reserved band to avoid special tokens.
        corpus_token_ids: flat token stream, required for PromptSource.corpus.
    """
    rng = np.random.default_rng(workload.seed)
    total = workload.num_requests + load.warmup_requests

    prompt_lens = _lognormal_ints(
        rng, workload.prompt_len_mean, workload.prompt_len_std,
        workload.prompt_len_min, workload.prompt_len_max, total,
    )
    output_lens = _lognormal_ints(
        rng, workload.output_len_mean, workload.output_len_std,
        workload.output_len_min, workload.output_len_max, total,
    )

    lo_id = 100  # skip the special-token band most tokenizers reserve
    hi_id = max(lo_id + 1, vocab_size - 1)

    # One shared prefix for the whole trace, sized off the mean prompt length.
    prefix_len = int(workload.shared_prefix_frac * workload.prompt_len_mean)
    if workload.prompt_source is PromptSource.synthetic:
        shared_prefix = rng.integers(lo_id, hi_id, size=prefix_len).tolist()
    else:
        if not corpus_token_ids:
            raise ValueError("PromptSource.corpus requires corpus_token_ids")
        shared_prefix = list(corpus_token_ids[:prefix_len])

    if load.mode is LoadMode.open:
        gaps = rng.exponential(1.0 / float(load.rate_rps), size=total)  # pyright: ignore[reportArgumentType]
        arrivals = np.cumsum(gaps)
        arrivals = arrivals - arrivals[0]  # first request at t=0
    else:
        arrivals = np.zeros(total, dtype=np.float64)

    requests: list[Request] = []
    corpus_len = len(corpus_token_ids) if corpus_token_ids else 0
    for i in range(total):
        plen = int(prompt_lens[i])
        suffix_len = max(1, plen - prefix_len)
        if workload.prompt_source is PromptSource.synthetic:
            suffix = rng.integers(lo_id, hi_id, size=suffix_len).tolist()
        else:
            start = int(rng.integers(0, max(1, corpus_len - suffix_len)))
            suffix = list(corpus_token_ids[start : start + suffix_len])  # pyright: ignore[reportOptionalSubscript]
            if len(suffix) < suffix_len:  # wrap
                suffix += list(corpus_token_ids[: suffix_len - len(suffix)])  # pyright: ignore[reportOptionalSubscript]
        requests.append(
            Request(
                req_id=i,
                prompt_token_ids=shared_prefix + suffix,
                output_len=int(output_lens[i]),
                arrival_offset_s=float(arrivals[i]),
                is_warmup=i < load.warmup_requests,
                meta={"shared_prefix_len": prefix_len},
            )
        )
    return requests


def trace_summary(requests: list[Request]) -> dict:
    measured = [r for r in requests if not r.is_warmup]
    plens = np.array([r.prompt_len for r in measured])
    olens = np.array([r.output_len for r in measured])
    return {
        "n_measured": len(measured),
        "n_warmup": len(requests) - len(measured),
        "prompt_len_mean": float(plens.mean()),
        "prompt_len_p99": float(np.percentile(plens, 99)),
        "output_len_mean": float(olens.mean()),
        "total_prompt_tokens": int(plens.sum()),
        "total_output_tokens": int(olens.sum()),
    }
