"""B1 -- the honest reference: HuggingFace `generate()` with defaults.
 
Every speedup is measured against THIS. B1 is the thing you actually have to beat.
 
Timestamping trick: a LogitsProcessor is invoked exactly once per generation
step, immediately after the forward pass that produced those logits. Stamping
there gives per-token timing without a text streamer (which would fold
detokenization into your ITL).
 
Known caveat, stated rather than hidden: `sync_per_token=True` inserts a
`cuda.synchronize()` every decode step. That is required for an honest ITL
distribution, but it also destroys CPU/GPU overlap -- which is exactly the thing
item 7 on the list is about. So always report BOTH the synced ITL distribution
and the sync-free end-to-end throughput. When you reach the CPU/GPU sync work,
flip this off and compare.
"""

from __future__ import annotations

import threading
import time
from typing import Iterator

from configs.config import EngineConfig
from src.utils.engine import Engine, Event
from src.utils.trace import Request



_DTYPES = {"float16": "float16", "bfloat16": "bfloat16", "float32": "float32"}

class HFBaselineEngine(Engine):
    name = "b1_hf_generate"

    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.model = None
        self._vocab = 0
        self._lock = threading.Lock()
        self.sync_per_token: bool = bool(cfg.extra.get("sync_per_token", True))
        self._n_forwards = 0

    def setup(self) -> None:
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM

        dtype = getattr(torch, _DTYPES[self.cfg.dtype.value])
        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_id, dtype=dtype, attn_implementation="sdpa",
        ).to(self.cfg.device).eval()  # pyright: ignore[reportArgumentType]
        conf = AutoConfig.from_pretrained(self.cfg.model_id)
        self._vocab = int(getattr(conf, "vocab_size", self.model.config.vocab_size))

    @property
    def vocab_size(self) -> int:
        return self._vocab

    def _make_stamper(self, stamps: list[float]):
        import torch
        from transformers import LogitsProcessor

        device = self.cfg.device
        sync = self.sync_per_token

        class Stamp(LogitsProcessor):
            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
                if sync and device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.synchronize()
                stamps.append(time.perf_counter())
                return scores

        return Stamp()

    def stream(self, req: Request) -> Iterator[Event]:
        import torch
        from transformers import LogitsProcessorList

        # Serialized on purpose. B1 has no batching -- concurrent requests queue.
        # Watching the queue term dominate TTFT is the whole point of item 2.
        with self._lock:
            yield Event(kind="scheduled", t=time.perf_counter())

            ids = torch.tensor([req.prompt_token_ids], dtype=torch.long, device=self.cfg.device)
            attn = torch.ones_like(ids)
            n = req.output_len
            stamps: list[float] = []

            with torch.inference_mode():
                out = self.model.generate(
                    input_ids=ids,
                    attention_mask=attn,
                    do_sample=not self.cfg.greedy,
                    min_new_tokens=n,
                    max_new_tokens=n,
                    use_cache=True,
                    logits_processor=LogitsProcessorList([self._make_stamper(stamps)]),
                    pad_token_id=self.model.config.eos_token_id
                    if getattr(self.model.config, "eos_token_id", None) is not None
                    else 0,
                )
                if self.cfg.device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.synchronize()

            end_t = time.perf_counter()
            self._n_forwards += n

            gen = out[0, ids.shape[1]:].tolist()
            for i, tok in enumerate(gen):
                t = stamps[i] if i < len(stamps) else end_t
                yield Event(kind="token", t=t, token_id=int(tok), index=1)
            yield Event(kind="done", t=end_t)

    def reset_stats(self) -> None:
        self._n_forwards = 0

    def stats(self) -> dict:
        return {"forward_steps": self._n_forwards, "sync_per_token": self.sync_per_token}

    def teardown(self) -> None:
        import torch

        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
