from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator
from torch._C import _get_cpu_capability

SCHEMA_VERSION = "0.1.0"

class Dtype(str, Enum):
    float16 = "float16"
    bfloat16 = "bfloat16"
    float32 = "float32"

class LoadMode(str, Enum):
    closed = "closed"
    open = "open"

class PromptSource(str, Enum):
    synthetic = "synthetic"
    corpus = "corpus"

class WorkloadSpec(BaseModel):
    """The thing that must never change between runs."""

    model_config = {"frozen": True}

    name: str = Field(description="Human label, e.g. 'chat_short'")

    num_requests: int = Field(gt=0)
    seed: int = 47

    prompt_source: PromptSource = PromptSource.synthetic
    prompt_len_mean: int = Field(gt=0, description="Tokens, lognormal mean")
    prompt_len_std: float = Field(ge=0.0, default=1.0)
    prompt_len_min: int = Field(gt=0, default=8)
    prompt_len_max: int = Field(gt=0, default=8192)

    # Fraction of every prompt that is a shared system prefix. Set > 0 for prefix caching.
    shared_prefix_frac: float = Field(ge=0.0, le=0.9, default=0.0)

    output_len_mean: int = Field(gt=0)
    output_len_std: float = Field(ge=0.0, default=0.0)
    output_len_min: int = Field(gt=0, default=1)
    output_len_max: int = Field(gt=0, default=2048)

    # Restricted to generating only till output_len tokens, ignoring EOS.
    force_exact_output_len: bool = True

    @model_validator(mode="after")
    def _check_bounds(self) -> "WorkloadSpec":
        if self.prompt_len_min > self.prompt_len_max:
            raise ValueError("prompt_len_min > prompt_len_max")
        if self.output_len_min > self.output_len_max:
            raise ValueError("output_len_min > output_len_max")
        
        return self

    def fingerprint(self) -> str:
        blob = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

class LoadSpec(BaseModel):
    model_config = {"frozen": True}

    mode: LoadMode = LoadMode.closed
    # closed mode: number of concurrent in-flight requests.
    concurrency: int = Field(gt=0, default=1)
    # Open mode: Poisson arrival rate in requests/second.
    rate_rps: float | None = Field(gt=0.0, default=None)
    # Requests discarded from the head of the trace before measuring.
    warmup_requests: int = Field(ge=0, default=8)

    @model_validator(mode="after")
    def _check_mode(self) -> "LoadSpec":
        if self.mode is LoadMode.open and self.rate_rps is None:
            raise ValueError("open-loop load requires rate_rps")
        return self

class SLOSpec(BaseModel):
    model_config = {"frozen": True}

    ttft_p99_ms: float = Field(gt=0, default=1000.0)
    tpot_p99_ms: float = Field(gt=0, default=100.0)

class EngineConfig(BaseModel):
    model_config = {"frozen": True}

    # Registry key
    engine: str
    model_id: str = "Qwen/Qwen3-4B-Instruct-2507"
    dtype: Dtype = Dtype.bfloat16
    device: str = "cuda"
    
    # Greedy only decoding."
    greedy: bool = True
    max_batch_size: int = Field(gt=0, default=1)
    extra: dict = Field(default_factory=dict)

class RunConfig(BaseModel):
    schema_version: str = SCHEMA_VERSION
    run_name: str
    phase: str = Field(description="eg. 'p0_baseline', 'p1_kv_cache")
    engine: EngineConfig
    workload: WorkloadSpec
    load: LoadSpec
    slo: SLOSpec = SLOSpec()
    wandb_project: str = "llm_inference_from_scratch"
    wandb_entity: str | None = None

    # Sampling period for GPU utilization / memory telemetry.
    telemetry_interval_s: float = Field(gt=0.0, default=0.25)
    output_dir: str = "runs"

    def comparison_key(self) -> str:
        """Two runs are compariable only if this matches."""
        return f"{self.workload.fingerprint()}|{self.load.mode.value}|{self.load.concurrency}|{self.load.rate_rps}"

class EnvFingerprint(BaseModel):
    """Captured at runtime"""

    python: str
    platform: str
    torch: str | None = None
    cuda: str | None = None
    cudnn: str | None = None
    gpu_name: str | None = None
    gpu_capability: str | None = None
    gpu_count: int = 0
    gpu_mem_total_gb: float | None = None
    driver: str | None = None
    git_sha: str | None = None
    git_dirty: bool = False
    cuda_visible_devices: str | None = None
    torch_device_index: int | None = None
    nvml_physical_device: int | None = None

    @classmethod
    def capture(cls) -> "EnvFingerprint":
        info: dict = {
            "python": platform.python_version(),
            "platform": platform.platform(),
        }
        try:
            import os
            import torch

            info["torch"] = torch.__version__
            info["cuda"] = torch.version.cuda
            info["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
            try:
                info["cudnn"] = str(torch.backends.cudnn.version())
            except Exception:
                info["cudnn"] = None
            if torch.cuda.is_available():
                info["gpu_count"] = torch.cuda.device_count()
                idx = torch.cuda.current_device()
                info["torch_device_index"] = idx
                vis = info["cuda_visible_devices"]
                if vis:
                    ids = [x.strip() for x in vis.split(",") if x.strip() != ""]
                    try:
                        info["nvml_physical_device"] = int(ids[idx])
                    except (ValueError, IndexError):
                        info["nvml_physical_device"] = idx
                else:
                    info["nvml_physical_device"] = idx
                props = torch.cuda.get_device_properties(idx)
                info["gpu_name"] = props.name
                info["gpu_capability"] = f"{props.major}.{props.minor}"
                info["gpu_mem_total_gb"] = round(props.total_memory / 1024**3, 2)
        except Exception:
            pass
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0:
                info["driver"] = out.stdout.strip().splitlines()[0]
        except Exception:
            pass
        try:
            sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
            if sha.returncode == 0:
                info["git_sha"] = sha.stdout.strip()
                st = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5)
                info["git_dirty"] = bool(st.stdout.strip())
        except Exception:
            pass
        return cls(**info)
