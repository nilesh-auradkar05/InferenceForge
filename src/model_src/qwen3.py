"""Qwen3 forward pass, from scratch, on HuggingFace weights. Phase 1.
 
Nothing here calls a `transformers` module at runtime -- we borrow the weights
and rebuild the computation. That is deliberate: paged attention, prefix reuse,
custom kernels, and speculative verification all require owning this loop.
 
Architecture notes that bite people:
  * Qwen3 applies a per-head RMSNorm to q and k (`q_norm`, `k_norm`) AFTER the
    projection and reshape but BEFORE RoPE. Skip it and your logits are subtly
    wrong -- greedy decoding often still looks fluent, which is why this bug
    survives for days.
  * GQA: `num_key_value_heads` < `num_attention_heads`. K/V are expanded by
    repeat_interleave to match Q. This is also exactly why the KV cache is
    smaller than you would naively expect.
  * No biases on any projection. `tie_word_embeddings` is often True on small
    Qwen3 variants -- check it or lm_head will be garbage.
"""

from __future__ import annotations
 
import math
from dataclasses import dataclass
 
import torch
import torch.nn.functional as F
 
 
@dataclass(slots=True)
class ModelDims:
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    hidden: int
    vocab: int
    rms_eps: float
    rope_theta: float
    tie_embeddings: bool
 
    @property
    def kv_group_size(self) -> int:
        return self.n_heads // self.n_kv_heads
 
    def kv_bytes_per_token(self, dtype_bytes: int = 2) -> int:
        """The number that decides how many sequences fit in memory."""
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * dtype_bytes
 
 
def dims_from_hf_config(conf) -> ModelDims:
    head_dim = getattr(conf, "head_dim", None) or conf.hidden_size // conf.num_attention_heads
    rope = getattr(conf, "rope_parameters", None) or {}
    theta = rope.get("rope_theta") if isinstance(rope, dict) else None
    if theta is None:
        theta = getattr(conf, "rope_theta", 10000.0)
    return ModelDims(
        n_layers=conf.num_hidden_layers,
        n_heads=conf.num_attention_heads,
        n_kv_heads=getattr(conf, "num_key_value_heads", conf.num_attention_heads),
        head_dim=head_dim,
        hidden=conf.hidden_size,
        vocab=conf.vocab_size,
        rms_eps=getattr(conf, "rms_norm_eps", 1e-6),
        rope_theta=float(theta),
        tie_embeddings=bool(getattr(conf, "tie_word_embeddings", False)),
    )
 
 
def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    # Computed in fp32 then cast back -- matching HF exactly. Doing this in
    # bf16 introduces drift that compounds across 36 layers.
    dtype = x.dtype
    x = x.float()
    x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return (x.to(dtype)) * weight
 
 
class RopeTable:
    """Precomputed cos/sin. Built once at setup, indexed by absolute position.
 
    Recomputing this per step is a classic hidden cost: it is small FLOPs but
    it is a CPU-side tensor construction inside your decode loop, which shows
    up as launch overhead at exactly the moment you are memory-bound.
    """
 
    def __init__(self, head_dim: int, max_len: int, theta: float, device, dtype):
        inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        pos = torch.arange(max_len, device=device).float()
        freqs = torch.outer(pos, inv)              # [max_len, head_dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)    # [max_len, head_dim]
        self.cos = emb.cos().to(dtype)
        self.sin = emb.sin().to(dtype)
 
    def get(self, start: int, length: int):
        return self.cos[start : start + length], self.sin[start : start + length]
 
    def gather(self, position_ids: torch.Tensor):
        """position_ids: [B, S] -> cos/sin of [B, S, D].
 
        Needed for left-padded batches: every row sits at a different absolute
        position, so a single shared slice would rotate the short sequences to
        the wrong angle. This bug does not crash -- it quietly degrades output.
        """
        return self.cos[position_ids], self.sin[position_ids]
 
 
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)
 
 
def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """q: [B, H, S, D]. cos/sin: [S, D] (shared) or [B, S, D] (per-row)."""
    if cos.dim() == 2:
        cos, sin = cos[None, None], sin[None, None]
    else:
        cos, sin = cos[:, None], sin[:, None]
    return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin
 
 
class LayerWeights:
    __slots__ = (
        "q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm",
        "gate_proj", "up_proj", "down_proj", "input_ln", "post_attn_ln",
    )
 
    def __init__(self, sd: dict, i: int):
        p = f"model.layers.{i}."
        self.q_proj = sd[p + "self_attn.q_proj.weight"]
        self.k_proj = sd[p + "self_attn.k_proj.weight"]
        self.v_proj = sd[p + "self_attn.v_proj.weight"]
        self.o_proj = sd[p + "self_attn.o_proj.weight"]
        self.q_norm = sd[p + "self_attn.q_norm.weight"]
        self.k_norm = sd[p + "self_attn.k_norm.weight"]
        self.gate_proj = sd[p + "mlp.gate_proj.weight"]
        self.up_proj = sd[p + "mlp.up_proj.weight"]
        self.down_proj = sd[p + "mlp.down_proj.weight"]
        self.input_ln = sd[p + "input_layernorm.weight"]
        self.post_attn_ln = sd[p + "post_attention_layernorm.weight"]
 
 
class Qwen3Scratch:
    """Stateless forward. All sequence state lives in the cache object."""
 
    ATTN_IMPLS = ("repeat", "broadcast", "enable_gqa")

    def __init__(self, state_dict: dict, dims: ModelDims, device: str, dtype: torch.dtype,
                 max_position: int = 8192, attn_impl: str = "broadcast"):
        if attn_impl not in self.ATTN_IMPLS:
            raise ValueError(f"attn_impl must be one of {self.ATTN_IMPLS}, got {attn_impl!r}")
        self.attn_impl = attn_impl
        self.d = dims
        self.device = device
        self.dtype = dtype
        sd = {k: v.to(device=device, dtype=dtype) for k, v in state_dict.items()}
        self.embed = sd["model.embed_tokens.weight"]
        self.final_norm = sd["model.norm.weight"]
        self.lm_head = self.embed if dims.tie_embeddings else sd["lm_head.weight"]
        self.layers = [LayerWeights(sd, i) for i in range(dims.n_layers)]
        self.rope = RopeTable(dims.head_dim, max_position, dims.rope_theta, device, dtype)
        self.scale = 1.0 / math.sqrt(dims.head_dim)
 
    def _attend(self, q: torch.Tensor, k_all: torch.Tensor, v_all: torch.Tensor,
                causal: torch.Tensor | None) -> torch.Tensor:
        """q: [B, H, S, D]. k_all/v_all: [B, KVH, T, D]. Returns [B, H, S, D].

        The GQA expansion is the single largest term in the decode step once the
        batch grows. `repeat` materialises a kv_group_size-fold copy of the whole
        gathered history, per layer, per step: at B=26/T=870 that is ~26 GB of
        traffic per step against ~8 GB for the weights themselves. `broadcast`
        regroups Q instead, so K/V are never copied. Kept selectable because the
        winner is a measurement, not an assumption -- see tests/test_attn_impl.py.
        """
        d = self.d
        G = d.kv_group_size
        if G == 1 or self.attn_impl == "repeat":
            if G > 1:
                k_all = k_all.repeat_interleave(G, dim=1)
                v_all = v_all.repeat_interleave(G, dim=1)
            return F.scaled_dot_product_attention(
                q, k_all, v_all, attn_mask=causal, scale=self.scale)

        if self.attn_impl == "enable_gqa":
            # torch >= 2.5. Correct, but several backends implement it by
            # expanding K/V internally, so it is not guaranteed to save traffic.
            return F.scaled_dot_product_attention(
                q, k_all, v_all, attn_mask=causal, scale=self.scale, enable_gqa=True)

        # broadcast: view Q as [B, KVH, G, S, D] and give K/V a singleton group
        # axis. SDPA treats every dim but the last two as batch and broadcasts
        # them, so the singleton costs nothing. Head h of the `repeat` layout
        # reads kv head h // G; q.view groups consecutive G heads, so
        # qg[:, i, g] is head i*G+g -> kv head i. Same mapping, no copy.
        B, H, S, D = q.shape
        qg = q.view(B, d.n_kv_heads, G, S, D)
        kg = k_all.unsqueeze(2)                  # [B, KVH, 1, T, D]
        vg = v_all.unsqueeze(2)
        mask = causal.unsqueeze(2) if causal is not None else None
        out = F.scaled_dot_product_attention(qg, kg, vg, attn_mask=mask, scale=self.scale)
        return out.reshape(B, H, S, D)

    @torch.inference_mode()
    def forward(
        self,
        token_ids: torch.Tensor,
        cache,
        *,
        start_pos: int,
        position_ids: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """token_ids: [B, S]. Returns logits for the LAST position only: [B, vocab].
 
        Returning only the last row is not a shortcut -- during decode the other
        rows are never read, and materializing [B, S, vocab] for a 2048-token
        prefill costs ~600 MB of pure waste at vocab=151k.
        """
        d = self.d
        B, S = token_ids.shape
        h = F.embedding(token_ids, self.embed)
        if position_ids is None:
            cos, sin = self.rope.get(start_pos, S)
        else:
            cos, sin = self.rope.gather(position_ids)
        causal = _attn_mask(S, start_pos, self.device, self.dtype, valid_mask)
 
        for i, lw in enumerate(self.layers):
            residual = h
            x = rms_norm(h, lw.input_ln, d.rms_eps)
 
            q = F.linear(x, lw.q_proj).view(B, S, d.n_heads, d.head_dim)
            k = F.linear(x, lw.k_proj).view(B, S, d.n_kv_heads, d.head_dim)
            v = F.linear(x, lw.v_proj).view(B, S, d.n_kv_heads, d.head_dim)
 
            # Qwen3-specific: per-head RMSNorm before RoPE.
            q = rms_norm(q, lw.q_norm, d.rms_eps)
            k = rms_norm(k, lw.k_norm, d.rms_eps)
 
            q = q.transpose(1, 2)   # [B, H, S, D]
            k = k.transpose(1, 2)   # [B, KVH, S, D]
            v = v.transpose(1, 2)
            q, k = apply_rope(q, k, cos, sin)
 
            k_all, v_all = cache.update(i, k, v, start_pos)
 
            attn = self._attend(q, k_all, v_all, causal)
            attn = attn.transpose(1, 2).reshape(B, S, d.n_heads * d.head_dim)
            h = residual + F.linear(attn, lw.o_proj)
 
            residual = h
            x = rms_norm(h, lw.post_attn_ln, d.rms_eps)
            h = residual + F.linear(F.silu(F.linear(x, lw.gate_proj)) * F.linear(x, lw.up_proj),
                                    lw.down_proj)
 
        h = rms_norm(h[:, -1, :], self.final_norm, d.rms_eps)
        return F.linear(h, self.lm_head)
 
 
def _attn_mask(S: int, start_pos: int, device, dtype,
               valid_mask: torch.Tensor | None) -> torch.Tensor | None:
    """Additive mask over the cached span [0, start_pos + S).
 
    Combines causality with per-row validity. `valid_mask` is [B, total] and
    marks which cache slots hold real tokens -- left padding writes garbage into
    the low positions, and without this every row attends to its neighbours'
    padding. Returns None only for the unpadded single-token decode case, where
    every position is legitimately visible.
    """
    total = start_pos + S
    if S == 1 and valid_mask is None:
        return None
    q_idx = torch.arange(start_pos, total, device=device).unsqueeze(1)
    k_idx = torch.arange(total, device=device).unsqueeze(0)
    allowed = k_idx <= q_idx                                  # [S, total]
    if valid_mask is not None:
        allowed = allowed.unsqueeze(0) & valid_mask[:, None, :]  # [B, S, total]
        allowed = allowed.unsqueeze(1)                           # [B, 1, S, total]
    else:
        allowed = allowed[None, None, :, :]
    mask = torch.zeros(allowed.shape, device=device, dtype=dtype)
    return mask.masked_fill_(~allowed, torch.finfo(dtype).min)
 