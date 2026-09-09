"""Decoder-only Transformer LM (LLaMA-style) in JAX.

Architecture (identical math is re-implemented in TypeScript for the browser):
  - byte-level BPE embeddings, TIED input/output embeddings
  - RoPE positional encoding (NeoX/halves style), theta=10000
  - pre-norm transformer blocks: RMSNorm -> MHA (causal) -> residual ->
    RMSNorm -> SwiGLU MLP -> residual
  - NO biases anywhere, NO dropout (tiny models, limited data)
  - final RMSNorm -> logits via embedding transpose

Params layout: flat dict[str, jnp.ndarray] with keys:
  tok_emb, final_norm,
  L{i}.attn_norm, L{i}.wq, L{i}.wk, L{i}.wv, L{i}.wo,
  L{i}.mlp_norm, L{i}.wgate, L{i}.wup, L{i}.wdown
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass
class LMConfig:
    name: str
    vocab: int = 4096
    d_model: int = 128
    n_layer: int = 5
    n_head: int = 4
    d_ff: int = 336
    ctx: int = 256
    rope_theta: float = 10000.0
    rms_eps: float = 1e-6

    @property
    def d_head(self):
        assert self.d_model % self.n_head == 0
        return self.d_model // self.n_head


def count_params(cfg: LMConfig) -> int:
    d, h, ff, L, V = cfg.d_model, cfg.n_head, cfg.d_ff, cfg.n_layer, cfg.vocab
    per_layer = 4 * d * d + 3 * d * ff + 2 * d  # qkv o + gate/up/down + 2 norms
    return V * d + L * per_layer + d  # tied emb + layers + final norm


def init_params(rng, cfg: LMConfig):
    d, L, V, ff = cfg.d_model, cfg.n_layer, cfg.vocab, cfg.d_ff
    # Split enough keys: emb + L*8 matrices. Norms are ones (no rng needed).
    keys = jax.random.split(rng, 1 + L * 8)
    p = {}
    p["tok_emb"] = jax.random.normal(keys[0], (V, d), dtype=jnp.float32) * 0.02
    p["final_norm"] = jnp.ones((d,), dtype=jnp.float32)
    k = 1
    for i in range(L):
        pre = f"L{i}."
        p[pre + "attn_norm"] = jnp.ones((d,), dtype=jnp.float32)
        p[pre + "mlp_norm"] = jnp.ones((d,), dtype=jnp.float32)
        p[pre + "wq"] = jax.random.normal(keys[k], (d, d), dtype=jnp.float32) * 0.02; k += 1
        p[pre + "wk"] = jax.random.normal(keys[k], (d, d), dtype=jnp.float32) * 0.02; k += 1
        p[pre + "wv"] = jax.random.normal(keys[k], (d, d), dtype=jnp.float32) * 0.02; k += 1
        p[pre + "wo"] = jax.random.normal(keys[k], (d, d), dtype=jnp.float32) * 0.02; k += 1
        p[pre + "wgate"] = jax.random.normal(keys[k], (d, ff), dtype=jnp.float32) * 0.02; k += 1
        p[pre + "wup"] = jax.random.normal(keys[k], (d, ff), dtype=jnp.float32) * 0.02; k += 1
        p[pre + "wdown"] = jax.random.normal(keys[k], (ff, d), dtype=jnp.float32) * 0.02; k += 1
        # 8th key unused (kept for stable key schedule)
        k += 1
    return p


def rms_norm(x, w, eps):
    return x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps) * w


def rope(x, theta):
    """x: [B, H, T, hd] -> rotated. Positions 0..T-1. NeoX (halves) style."""
    B, H, T, hd = x.shape
    half = hd // 2
    pos = jnp.arange(T, dtype=jnp.float32)
    freqs = 1.0 / (theta ** (jnp.arange(half, dtype=jnp.float32) * 2.0 / hd))
    ang = pos[:, None] * freqs[None, :]
    sin = jnp.sin(ang)[None, None, :, :]
    cos = jnp.cos(ang)[None, None, :, :]
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def make_block_fn(cfg: LMConfig):
    d, h, hd = cfg.d_model, cfg.n_head, cfg.d_head
    scale = hd ** -0.5

    def block(p, x):
        """p: layer param dict, x: [B, T, d] -> [B, T, d]."""
        B, T, _ = x.shape
        h0 = rms_norm(x, p["attn_norm"], cfg.rms_eps)
        q = jnp.matmul(h0, p["wq"]).reshape(B, T, h, hd).transpose(0, 2, 1, 3)
        k = jnp.matmul(h0, p["wk"]).reshape(B, T, h, hd).transpose(0, 2, 1, 3)
        v = jnp.matmul(h0, p["wv"]).reshape(B, T, h, hd).transpose(0, 2, 1, 3)
        q, k = rope(q, cfg.rope_theta), rope(k, cfg.rope_theta)
        scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * scale
        mask = jnp.triu(jnp.ones((T, T), dtype=jnp.float32), k=1) * -1e10
        scores = scores + mask[None, None, :, :]
        attn = jax.nn.softmax(scores, axis=-1)
        o = jnp.matmul(attn, v).transpose(0, 2, 1, 3).reshape(B, T, d)
        x = x + jnp.matmul(o, p["wo"])
        h1 = rms_norm(x, p["mlp_norm"], cfg.rms_eps)
        x = x + jnp.matmul(jax.nn.silu(jnp.matmul(h1, p["wgate"])) * jnp.matmul(h1, p["wup"]), p["wdown"])
        return x

    return jax.remat(block)


def forward(params, ids, cfg: LMConfig, block_fn=None):
    """ids: [B, T] int32 -> logits [B, T, V]."""
    if block_fn is None:
        block_fn = make_block_fn(cfg)
    x = params["tok_emb"][ids]
    for i in range(cfg.n_layer):
        pre = f"L{i}."
        p = {k: params[pre + k] for k in
             ("attn_norm", "wq", "wk", "wv", "wo", "mlp_norm", "wgate", "wup", "wdown")}
        x = block_fn(p, x)
    x = rms_norm(x, params["final_norm"], cfg.rms_eps)
    return jnp.matmul(x, params["tok_emb"].T)


def loss_fn(params, inputs, targets, weights, cfg, block_fn):
    logits = forward(params, inputs, cfg, block_fn)
    logp = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]
    w = weights if weights is not None else jnp.ones_like(nll)
    return jnp.sum(nll * w) / jnp.maximum(jnp.sum(w), 1.0)
