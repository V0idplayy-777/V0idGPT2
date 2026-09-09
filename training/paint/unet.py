"""Tiny conditional U-Net diffusion model (Paint.exe) in JAX.

Image: 32x32x3. Channels (32, 64, 128), ResNet blocks (1, 1) + mid (1 block
+ 4-head attention + 1 block). Conditioning: byte-level text encoder
(256 x 32 embed, mean pool, MLP -> 128-d) + sinusoidal time embedding,
injected via FiLM scale/shift in every ResBlock. Activations NHWC to match
the TypeScript implementation exactly.

Param layout: flat dict[str, array]. Conv kernels OIHW, linears [in, out].
"""
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

GROUPS = 8
GN_EPS = 1e-5


@dataclass
class PaintConfig:
    ch: tuple = (32, 64, 128)
    n_down: tuple = (1, 1)
    mid_blocks: int = 1
    heads: int = 4
    byte_dim: int = 32
    text_dim: int = 128
    time_dim: int = 128
    t_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    img: int = 32
    max_text: int = 64


def count_paint_params(cfg: PaintConfig, params) -> int:
    return sum(int(np.size) for np in
               jax.tree_util.tree_leaves(params))


def _n(rng, shape):
    return jax.random.normal(rng, shape, dtype=jnp.float32) * 0.02


def init_paint_params(rng, cfg: PaintConfig):
    c0, c1, c2 = cfg.ch
    td = cfg.text_dim + cfg.time_dim
    keys = iter(jax.random.split(rng, 500))
    p = {}
    p["byte_emb"] = _n(next(keys), (256, cfg.byte_dim))
    p["txt_mlp1"] = _n(next(keys), (cfg.byte_dim, cfg.text_dim))
    p["txt_mlp1_b"] = jnp.zeros((cfg.text_dim,), jnp.float32)
    p["txt_mlp2"] = _n(next(keys), (cfg.text_dim, cfg.text_dim))
    p["txt_mlp2_b"] = jnp.zeros((cfg.text_dim,), jnp.float32)
    p["null_cond"] = jnp.zeros((cfg.text_dim,), jnp.float32)
    p["t_mlp1"] = _n(next(keys), (64, cfg.time_dim))
    p["t_mlp1_b"] = jnp.zeros((cfg.time_dim,), jnp.float32)
    p["t_mlp2"] = _n(next(keys), (cfg.time_dim, cfg.time_dim))
    p["t_mlp2_b"] = jnp.zeros((cfg.time_dim,), jnp.float32)
    p["stem"] = _n(next(keys), (c0, 3, 3, 3))

    def res(prefix, cin, cout):
        p[prefix + ".conv1"] = _n(next(keys), (cout, cin, 3, 3))
        p[prefix + ".gn1_w"] = jnp.ones((cin,), jnp.float32)
        p[prefix + ".gn1_b"] = jnp.zeros((cin,), jnp.float32)
        p[prefix + ".conv2"] = _n(next(keys), (cout, cout, 3, 3))
        p[prefix + ".gn2_w"] = jnp.ones((cout,), jnp.float32)
        p[prefix + ".gn2_b"] = jnp.zeros((cout,), jnp.float32)
        p[prefix + ".film_w"] = _n(next(keys), (td, 2 * cout))
        p[prefix + ".film_b"] = jnp.zeros((2 * cout,), jnp.float32)
        if cin != cout:
            p[prefix + ".res"] = _n(next(keys), (cout, cin, 1, 1))

    for i in range(cfg.n_down[0]):
        res(f"dn0.b{i}", c0, c0)
    p["dn0.down"] = _n(next(keys), (c1, c0, 3, 3))
    for i in range(cfg.n_down[1]):
        res(f"dn1.b{i}", c1, c1)
    p["dn1.down"] = _n(next(keys), (c2, c1, 3, 3))
    for i in range(cfg.mid_blocks):
        res(f"mid.b{i}", c2, c2)
    p["mid.attn.gn_w"] = jnp.ones((c2,), jnp.float32)
    p["mid.attn.gn_b"] = jnp.zeros((c2,), jnp.float32)
    p["mid.attn.pos"] = jnp.zeros((64, c2), jnp.float32)
    p["mid.attn.wq"] = _n(next(keys), (c2, c2))
    p["mid.attn.wk"] = _n(next(keys), (c2, c2))
    p["mid.attn.wv"] = _n(next(keys), (c2, c2))
    p["mid.attn.wo"] = _n(next(keys), (c2, c2))
    for i in range(cfg.mid_blocks):
        res(f"mid.c{i}", c2, c2)
    # up: concat skips; conv1 takes concatenated channels
    res("up1.b0", c2 + c1, c1)
    res("up0.b0", c1 + c0, c0)
    p["out_gn_w"] = jnp.ones((c0,), jnp.float32)
    p["out_gn_b"] = jnp.zeros((c0,), jnp.float32)
    p["out_conv"] = jnp.zeros((3, c0, 3, 3), jnp.float32)
    return p


# ------------------------------------------------------------------ ops
def conv(x, k, stride=1):
    return lax.conv_general_dilated(
        x, k, (stride, stride), "SAME",
        dimension_numbers=("NHWC", "OIHW", "NHWC"))


def group_norm(x, w, b, groups=GROUPS, eps=GN_EPS):
    B, H, W, C = x.shape
    g = groups
    xg = x.reshape(B, H, W, g, C // g)
    mu = jnp.mean(xg, axis=(1, 2, 4), keepdims=True)
    va = jnp.var(xg, axis=(1, 2, 4), keepdims=True)
    xg = (xg - mu) * jax.lax.rsqrt(va + eps)
    x = xg.reshape(B, H, W, C)
    return x * w[None, None, None, :] + b[None, None, None, :]


def res_block(p, prefix, x, temb):
    h = group_norm(x, p[prefix + ".gn1_w"], p[prefix + ".gn1_b"])
    h = jax.nn.silu(h)
    h = conv(h, p[prefix + ".conv1"])
    h = group_norm(h, p[prefix + ".gn2_w"], p[prefix + ".gn2_b"])
    film = jnp.matmul(temb, p[prefix + ".film_w"]) + p[prefix + ".film_b"]
    sc, sh = jnp.split(film, 2, axis=-1)
    h = h * (1.0 + sc[:, None, None, :]) + sh[:, None, None, :]
    h = jax.nn.silu(h)
    h = conv(h, p[prefix + ".conv2"])
    r = x if prefix + ".res" not in p else conv(x, p[prefix + ".res"])
    return r + h


def attn_block(p, x, heads):
    B, H, W, C = x.shape
    hd = C // heads
    h = group_norm(x, p["mid.attn.gn_w"], p["mid.attn.gn_b"])
    seq = h.reshape(B, H * W, C) + p["mid.attn.pos"][None, :, :]
    q = jnp.matmul(seq, p["mid.attn.wq"]).reshape(B, H * W, heads, hd).transpose(0, 2, 1, 3)
    k = jnp.matmul(seq, p["mid.attn.wk"]).reshape(B, H * W, heads, hd).transpose(0, 2, 1, 3)
    v = jnp.matmul(seq, p["mid.attn.wv"]).reshape(B, H * W, heads, hd).transpose(0, 2, 1, 3)
    att = jax.nn.softmax(jnp.matmul(q, k.transpose(0, 1, 3, 2)) * (hd ** -0.5), axis=-1)
    o = jnp.matmul(att, v).transpose(0, 2, 1, 3).reshape(B, H * W, C)
    o = jnp.matmul(o, p["mid.attn.wo"]).reshape(B, H, W, C)
    return x + o


def upsample(x):
    return jnp.repeat(jnp.repeat(x, 2, axis=1), 2, axis=2)


def time_embed(t, dim=64):
    t = t.astype(jnp.float32)
    half = dim // 2
    div = jnp.exp(jnp.arange(half, dtype=jnp.float32) * -(jnp.log(10000.0) / half))
    ang = t[:, None] * div[None, :]
    return jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)


def encode_text(p, cfg: PaintConfig, byte_ids, mask):
    e = p["byte_emb"][byte_ids]  # [B, L, 32]
    e = e * mask[..., None]
    pooled = e.sum(axis=1) / jnp.maximum(mask.sum(axis=1, keepdims=True), 1.0)
    h = jax.nn.silu(jnp.matmul(pooled, p["txt_mlp1"]) + p["txt_mlp1_b"])
    return jnp.matmul(h, p["txt_mlp2"]) + p["txt_mlp2_b"]


def apply_unet(p, cfg: PaintConfig, x, t, cond):
    te = time_embed(t)
    te = jax.nn.silu(jnp.matmul(te, p["t_mlp1"]) + p["t_mlp1_b"])
    te = jnp.matmul(te, p["t_mlp2"]) + p["t_mlp2_b"]
    temb = jnp.concatenate([te, cond], axis=-1)
    h = conv(x, p["stem"])
    skips = []
    for i in range(cfg.n_down[0]):
        h = res_block(p, f"dn0.b{i}", h, temb)
    skips.append(h)
    h = conv(h, p["dn0.down"], stride=2)
    for i in range(cfg.n_down[1]):
        h = res_block(p, f"dn1.b{i}", h, temb)
    skips.append(h)
    h = conv(h, p["dn1.down"], stride=2)
    for i in range(cfg.mid_blocks):
        h = res_block(p, f"mid.b{i}", h, temb)
    h = attn_block(p, h, cfg.heads)
    for i in range(cfg.mid_blocks):
        h = res_block(p, f"mid.c{i}", h, temb)
    h = upsample(h)
    h = jnp.concatenate([h, skips.pop()], axis=-1)
    h = res_block(p, "up1.b0", h, temb)
    h = upsample(h)
    h = jnp.concatenate([h, skips.pop()], axis=-1)
    h = res_block(p, "up0.b0", h, temb)
    h = group_norm(h, p["out_gn_w"], p["out_gn_b"])
    h = jax.nn.silu(h)
    return conv(h, p["out_conv"])


# ------------------------------------------------------------- diffusion
def schedule(cfg: PaintConfig):
    betas = np.linspace(cfg.beta_start, cfg.beta_end, cfg.t_steps).astype(np.float64)
    alphas = 1.0 - betas
    abar = np.cumprod(alphas)
    return betas.astype(np.float32), abar.astype(np.float32)


def ddim_times(steps, t_steps=1000):
    import numpy as _np
    return _np.linspace(0, t_steps - 1, steps).round().astype(int)


import numpy as np  # noqa: E402  (kept late to group jax imports first)
