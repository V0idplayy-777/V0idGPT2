"""Throughput + peak-RSS benchmark for each model size (dev tool).

Env: VT_ONLY=name|all  VT_T=256|128  VT_REMAT=1|0  VT_B=override batch
"""
import sys
import os
import time
import resource
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "common"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jax
import jax.numpy as jnp
import model as M
from configs import CONFIGS

ONLY = os.environ.get("VT_ONLY", "all")
T_OVERRIDE = int(os.environ.get("VT_T", "0"))
USE_REMAT = os.environ.get("VT_REMAT", "1") == "1"
B_OVERRIDE = int(os.environ.get("VT_B", "0"))
BATCHES = {"potato": [48], "toaster": [16], "microwave": [8],
           "blender": [4], "nuclearfridge": [2]}


def peak_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


print(f"remat={USE_REMAT} T_override={T_OVERRIDE} B_override={B_OVERRIDE}", flush=True)
for name, blist in BATCHES.items():
    if ONLY != "all" and ONLY != name:
        continue
    cfg = CONFIGS[name]
    T = T_OVERRIDE or cfg.ctx
    params = init_params = M.init_params(jax.random.PRNGKey(0), cfg)
    if USE_REMAT:
        block_fn = M.make_block_fn(cfg)
    else:
        # same math, no checkpointing
        _inner = M.make_block_fn.__wrapped__ if hasattr(M.make_block_fn, "__wrapped__") else None

        def block_fn(p, x, _cfg=cfg):
            B, TT, d = x.shape
            h, hd = _cfg.n_head, _cfg.d_head
            h0 = M.rms_norm(x, p["attn_norm"], _cfg.rms_eps)
            q = jnp.matmul(h0, p["wq"]).reshape(B, TT, h, hd).transpose(0, 2, 1, 3)
            k = jnp.matmul(h0, p["wk"]).reshape(B, TT, h, hd).transpose(0, 2, 1, 3)
            v = jnp.matmul(h0, p["wv"]).reshape(B, TT, h, hd).transpose(0, 2, 1, 3)
            q, k = M.rope(q, _cfg.rope_theta), M.rope(k, _cfg.rope_theta)
            scores = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * (hd ** -0.5)
            mask = jnp.triu(jnp.ones((TT, TT), dtype=jnp.float32), k=1) * -1e10
            o = jnp.matmul(jax.nn.softmax(scores + mask[None, None, :, :], axis=-1), v)
            o = o.transpose(0, 2, 1, 3).reshape(B, TT, d)
            x = x + jnp.matmul(o, p["wo"])
            h1 = M.rms_norm(x, p["mlp_norm"], _cfg.rms_eps)
            x = x + jnp.matmul(jax.nn.silu(jnp.matmul(h1, p["wgate"])) * jnp.matmul(h1, p["wup"]), p["wdown"])
            return x

    @jax.jit
    def step(p, inp, tgt):
        l, g = jax.value_and_grad(M.loss_fn)(p, inp, tgt, None, cfg, block_fn)
        return l, jax.tree_util.tree_map(lambda x: x * 1.0, g)

    for B in ([B_OVERRIDE] if B_OVERRIDE else blist):
        try:
            rng = jax.random.PRNGKey(1)
            inp = jax.random.randint(rng, (B, T), 0, cfg.vocab)
            tgt = jax.random.randint(rng, (B, T), 0, cfg.vocab)
            l, _ = step(params, inp, tgt)
            l.block_until_ready()
            t0 = time.time()
            N = 2
            for _ in range(N):
                l, _ = step(params, inp, tgt)
            l.block_until_ready()
            dt = (time.time() - t0) / N
            print(f"{name}: B={B} T={T} step={dt:.2f}s toks/s={B*T/dt:.0f} peakRSS={peak_gb():.2f}GB", flush=True)
        except Exception as e:
            print(f"{name}: B={B} T={T} FAILED {type(e).__name__}: {str(e)[:150]}", flush=True)
    del params
