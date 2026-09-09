"""Export Paint.exe EMA params -> browser files + parity fixtures.

Writes public/models/paintexe/:
  config.json, weights.f32.bin (magic "V0P1", 16-byte aligned fp32 payloads),
  parity.json (text-cond vector, one UNet forward, one 8-step DDIM run).

Conventions (same spirit as the LM exporter):
  - linears stored OUTPUT-MAJOR ([out, in])
  - conv kernels OIHW, 1-D vectors and pos-embedding as-is.
"""
import argparse
import json
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import jax
import jax.numpy as jnp

from train_paint import ddim_sample, text_to_bytes
from unet import PaintConfig, apply_unet, count_paint_params, ddim_times, encode_text, schedule

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LINEAR_SUFFIX = ("txt_mlp1", "txt_mlp2", "t_mlp1", "t_mlp2",
                 "film_w", "wq", "wk", "wv", "wo")


def export(ckpt_path):
    cfg = PaintConfig()
    _, abar = schedule(cfg)
    z = np.load(ckpt_path)
    params = {k: np.array(z[k], dtype=np.float32) for k in z.files}
    print(f"loaded {ckpt_path}: {len(params)} tensors", flush=True)

    outdir = os.path.join(REPO, "public", "models", "paintexe")
    os.makedirs(outdir, exist_ok=True)

    table = []
    chunks = []
    offset = 0
    for n in sorted(params.keys()):
        w = params[n]
        if w.ndim == 2 and n.split(".")[-1] in LINEAR_SUFFIX:
            w = np.ascontiguousarray(w.T)
        raw = np.ascontiguousarray(w).tobytes()
        table.append({"name": n, "shape": list(w.shape), "dtype": "f32",
                      "offset": offset, "length": len(raw)})
        chunks.append(raw)
        pad = (16 - (len(raw) % 16)) % 16
        chunks.append(b"\x00" * pad)
        offset += len(raw) + pad

    with open(os.path.join(outdir, "weights.f32.bin"), "wb") as f:
        f.write(b"V0P1")
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<II", 0, 0))
        for c in chunks:
            f.write(c)
    size = os.path.getsize(os.path.join(outdir, "weights.f32.bin"))
    print(f"wrote weights.f32.bin ({size/1e6:.2f} MB)", flush=True)

    n_params = count_paint_params(cfg, params)
    meta, val_loss = {}, None
    meta_path = os.path.join(os.path.dirname(ckpt_path), "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    log_path = os.path.join(os.path.dirname(ckpt_path), "log.jsonl")
    if os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                try:
                    dd = json.loads(line)
                except Exception:
                    continue
                if "val_loss" in dd:
                    val_loss = dd["val_loss"]

    config = {
        "name": "paintexe",
        "arch": "v0id-unet-1",
        "img": cfg.img,
        "channels": list(cfg.ch),
        "nDown": list(cfg.n_down),
        "midBlocks": cfg.mid_blocks,
        "heads": cfg.heads,
        "byteDim": cfg.byte_dim,
        "textDim": cfg.text_dim,
        "timeDim": cfg.time_dim,
        "maxText": cfg.max_text,
        "tSteps": cfg.t_steps,
        "betaStart": cfg.beta_start,
        "betaEnd": cfg.beta_end,
        "groups": 8,
        "gnEps": 1e-5,
        "paramCount": n_params,
        "tensors": table,
        "labels": ["airplane", "car", "bird", "cat", "deer", "dog",
                   "frog", "horse", "ship", "truck"],
        "training": {"steps": meta.get("step", 0), "valLoss": val_loss},
    }
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(config, f)
    print(f"paramCount: {n_params:,}", flush=True)

    # ---- parity fixtures ----
    jp = {k: jnp.array(v) for k, v in params.items()}
    prompt = "a photo of a cat"
    ids, m = text_to_bytes(prompt, cfg.max_text)
    cond = np.asarray(encode_text(jp, cfg, jnp.array(ids[None]), jnp.array(m[None])))[0]
    xrng = np.random.default_rng(11)
    x_t = xrng.standard_normal((1, 32, 32, 3), dtype=np.float32)
    eps_pred = np.asarray(apply_unet(jp, cfg, jnp.array(x_t), jnp.array([500]), jnp.array(cond[None])))[0]
    ts8 = ddim_times(8, cfg.t_steps)
    x_init = np.random.default_rng(5).standard_normal((1, 32, 32, 3), dtype=np.float32)[0]
    ddim8 = ddim_sample(jp, cfg, abar, [prompt], steps=8, seed=5, w=1.0)[0]
    with open(os.path.join(outdir, "parity.json"), "w") as f:
        json.dump({"prompt": prompt,
                   "cond": [float(v) for v in cond],
                   "x_t": [float(v) for v in x_t.reshape(-1)],
                   "t_single": 500,
                   "eps_pred": [float(v) for v in eps_pred.reshape(-1)],
                   "x_init": [float(v) for v in x_init.reshape(-1)],
                   "timesteps": [int(v) for v in ts8],
                   "ddim8": [float(v) for v in ddim8.reshape(-1)]}, f)
    print(f"parity: cond_mean={cond.mean():.4f} eps_mean={eps_pred.mean():.4f} "
          f"ddim8_mean={ddim8.mean():.4f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()
    export(args.ckpt)


if __name__ == "__main__":
    main()
