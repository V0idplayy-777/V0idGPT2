"""Export trained JAX params -> browser weight files + parity fixtures.

For each model writes public/models/<name>/:
  config.json      architecture + tensor table (byte offsets) + training meta
  weights.q8.bin   16-byte header ("V0Q8", version=1) + 16-byte aligned payloads:
                   fp32 for embeddings/norms, Q8_0 (int8 blocks of 32 + fp16
                   scale) for all 2-D linears, stored OUTPUT-MAJOR ([out, in]).
  parity.json      JAX reference logits (fp32 + dequantized-Q8) and a greedy
                   rollout for the TypeScript parity tests.

Usage:
  python training/export_weights.py --model potato \\
      --ckpt /home/user/checkpoints/potato_sft/step_1200.npz
"""
import argparse
import json
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "common"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer"))

import numpy as np

import jax
import jax.numpy as jnp

import model as M
from configs import CONFIGS
from bpe import BPETokenizer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Q8_2D = ("wq", "wk", "wv", "wo", "wgate", "wup", "wdown")


def quantize_q8(w):
    """w: [rows, cols] fp32 ([out, in]) -> (int8 [rows, stride], fp16bits [rows, nblocks])."""
    rows, cols = w.shape
    stride = (cols + 31) // 32 * 32
    nblocks = stride // 32
    qp = np.zeros((rows, stride), dtype=np.int8)
    scales = np.zeros((rows, nblocks), dtype=np.float32)
    for b in range(nblocks):
        s = b * 32
        e = min(cols, s + 32)
        if s >= cols:
            break
        blk = w[:, s:e]
        amax = np.abs(blk).max(axis=1)
        sc = np.where(amax == 0, 0.0, amax / 127.0).astype(np.float32)
        scales[:, b] = sc
        safe = np.where(sc == 0, 1.0, sc)[:, None]
        q = np.round(blk / safe)
        qp[:, s:e] = np.clip(q, -127, 127).astype(np.int8)
    s16 = scales.astype(np.float16).view(np.uint16)
    back = s16.view(np.float16).astype(np.float32)
    if not np.allclose(back, scales, rtol=1e-3, atol=1e-7):
        print("WARNING: fp16 scale overflow detected")
    return qp, s16


def dequantize_q8(qp, s16, shape):
    rows, cols = shape
    stride = qp.shape[1]
    nblocks = stride // 32
    scales = s16.view(np.float16).astype(np.float32)
    w = np.zeros((rows, cols), dtype=np.float32)
    for b in range(nblocks):
        s = b * 32
        e = min(cols, s + 32)
        if s >= cols:
            break
        w[:, s:e] = qp[:, s:e].astype(np.float32) * scales[:, b:b + 1]
    return w


def export(model_name, ckpt_path):
    cfg = CONFIGS[model_name]
    z = np.load(ckpt_path)
    params = {k: np.array(z[k], dtype=np.float32) for k in z.files}
    print(f"loaded {ckpt_path}: {len(params)} tensors", flush=True)

    outdir = os.path.join(REPO, "public", "models", model_name)
    os.makedirs(outdir, exist_ok=True)

    # canonical tensor order
    names = ["tok_emb", "final_norm"]
    for i in range(cfg.n_layer):
        names += [f"L{i}.attn_norm", f"L{i}.wq", f"L{i}.wk", f"L{i}.wv", f"L{i}.wo",
                  f"L{i}.mlp_norm", f"L{i}.wgate", f"L{i}.wup", f"L{i}.wdown"]
    assert set(names) == set(params.keys()), "checkpoint tensor mismatch"

    blobs = {}      # name -> (dtype, stored_shape, bytes)
    max_err = 0.0
    for n in names:
        w = params[n]
        short = n.split(".")[-1]
        if w.ndim == 2 and short in Q8_2D:
            wt = np.ascontiguousarray(w.T)  # -> [out, in]
            qp, s16 = quantize_q8(wt)
            blobs[n] = ("q8", list(wt.shape), qp.tobytes() + s16.tobytes())
            err = float(np.abs(dequantize_q8(qp, s16, wt.shape) - wt).max())
            max_err = max(max_err, err)
        else:
            assert short not in Q8_2D and w.ndim <= 2
            blobs[n] = ("f32", list(w.shape), np.ascontiguousarray(w).tobytes())

    # layout with 16-byte alignment
    table = []
    offset = 0
    chunks = []
    for n in names:
        dtype, shape, raw = blobs[n]
        table.append({"name": n, "shape": shape, "dtype": dtype,
                      "offset": offset, "length": len(raw)})
        chunks.append(raw)
        pad = (16 - (len(raw) % 16)) % 16
        chunks.append(b"\x00" * pad)
        offset += len(raw) + pad

    with open(os.path.join(outdir, "weights.q8.bin"), "wb") as f:
        f.write(b"V0Q8")
        f.write(struct.pack("<I", 1))
        f.write(struct.pack("<II", 0, 0))
        for c in chunks:
            f.write(c)
    size = os.path.getsize(os.path.join(outdir, "weights.q8.bin"))
    print(f"wrote weights.q8.bin ({size/1e6:.2f} MB), max Q8 abs err {max_err:.2e}", flush=True)

    # training meta
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
                    d = json.loads(line)
                except Exception:
                    continue
                if "val_loss" in d:
                    val_loss = d["val_loss"]

    config = {
        "name": model_name,
        "arch": "v0id-transformer-1",
        "vocab": cfg.vocab,
        "dModel": cfg.d_model,
        "nLayer": cfg.n_layer,
        "nHead": cfg.n_head,
        "dFF": cfg.d_ff,
        "ctx": cfg.ctx,
        "ropeTheta": cfg.rope_theta,
        "rmsEps": cfg.rms_eps,
        "tied": True,
        "bos": 4093, "eos": 4094, "pad": 4095,
        "format": "q8_0",
        "paramCount": M.count_params(cfg),
        "tensors": table,
        "training": {"tokens": meta.get("tokens", 0), "steps": meta.get("step", 0),
                     "stage": os.path.basename(os.path.dirname(ckpt_path)),
                     "valLoss": val_loss},
    }
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(config, f)
    print(f"paramCount: {M.count_params(cfg):,}", flush=True)

    # ---- parity fixtures (JAX reference for TS tests) ----
    tok = BPETokenizer()
    prompt = tok.chat_prompt("Hello!")
    ids = np.array([prompt], dtype=np.int32)
    jp = {k: jnp.array(v) for k, v in params.items()}
    block_fn = M.make_block_fn(cfg)
    logits_fp32 = np.asarray(M.forward(jp, ids, cfg, block_fn))[0, -1]
    # dequantized-Q8 reference (= what the TS engine should compute)
    jp_q = {}
    for n in names:
        w = params[n]
        short = n.split(".")[-1]
        if w.ndim == 2 and short in Q8_2D:
            qp, s16 = quantize_q8(np.ascontiguousarray(w.T))
            jp_q[n] = jnp.array(dequantize_q8(qp, s16, w.T.shape).T)
        else:
            jp_q[n] = jnp.array(w)
    logits_q8 = np.asarray(M.forward(jp_q, ids, cfg, block_fn))[0, -1]
    # greedy rollout (fp32 weights, argmax)
    greedy = []
    cur = list(prompt)
    for _ in range(8):
        lg = np.asarray(M.forward(jp, np.array([cur], dtype=np.int32), cfg, block_fn))[0, -1]
        nxt = int(np.argmax(lg))
        greedy.append(nxt)
        cur.append(nxt)
        if nxt == 4094:
            break
    with open(os.path.join(outdir, "parity.json"), "w") as f:
        json.dump({"prompt_ids": prompt,
                   "logits_fp32": [float(x) for x in logits_fp32],
                   "logits_q8": [float(x) for x in logits_q8],
                   "greedy_next": greedy}, f)
    top1_fp32 = int(np.argmax(logits_fp32))
    top1_q8 = int(np.argmax(logits_q8))
    print(f"parity: prompt_len={len(prompt)} top1_fp32={top1_fp32} top1_q8={top1_q8} "
          f"max|fp32-q8|={np.abs(logits_fp32-logits_q8).max():.3f} greedy={greedy}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(CONFIGS.keys()))
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()
    export(args.model, args.ckpt)


if __name__ == "__main__":
    main()
