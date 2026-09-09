"""Quality spot-checks: sample from a JAX checkpoint (full-forward loop).

Fast for potato/toaster; slow for the big models (use the browser / val loss).
Usage:
  python training/sample.py --model potato --ckpt .../step_2000.npz --prompt "Hello!"
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "common"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer"))

import numpy as np

import jax
import jax.numpy as jnp

import model as M
from configs import CONFIGS
from bpe import BPETokenizer, EOS_ID


def sample_next(logits, temp, top_k, top_p, rng):
    x = np.asarray(logits, dtype=np.float64)
    order = np.argsort(-x)
    k = len(x)
    if top_k > 0:
        k = min(k, top_k)
    if top_p < 1.0:
        t = temp if temp > 0 else 1.0
        ex = np.exp((x[order[:k]] - x[order[0]]) / t)
        probs = ex / ex.sum()
        cum = np.cumsum(probs)
        k = max(1, int(np.searchsorted(cum, top_p) + 1))
    if temp <= 0:
        return int(order[0])
    ex = np.exp((x[order[:k]] - x[order[0]]) / temp)
    probs = ex / ex.sum()
    return int(rng.choice(order[:k], p=probs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="potato", choices=list(CONFIGS.keys()))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", default="Hello!")
    ap.add_argument("--tokens", type=int, default=60)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--topk", type=int, default=40)
    ap.add_argument("--topp", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    cfg = CONFIGS[args.model]
    tok = BPETokenizer()
    z = np.load(args.ckpt)
    params = {k: jnp.array(np.array(z[k])) for k in z.files}
    block_fn = M.make_block_fn(cfg)

    @jax.jit
    def fwd(ids):
        return M.forward(params, ids, cfg, block_fn)[0, -1]

    rng = np.random.default_rng(args.seed)
    ids = tok.chat_prompt(args.prompt)
    print(f"prompt ids ({len(ids)}): {ids}", flush=True)
    out = []
    for _ in range(args.tokens):
        logits = np.asarray(fwd(jnp.array([np.array(ids, dtype=np.int32)])))
        nxt = sample_next(logits, args.temp, args.topk, args.topp, rng)
        if nxt == EOS_ID:
            break
        out.append(nxt)
        ids.append(nxt)
        if len(ids) >= cfg.ctx:
            break
    print("---- sample ----", flush=True)
    print(tok.decode(out), flush=True)
    print("---- ids ----", flush=True)
    print(out, flush=True)


if __name__ == "__main__":
    main()
