"""Sample Paint.exe from an EMA checkpoint (JAX reference sampler).

Usage:
  python training/paint/sample_paint.py --ckpt .../step_20000.ema.npz \\
      --prompt "a photo of a cat" --steps 20 --seed 3 --out /tmp/cat.png
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import jax
import jax.numpy as jnp
from PIL import Image

from train_paint import ddim_sample
from unet import PaintConfig, schedule


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", default="a photo of a cat")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--w", type=float, default=1.0)
    ap.add_argument("--out", default="/tmp/paint_sample.png")
    args = ap.parse_args()

    cfg = PaintConfig()
    _, abar = schedule(cfg)
    z = np.load(args.ckpt)
    params = {k: jnp.array(np.array(z[k])) for k in z.files}
    img = ddim_sample(params, cfg, abar, [args.prompt], steps=args.steps,
                      seed=args.seed, w=args.w)[0]
    px = (img * 255).astype(np.uint8)
    Image.fromarray(px).resize((256, 256), Image.NEAREST).save(args.out)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
