"""Train Paint.exe: text-conditional DDPM U-Net on CIFAR-10 (JAX).

Checkpoints -> /home/user/checkpoints/paint/ (params + EMA + sample grids).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import jax
import jax.numpy as jnp
import optax
from PIL import Image

from unet import (PaintConfig, apply_unet, count_paint_params, ddim_times,
                  encode_text, init_paint_params, schedule)

IMGDATA = "/home/user/imgdata"
CKPT = "/home/user/checkpoints/paint"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]
WORDS = {"airplane": ["airplane"], "automobile": ["car", "automobile"],
         "bird": ["bird"], "cat": ["cat"], "deer": ["deer"], "dog": ["dog"],
         "frog": ["frog"], "horse": ["horse"], "ship": ["ship", "boat"],
         "truck": ["truck"]}
TEMPLATES = ["a photo of a {}", "a picture of a {}", "{}", "a {}", "a photo of the {}"]


def text_to_bytes(text, L=64):
    b = text.encode("utf-8")[:L]
    ids = np.zeros((L,), dtype=np.int32)
    mask = np.zeros((L,), dtype=np.float32)
    ids[:len(b)] = np.frombuffer(b, dtype=np.uint8)
    mask[:len(b)] = 1.0
    return ids, mask


def make_prompt(label_idx, rng):
    cls = CLASSES[int(label_idx)]
    w = WORDS[cls][rng.integers(0, len(WORDS[cls]))]
    t = TEMPLATES[rng.integers(0, len(TEMPLATES))]
    return t.format(w)


def ddim_sample(params, cfg, abar, prompts, steps=20, seed=0, w=1.0):
    """Full DDIM sampling (used for monitoring grids + sample script logic)."""
    ts = ddim_times(steps, cfg.t_steps)
    null = np.asarray(params["null_cond"])
    conds = []
    for pr in prompts:
        ids, m = text_to_bytes(pr, cfg.max_text)
        c = encode_text(params, cfg, jnp.array(ids[None]), jnp.array(m[None]))
        conds.append(np.asarray(c)[0])
    conds = jnp.array(np.stack(conds))

    @jax.jit
    def apply(x, t, c):
        return apply_unet(params, cfg, x, t, c)

    rng = np.random.default_rng(seed)
    x = jnp.array(rng.standard_normal((len(prompts), 32, 32, 3), dtype=np.float32))
    B = len(prompts)
    for i in reversed(range(len(ts))):
        t = ts[i]
        t_prev = ts[i - 1] if i > 0 else -1
        ab_t = float(abar[t])
        ab_prev = float(abar[t_prev]) if t_prev >= 0 else 1.0
        tt = jnp.full((B,), t, dtype=jnp.int32)
        e_cond = apply(x, tt, conds)
        if w != 1.0:
            e_un = apply(x, tt, jnp.broadcast_to(jnp.array(null), conds.shape))
            eps = e_un + w * (e_cond - e_un)
        else:
            eps = e_cond
        x0 = (x - np.sqrt(1 - ab_t) * eps) / np.sqrt(ab_t)
        x = np.sqrt(ab_prev) * x0 + np.sqrt(max(0.0, 1 - ab_prev)) * eps
    return np.asarray(np.clip((np.asarray(x) + 1) / 2, 0, 1))


def save_grid(imgs, path, scale=4):
    n = len(imgs)
    h, w, _ = imgs[0].shape
    sheet = np.concatenate([(imgs[i] * 255).astype(np.uint8) for i in range(n)], axis=1)
    Image.fromarray(sheet).resize((w * n * scale, h * scale), Image.NEAREST).save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--resume", default="")
    ap.add_argument("--ckpt-every", type=int, default=2000)
    args = ap.parse_args()

    cfg = PaintConfig()
    betas, abar = schedule(cfg)
    abar_j = jnp.array(abar)
    print(f"paint unet training: steps={args.steps} B={args.batch} lr={args.lr}", flush=True)

    d = np.load(os.path.join(IMGDATA, "cifar_train.npz"))
    images, labels = d["images"], d["labels"]
    dv = np.load(os.path.join(IMGDATA, "cifar_val.npz"))
    vimages, vlabels = dv["images"][:512], dv["labels"][:512]
    print(f"train {images.shape}, val sample {vimages.shape}", flush=True)

    params = init_paint_params(jax.random.PRNGKey(args.seed), cfg)
    print(f"paint params: {count_paint_params(cfg, params):,}", flush=True)
    start = 0
    if args.resume:
        z = np.load(args.resume)
        params = {k: jnp.array(np.array(z[k])) for k in z.files}
        print(f"resumed params from {args.resume}", flush=True)
        ema_path = args.resume.replace(".npz", ".ema.npz")
        ema = None
        if os.path.exists(ema_path):
            ze = np.load(ema_path)
            ema = {k: jnp.array(np.array(ze[k])) for k in ze.files}
            print("resumed ema", flush=True)
        meta_path = os.path.join(os.path.dirname(args.resume), "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                start = int(json.load(f).get("step", 0))
    else:
        ema = None
    if ema is None:
        ema = jax.tree_util.tree_map(lambda x: x.copy(), params)

    def loss_fn(p, x, byte_ids, mask, rng):
        B = x.shape[0]
        r1, r2, r3 = jax.random.split(rng, 3)
        t = jax.random.randint(r1, (B,), 0, cfg.t_steps)
        eps = jax.random.normal(r2, x.shape)
        ab = abar_j[t][:, None, None, None]
        x_t = jnp.sqrt(ab) * x + jnp.sqrt(1.0 - ab) * eps
        cond = encode_text(p, cfg, byte_ids, mask)
        drop = jax.random.bernoulli(r3, 0.1, (B,))[:, None]
        cond = jnp.where(drop, p["null_cond"][None, :], cond)
        pred = apply_unet(p, cfg, x_t, t, cond)
        return jnp.mean((pred - eps) ** 2)

    @jax.jit
    def train_step(p, opt_state, x, byte_ids, mask, rng):
        (loss, grads) = jax.value_and_grad(loss_fn)(p, x, byte_ids, mask, rng)
        updates, opt_state = optimizer.update(grads, opt_state, p)
        return optax.apply_updates(p, updates), opt_state, loss

    @jax.jit
    def val_loss(p, x, byte_ids, mask, rng):
        return loss_fn(p, x, byte_ids, mask, rng)

    sched = optax.join_schedules(
        [optax.linear_schedule(0.0, args.lr, 500),
         optax.constant_schedule(args.lr)], [500])
    optimizer = optax.chain(optax.clip_by_global_norm(1.0),
                            optax.adamw(sched, b1=0.9, b2=0.99, weight_decay=0.0))
    opt_state = optimizer.init(params)

    os.makedirs(CKPT, exist_ok=True)
    logf = open(os.path.join(CKPT, "log.jsonl"), "a", encoding="utf-8")
    rng = np.random.default_rng(args.seed)
    jrng = jax.random.PRNGKey(args.seed + 1)
    # fixed val batch
    v_idx = np.random.default_rng(0).integers(0, len(vimages), size=64)
    vx = vimages[v_idx].astype(np.float32) / 127.5 - 1.0
    vb, vm = [], []
    for li in vlabels[v_idx]:
        ids, m = text_to_bytes(make_prompt(li, np.random.default_rng(int(li) + 1)), cfg.max_text)
        vb.append(ids)
        vm.append(m)
    vxb, vbb, vmb = jnp.array(vx), jnp.array(np.stack(vb)), jnp.array(np.stack(vm))
    grid_prompts = [f"a photo of a {WORDS[c][0]}" for c in CLASSES]

    t_last = time.time()
    B = args.batch
    for step in range(start + 1, args.steps + 1):
        idx = rng.integers(0, len(images), size=B)
        x = images[idx].astype(np.float32) / 127.5 - 1.0
        flip = rng.random(B) < 0.5
        x[flip] = x[flip, :, ::-1, :]
        bb, mm = [], []
        for li in labels[idx]:
            ids, m = text_to_bytes(make_prompt(li, rng), cfg.max_text)
            bb.append(ids)
            mm.append(m)
        jrng, sub = jax.random.split(jrng)
        params, opt_state, loss = train_step(
            params, opt_state, jnp.array(x), jnp.array(np.stack(bb)), jnp.array(np.stack(mm)), sub)
        ema = jax.tree_util.tree_map(lambda e, p: 0.9999 * e + 0.0001 * p, ema, params)
        if step % 50 == 0 or step == args.steps:
            lf = float(loss.block_until_ready())
            dt = time.time() - t_last
            t_last = time.time()
            print(f"step {step}/{args.steps} loss={lf:.4f} imgs/s={50*B/dt:.1f}", flush=True)
            logf.write(json.dumps({"step": step, "loss": lf}) + "\n")
            logf.flush()
        if step % 500 == 0 or step == args.steps:
            jrng, sub = jax.random.split(jrng)
            vl = float(val_loss(ema, vxb, vbb, vmb, sub).block_until_ready())
            print(f"  [val] step {step} loss={vl:.4f}", flush=True)
            logf.write(json.dumps({"step": step, "val_loss": vl}) + "\n")
            logf.flush()
        if step % args.ckpt_every == 0 or step == args.steps:
            np.savez(os.path.join(CKPT, f"step_{step}.npz"),
                     **{k: np.asarray(v) for k, v in params.items()})
            np.savez(os.path.join(CKPT, f"step_{step}.ema.npz"),
                     **{k: np.asarray(v) for k, v in ema.items()})
            with open(os.path.join(CKPT, "meta.json"), "w") as f:
                json.dump({"step": step}, f)
            print(f"  sampling grid...", flush=True)
            grid = ddim_sample(ema, cfg, abar, grid_prompts, steps=16, seed=step, w=1.0)
            save_grid(grid, os.path.join(CKPT, f"grid_{step}.png"))
            print(f"  saved step_{step}", flush=True)
    logf.close()
    print("PAINT TRAINING DONE", flush=True)


if __name__ == "__main__":
    main()
