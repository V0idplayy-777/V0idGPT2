"""Pretrain + chat-SFT trainer for the five V0idGPT2 language models (JAX).

Pipeline:
  1. --pretokenize : corpus text -> tokens.bin / chat.npz (uint16)
  2. --stage pretrain : causal LM training on tokens.bin
  3. --stage sft      : chat fine-tune on chat.npz (assistant-span loss mask)

Checkpoints (-> /home/user/checkpoints/<model>_<stage>/) store fp32 params
(.npz) + meta.json; training can resume from any checkpoint (fresh optimizer).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "common"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenizer"))

import numpy as np

import jax
import jax.numpy as jnp
import optax

import model as M
from configs import CONFIGS
from bpe import BPETokenizer, EOS_ID, PAD_ID

CORPUS = "/home/user/textdata/corpus"
CKPT_ROOT = "/home/user/checkpoints"
DOC_SEP = "<|endoftext|>"


# ------------------------------------------------------------- pretokenization
def pretokenize():
    tok = BPETokenizer()
    os.makedirs(CORPUS, exist_ok=True)

    def pack_text_file(name):
        path = os.path.join(CORPUS, name + ".txt")
        with open(path, encoding="utf-8") as f:
            text = f.read()
        docs = [d for d in text.split(DOC_SEP) if d.strip()]
        ids = []
        for d in docs:
            ids.extend(tok.encode(d.strip()))
            ids.append(EOS_ID)
        arr = np.array(ids, dtype=np.uint32)
        assert arr.max() < 65536
        arr.astype(np.uint16).tofile(os.path.join(CORPUS, name + ".bin"))
        print(f"{name}: {len(docs)} docs -> {len(ids)} tokens", flush=True)

    def pack_chat(name):
        ids_all, w_all, offsets = [], [], [0]
        n_trunc = 0
        with open(os.path.join(CORPUS, name + ".jsonl"), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                ids, w = tok.encode_chat(d["messages"])
                if len(ids) > 2048:
                    ids, w = ids[-2048:], w[-2048:]
                    n_trunc += 1
                ids_all.extend(ids)
                w_all.extend([1 if x > 0.5 else 0 for x in w])
                offsets.append(len(ids_all))
        np.savez_compressed(
            os.path.join(CORPUS, name + ".npz"),
            ids=np.array(ids_all, dtype=np.uint16),
            weights=np.array(w_all, dtype=np.uint8),
            offsets=np.array(offsets, dtype=np.int64),
        )
        print(f"{name}: {len(offsets)-1} dialogues -> {len(ids_all)} tokens ({n_trunc} truncated)", flush=True)

    pack_text_file("pretrain")
    pack_text_file("val")
    pack_chat("chat")
    pack_chat("val_chat")


# ------------------------------------------------------------------- batching
class PretrainData:
    def __init__(self, name="pretrain"):
        self.tok = np.memmap(os.path.join(CORPUS, name + ".bin"), dtype=np.uint16, mode="r")
        self.n = len(self.tok)
        print(f"loaded {name}: {self.n} tokens", flush=True)

    def batch(self, rng, B, T):
        starts = rng.integers(0, self.n - (T + 1), size=B)
        x = np.stack([np.asarray(self.tok[s:s + T + 1], dtype=np.int32) for s in starts])
        return {"inputs": x[:, :-1], "targets": x[:, 1:], "weights": None}


class ChatData:
    def __init__(self, name="chat"):
        d = np.load(os.path.join(CORPUS, name + ".npz"))
        self.ids = d["ids"]
        self.weights = d["weights"]
        self.offsets = d["offsets"]
        self.n = len(self.offsets) - 1
        print(f"loaded {name}: {self.n} dialogues", flush=True)

    def batch(self, rng, B, T):
        L = T + 1
        idx = rng.integers(0, self.n, size=B)
        x = np.full((B, L), PAD_ID, dtype=np.int32)
        w = np.zeros((B, L), dtype=np.float32)
        for i, j in enumerate(idx):
            s, e = int(self.offsets[j]), int(self.offsets[j + 1])
            seq = np.asarray(self.ids[s:e], dtype=np.int32)[-L:]
            ww = np.asarray(self.weights[s:e], dtype=np.float32)[-L:]
            x[i, -len(seq):] = seq
            w[i, -len(seq):] = ww
        return {"inputs": x[:, :-1], "targets": x[:, 1:], "weights": w[:, 1:]}


# ------------------------------------------------------------------- training
def train_step_factory(cfg, block_fn, optimizer):
    @jax.jit
    def step(params, opt_state, inputs, targets, weights):
        w = None if weights is None else weights
        (loss, grads) = jax.value_and_grad(M.loss_fn)(params, inputs, targets, w, cfg, block_fn)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    return step


def eval_factory(cfg, block_fn):
    @jax.jit
    def ev(params, inputs, targets, weights):
        return M.loss_fn(params, inputs, targets, weights, cfg, block_fn)
    return ev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretokenize", action="store_true")
    ap.add_argument("--model", default="potato", choices=list(CONFIGS.keys()))
    ap.add_argument("--stage", default="pretrain", choices=["pretrain", "sft"])
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--resume", default="")
    ap.add_argument("--fresh-schedule", action="store_true",
                    help="init weights from --resume but restart step count/schedule (for SFT)")
    ap.add_argument("--base-tokens", type=int, default=0,
                    help="tokens already seen in a previous stage (added to the total)")
    ap.add_argument("--ckpt-every", type=int, default=800)
    ap.add_argument("--val-every", type=int, default=400)
    args = ap.parse_args()

    if args.pretokenize:
        pretokenize()
        return

    cfg = CONFIGS[args.model]
    T = cfg.ctx
    B = args.batch
    print(f"model={args.model} stage={args.stage} steps={args.steps} B={B} T={T} lr={args.lr} seed={args.seed}", flush=True)
    print(f"params: {M.count_params(cfg):,}", flush=True)

    tok_data = PretrainData("pretrain") if args.stage == "pretrain" else None
    chat_data = ChatData("chat") if args.stage == "sft" else None
    val_data = PretrainData("val") if args.stage == "pretrain" else ChatData("val_chat")
    data = tok_data or chat_data

    ckpt_dir = os.path.join(CKPT_ROOT, f"{args.model}_{args.stage}")
    os.makedirs(ckpt_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    params = M.init_params(jax.random.PRNGKey(args.seed), cfg)
    start_step = 0
    resumed_tokens = 0
    if args.resume:
        print(f"resuming from {args.resume}", flush=True)
        z = np.load(args.resume)
        params = {k: jnp.array(z[k]) for k in z.files}
        meta_path = os.path.join(os.path.dirname(args.resume), "meta.json")
        if os.path.exists(meta_path) and not args.fresh_schedule:
            with open(meta_path) as f:
                meta = json.load(f)
                start_step = int(meta.get("step", 0))
                resumed_tokens = int(meta.get("tokens", 0))
            print(f"resuming at step {start_step} ({resumed_tokens} tokens)", flush=True)
        if args.fresh_schedule:
            print("fresh schedule: step count restarted", flush=True)

    warmup = min(300, max(10, args.steps // 10))
    warmup = min(warmup, max(args.steps - 1, 1))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=args.lr, warmup_steps=warmup,
        decay_steps=max(args.steps - warmup, 1), end_value=args.lr * 0.05)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=schedule, b1=0.9, b2=0.95, eps=1e-8,
                    weight_decay=0.1,
                    mask=lambda tree: jax.tree_util.tree_map(lambda p: p.ndim == 2, tree)),
    )
    opt_state = optimizer.init(params)
    block_fn = M.make_block_fn(cfg)
    step_fn = train_step_factory(cfg, block_fn, optimizer)
    eval_fn = eval_factory(cfg, block_fn)

    log_path = os.path.join(ckpt_dir, "log.jsonl")
    logf = open(log_path, "a", encoding="utf-8")
    tokens_done = args.base_tokens + resumed_tokens
    t_last = time.time()
    for step in range(start_step + 1, args.steps + 1):
        batch = data.batch(rng, B, T)
        params, opt_state, loss = step_fn(
            params, opt_state,
            jnp.array(batch["inputs"]),
            jnp.array(batch["targets"]),
            None if batch["weights"] is None else jnp.array(batch["weights"]))
        tokens_done += B * T
        if step % 25 == 0 or step == args.steps:
            loss_f = float(loss.block_until_ready())
            dt = time.time() - t_last
            tps = (25 * B * T) / max(dt, 1e-6)
            t_last = time.time()
            lr_now = float(schedule(step - 1))
            print(f"step {step}/{args.steps} loss={loss_f:.4f} lr={lr_now:.2e} toks={tokens_done} tok/s={tps:.0f}", flush=True)
            logf.write(json.dumps({"step": step, "loss": loss_f, "lr": lr_now, "tokens": tokens_done}) + "\n")
            logf.flush()
        if step % args.val_every == 0 or step == args.steps:
            vrng = np.random.default_rng(999)
            losses = []
            for _ in range(12 if args.stage == "pretrain" else 8):
                vb = val_data.batch(vrng, B, T)
                vl = eval_fn(params, jnp.array(vb["inputs"]), jnp.array(vb["targets"]),
                              None if vb["weights"] is None else jnp.array(vb["weights"]))
                losses.append(float(vl.block_until_ready()))
            print(f"  [val] step {step} loss={np.mean(losses):.4f}", flush=True)
            logf.write(json.dumps({"step": step, "val_loss": float(np.mean(losses))}) + "\n")
            logf.flush()
        if step % args.ckpt_every == 0 or step == args.steps:
            path = os.path.join(ckpt_dir, f"step_{step}.npz")
            np.savez(path, **{k: np.asarray(v) for k, v in params.items()})
            with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
                json.dump({"step": step, "tokens": tokens_done,
                           "loss": float(loss.block_until_ready())}, f)
            print(f"  saved {path}", flush=True)
    logf.close()
    print("TRAINING DONE", flush=True)


if __name__ == "__main__":
    main()
