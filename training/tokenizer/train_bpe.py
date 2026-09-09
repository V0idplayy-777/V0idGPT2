"""Train the byte-level BPE tokenizer (GPT-2 style) for V0idGPT2.

Vocabulary layout (VOCAB_SIZE = 4096):
  ids 0..255    : raw UTF-8 bytes
  ids 256..4092 : learned merges (3837 merges, new id = 256 + merge_index)
  id 4093       : <bos>, id 4094 : <eos>, id 4095 : <pad>

Pretokenization: split text into (maximal non-whitespace run | single
whitespace char) pieces; BPE merges are learned/applied within each piece.

Outputs (training/tokenizer/out/):
  tokenizer.json  (compact: int-pair merges + specials; shipped to the web app)
  vocab.json      (rendered human-readable vocab, GPT-2 bytes_to_unicode style)
  merges.txt      (rendered merges, one "a b" per line)
"""
import json
import os
import random
import re
import time
from collections import Counter, defaultdict

VOCAB_SIZE = 4096
N_SPECIALS = 3
N_MERGES = VOCAB_SIZE - 256 - N_SPECIALS  # 3837
BOS_ID = VOCAB_SIZE - 3  # 4093
EOS_ID = VOCAB_SIZE - 2  # 4094
PAD_ID = VOCAB_SIZE - 1  # 4095
DOC_SEP = "<|endoftext|>"
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def bytes_to_unicode():
    """GPT-2 style mapping of every byte 0..255 to a printable unicode char."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def get_pairs(ids):
    return {(ids[i], ids[i + 1]) for i in range(len(ids) - 1)}


def sample_training_text():
    """~10MB mixed sample of pretrain + chat text."""
    base = "/home/user/textdata/corpus"
    chunks = []
    with open(os.path.join(base, "pretrain.txt"), encoding="utf-8") as f:
        chunks.append(f.read(8_000_000))
    # chat: sample message contents
    msgs = []
    with open(os.path.join(base, "chat.jsonl"), encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except Exception:
                continue
            for m in d.get("messages", []):
                msgs.append(m.get("content", ""))
            if len(msgs) > 60000:
                break
    rng = random.Random(7)
    rng.shuffle(msgs)
    chat_text, size = [], 0
    for m in msgs:
        chat_text.append(m)
        size += len(m)
        if size > 2_500_000:
            break
    chunks.append("\n".join(chat_text))
    text = "\n".join(chunks)
    text = text.replace(DOC_SEP, "\n")
    return text


def train(text, n_merges=N_MERGES, verbose=True):
    t0 = time.time()
    pieces = re.findall(r"\S+|\s", text)
    counts = Counter(pieces)
    # Drop hapax pieces: they can never win a merge on their own and skipping
    # them massively reduces the working set (standard practice).
    counts = {p: c for p, c in counts.items() if c >= 2}
    if verbose:
        print(f"pretokenized pieces: {len(pieces)}, unique (count>=2): {len(counts)}", flush=True)

    splits = {}   # bytes piece -> list[int]
    weights = {}  # bytes piece -> count
    for piece, c in counts.items():
        bb = piece.encode("utf-8")
        if bb in splits:
            weights[bb] += c
        else:
            splits[bb] = list(bb)
            weights[bb] = c

    pair_counts = Counter()
    index = defaultdict(set)
    for key, ids in splits.items():
        w = weights[key]
        for p in get_pairs(ids):
            pair_counts[p] += w
            index[p].add(key)

    merges = []
    vocab_bytes = {i: bytes([i]) for i in range(256)}
    for i in range(n_merges):
        if not pair_counts:
            break
        best = max(pair_counts, key=pair_counts.get)
        if pair_counts[best] < 2:
            break
        a, b = best
        new_id = 256 + len(merges)
        merges.append([a, b])
        vocab_bytes[new_id] = vocab_bytes[a] + vocab_bytes[b]
        affected = list(index.get(best, ()))
        for key in affected:
            ids = splits[key]
            new_ids = []
            j = 0
            while j < len(ids):
                if j < len(ids) - 1 and ids[j] == a and ids[j + 1] == b:
                    new_ids.append(new_id)
                    j += 2
                else:
                    new_ids.append(ids[j])
                    j += 1
            w = weights[key]
            for p in get_pairs(ids):
                c = pair_counts[p] - w
                if c <= 0:
                    pair_counts.pop(p, None)
                else:
                    pair_counts[p] = c
                s = index.get(p)
                if s is not None:
                    s.discard(key)
            splits[key] = new_ids
            for p in get_pairs(new_ids):
                pair_counts[p] += w
                index[p].add(key)
        index.pop(best, None)
        pair_counts.pop(best, None)
        if verbose and (len(merges) % 500 == 0):
            print(f"  merge {len(merges)}/{n_merges}  ({time.time()-t0:.0f}s)", flush=True)
    if verbose:
        print(f"learned {len(merges)} merges in {time.time()-t0:.0f}s", flush=True)
    # Pad merge list if corpus was too small (keeps ids stable at 4096).
    while len(merges) < n_merges:
        merges.append([0, 0])
    return merges


def save(merges):
    os.makedirs(OUTDIR, exist_ok=True)
    b2u = bytes_to_unicode()
    vocab_bytes = {i: bytes([i]) for i in range(256)}
    for i, (a, b) in enumerate(merges):
        vocab_bytes[256 + i] = vocab_bytes[a] + vocab_bytes[b]
    # compact web format: int-pair merges (new id = 256 + index, implicit)
    with open(os.path.join(OUTDIR, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump({
            "vocab_size": VOCAB_SIZE,
            "merges": merges,
            "special": {"bos": BOS_ID, "eos": EOS_ID, "pad": PAD_ID},
        }, f)
    # human-readable rendered vocab
    def render(b: bytes) -> str:
        return "".join(b2u[x] for x in b)
    vocab = {render(vocab_bytes[i]): i for i in range(256 + len(merges))}
    vocab["<bos>"] = BOS_ID
    vocab["<eos>"] = EOS_ID
    vocab["<pad>"] = PAD_ID
    with open(os.path.join(OUTDIR, "vocab.json"), "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    with open(os.path.join(OUTDIR, "merges.txt"), "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{render(vocab_bytes[a])} {render(vocab_bytes[b])}\n")
    print(f"saved tokenizer to {OUTDIR}", flush=True)


def main():
    text = sample_training_text()
    print(f"training-text chars: {len(text)}", flush=True)
    merges = train(text)
    save(merges)


if __name__ == "__main__":
    main()
