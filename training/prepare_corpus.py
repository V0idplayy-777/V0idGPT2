"""Build pretraining + chat corpora from the fetched raw datasets.

Inputs (outside the git repo, fetched via git clone from public GitHub repos):
  /home/user/textdata/nltk/packages/corpora/*.zip   (NLTK data: twitter, europarl, reuters,
                                                     gutenberg, movie_reviews, brown, abc,
                                                     state_union, webtext, subjectivity,
                                                     sentence_polarity, shakespeare, genesis,
                                                     inaugural, nps_chat, pros_cons, conll2000,
                                                     switchboard)
  /home/user/textdata/wikitext2-raw/wikitext2.txt
  /home/user/textdata/char-rnn/data/tinyshakespeare/input.txt
  /home/user/textdata/DailyDialogDataset/DailyDialog.csv
  /home/user/textdata/cornell/movie_lines.txt + movie_conversations.txt

Outputs:
  /home/user/textdata/corpus/pretrain.txt        (documents separated by a marker line)
  /home/user/textdata/corpus/val.txt
  /home/user/textdata/corpus/chat.jsonl          ({"messages":[{"role","content"}...]})
  /home/user/textdata/corpus/val_chat.jsonl
"""
import csv
import io
import json
import os
import random
import re
import zipfile

TEXTDATA = "/home/user/textdata"
OUTDIR = os.path.join(TEXTDATA, "corpus")
DOC_SEP = "\n\n<|endoftext|>\n\n"
random.seed(1337)


def clean_ws(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s.strip()


def strip_tags(s: str) -> str:
    return re.sub(r"<[^>]*>", " ", s)


# ---------------------------------------------------------------- nltk helpers
def read_zip_texts(zipname, picker=None):
    """Yield (arcname, text) for text files inside an NLTK corpus zip."""
    path = os.path.join(TEXTDATA, "nltk", "packages", "corpora", zipname)
    out = []
    SKIP_EXT = (".gif", ".png", ".jpg", ".jpeg", ".pdf", ".zip", ".gz", ".bz2",
                ".pyc", ".so", ".dll", ".exe", ".mp3", ".wav")
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            lname = name.lower()
            if lname.endswith(SKIP_EXT):
                continue
            if picker and not picker(name):
                continue
            try:
                raw = zf.read(name)
            except KeyError:
                continue
            try:
                txt = raw.decode("utf-8")
            except UnicodeDecodeError:
                txt = raw.decode("latin-1")
            out.append((name, txt))
    return out


def brown_detag(txt):
    toks = []
    for tok in txt.split():
        if "/" in tok:
            tok = tok.rsplit("/", 1)[0]
        if tok in ("``", "''"):
            tok = '"'
        toks.append(tok)
    s = " ".join(toks)
    s = re.sub(r"\s+([.,!?;:')\]}])", r"\1", s)
    s = re.sub(r"([(\[{\"])\s+", r"\1", s)
    return s


def conll_words(txt):
    sents, cur = [], []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            if cur:
                sents.append(cur)
                cur = []
            continue
        parts = line.split()
        if parts:
            cur.append(parts[0])
    if cur:
        sents.append(cur)
    out = []
    for w in sents:
        s = " ".join(w)
        s = re.sub(r"\s+([.,!?;:')\]}])", r"\1", s)
        s = re.sub(r"([(\[{])\s+", r"\1", s)
        out.append(s)
    return "\n".join(out)


# ------------------------------------------------------------------ collectors
pretrain_docs = []
chat_dialogues = []  # list of [turn1, turn2, ...] plain strings, alternating user/assistant


def add_doc(source, text):
    text = clean_ws(text)
    if len(text) < 100:
        return
    pretrain_docs.append(text)


def add_chat(source, turns):
    turns = [clean_ws(t) for t in turns]
    turns = [t for t in turns if t and len(t) < 600]
    if len(turns) >= 2:
        chat_dialogues.append(turns[:12])


def process_nltk():
    # twitter_samples: JSON lists of tweet strings
    for name, txt in read_zip_texts("twitter_samples.zip", lambda n: n.endswith(".json")):
        try:
            tweets = json.loads(txt)
        except Exception:
            continue
        if isinstance(tweets, list):
            buf = []
            for t in tweets:
                if not isinstance(t, str):
                    continue
                t = re.sub(r"https?://\S+", "", t)
                t = clean_ws(t)
                if len(t) > 20:
                    buf.append(t)
            # pack tweets into ~2KB documents
            doc, size = [], 0
            for t in buf:
                doc.append(t)
                size += len(t)
                if size > 2000:
                    add_doc("twitter", "\n".join(doc))
                    doc, size = [], 0
            if doc:
                add_doc("twitter", "\n".join(doc))

    # europarl_raw: english only
    for name, txt in read_zip_texts("europarl_raw.zip", lambda n: "/english/" in n.lower() or n.lower().startswith("english")):
        # files are one-paragraph-per-line; pack into docs
        paras = [p.strip() for p in txt.splitlines() if p.strip()]
        doc, size = [], 0
        for p in paras:
            doc.append(p)
            size += len(p)
            if size > 3000:
                add_doc("europarl", "\n".join(doc))
                doc, size = [], 0
        if doc:
            add_doc("europarl", "\n".join(doc))

    # reuters: SGML -> strip tags
    for name, txt in read_zip_texts("reuters.zip"):
        body = strip_tags(txt)
        body = re.sub(r"&#\d+;", " ", body)
        add_doc("reuters", body)

    # gutenberg / movie_reviews / abc / state_union / inaugural: raw text files
    for z in ["gutenberg.zip", "movie_reviews.zip", "abc.zip", "state_union.zip", "inaugural.zip"]:
        for name, txt in read_zip_texts(z, lambda n: n.lower().endswith(".txt")):
            add_doc(z, txt)

    # webtext: raw, keep conversational files for chat too
    for name, txt in read_zip_texts("webtext.zip", lambda n: n.lower().endswith(".txt")):
        add_doc("webtext", txt)
        if "overheard" in name.lower():
            # overheard.txt has short dialogues separated by blank lines
            for chunk in re.split(r"\n\s*\n", txt):
                turns = [l.strip() for l in chunk.splitlines() if l.strip()]
                if len(turns) >= 2:
                    add_chat("overheard", turns)

    # subjectivity / sentence_polarity / pros_cons: sentence collections -> pack
    for z in ["subjectivity.zip", "sentence_polarity.zip", "pros_cons.zip"]:
        lines = []
        for name, txt in read_zip_texts(z, lambda n: n.lower().endswith(".txt")):
            for line in txt.splitlines():
                line = line.strip().lstrip("+*- ").strip()
                if len(line) > 15:
                    lines.append(line)
        doc, size = [], 0
        for line in lines:
            doc.append(line)
            size += len(line)
            if size > 2000:
                add_doc(z, "\n".join(doc))
                doc, size = [], 0
        if doc:
            add_doc(z, "\n".join(doc))

    # shakespeare: XML -> strip tags
    for name, txt in read_zip_texts("shakespeare.zip"):
        add_doc("shakespeare", strip_tags(txt))

    # genesis: english only
    for name, txt in read_zip_texts("genesis.zip", lambda n: "english" in n.lower()):
        add_doc("genesis", txt)

    # brown: word/tag -> strip tags
    for name, txt in read_zip_texts("brown.zip"):
        add_doc("brown", brown_detag(txt))

    # conll2000: word/tag/chunk columns
    for name, txt in read_zip_texts("conll2000.zip", lambda n: n.lower().endswith(".txt")):
        add_doc("conll2000", conll_words(txt))

    # switchboard: real two-party phone conversations.
    #   transcript -> chat corpus (speaker turns), tagged -> pretrain (detagged)
    for name, txt in read_zip_texts("switchboard.zip", lambda n: os.path.basename(n).lower() in ("tagged", "transcript")):
        if os.path.basename(name).lower() == "transcript":
            turns = []
            for line in txt.splitlines():
                m = re.match(r"^\s*[AB]\.\d+:\s*(.*)$", line)
                if m:
                    t = re.sub(r"\{[^}]*\}", " ", m.group(1))
                    t = re.sub(r"\[[^\]]*\]", " ", t)
                    t = clean_ws(t)
                    if t:
                        turns.append(t)
            for i in range(0, len(turns), 10):
                add_chat("switchboard", turns[i:i + 10])
        else:
            t = re.sub(r"^\s*[AB]\.\d+:\s*", "", txt, flags=re.MULTILINE)
            t = re.sub(r"\{[^}]*\}", " ", t)
            t = re.sub(r"\[[^\]]*\]", " ", t)
            t = re.sub(r"\s*/\s*", " ", t)
            add_doc("switchboard", brown_detag(t))

    # nps_chat: XML chat posts -> dialogues
    for name, txt in read_zip_texts("nps_chat.zip", lambda n: n.lower().endswith(".xml")):
        posts = re.findall(r"<Post[^>]*>(.*?)</Post>", txt, re.DOTALL)
        turns = []
        for p in posts:
            p = strip_tags(p)
            p = re.sub(r"\s+", " ", p).strip()
            if p and len(p) < 400:
                turns.append(p)
        # split long sessions into chunks of <= 10 turns
        for i in range(0, len(turns), 10):
            add_chat("nps_chat", turns[i:i + 10])


def process_wikitext2():
    path = os.path.join(TEXTDATA, "wikitext2-raw", "wikitext2.txt")
    with open(path, encoding="utf-8") as f:
        txt = f.read()
    # drop weird formula-heavy lines, keep the rest; split articles on big headings
    chunks = re.split(r"\n\s*=\s[^=\n]+\s=\s*\n", txt)
    for ch in chunks:
        ch = re.sub(r"@-@", "", ch)
        add_doc("wikitext2", ch)


def process_tinyshakespeare():
    path = os.path.join(TEXTDATA, "char-rnn", "data", "tinyshakespeare", "input.txt")
    with open(path, encoding="utf-8") as f:
        txt = f.read()
    add_doc("tinyshakespeare", txt)


def process_dailydialog():
    path = os.path.join(TEXTDATA, "DailyDialogDataset", "DailyDialog.csv")
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dlg = row.get("Dialogues_Text", "")
            if not dlg:
                continue
            turns = [t.strip() for t in dlg.split("$$$")]
            # detokenize a bit: "word ." -> "word."
            turns = [re.sub(r"\s+([.,!?;:')\]}])", r"\1", t) for t in turns]
            add_chat("dailydialog", turns)


def process_cornell():
    base = os.path.join(TEXTDATA, "cornell")
    lines = {}
    with open(os.path.join(base, "movie_lines.txt"), encoding="utf-8", errors="replace") as f:
        for row in f:
            parts = row.rstrip("\n").split(" +++$+++ ")
            if len(parts) == 5:
                lines[parts[0].strip()] = parts[4].strip()
    with open(os.path.join(base, "movie_conversations.txt"), encoding="utf-8", errors="replace") as f:
        for row in f:
            parts = row.rstrip("\n").split(" +++$+++ ")
            if len(parts) != 4:
                continue
            ids = eval(parts[3])  # e.g. ['L1045', 'L1044', ...]
            turns = [lines[i] for i in ids if i in lines and lines[i]]
            if 2 <= len(turns) <= 14:
                add_chat("cornell", turns)


def main():
    print("processing NLTK corpora...", flush=True)
    process_nltk()
    print(f"  pretrain docs: {len(pretrain_docs)}, chat dialogues: {len(chat_dialogues)}", flush=True)
    print("processing wikitext2/tinyshakespeare...", flush=True)
    process_wikitext2()
    process_tinyshakespeare()
    print("processing dailydialog/cornell...", flush=True)
    process_dailydialog()
    process_cornell()
    print(f"TOTAL pretrain docs: {len(pretrain_docs)}, chat dialogues: {len(chat_dialogues)}", flush=True)

    os.makedirs(OUTDIR, exist_ok=True)
    rng = random.Random(1337)
    # validation split: ~0.5% of docs
    idx = list(range(len(pretrain_docs)))
    rng.shuffle(idx)
    n_val = max(50, len(idx) // 200)
    val_idx = set(idx[:n_val])
    train_docs = [d for i, d in enumerate(pretrain_docs) if i not in val_idx]
    val_docs = [d for i, d in enumerate(pretrain_docs) if i in val_idx]
    with open(os.path.join(OUTDIR, "pretrain.txt"), "w", encoding="utf-8") as f:
        f.write(DOC_SEP.join(train_docs))
    with open(os.path.join(OUTDIR, "val.txt"), "w", encoding="utf-8") as f:
        f.write(DOC_SEP.join(val_docs))

    # chat -> jsonl with roles
    def to_msgs(turns):
        return [{"role": "user" if i % 2 == 0 else "assistant", "content": t}
                for i, t in enumerate(turns)]
    cidx = list(range(len(chat_dialogues)))
    rng.shuffle(cidx)
    n_cval = max(50, len(cidx) // 100)
    cval = set(cidx[:n_cval])
    with open(os.path.join(OUTDIR, "chat.jsonl"), "w", encoding="utf-8") as f:
        for i, turns in enumerate(chat_dialogues):
            if i not in cval:
                f.write(json.dumps({"messages": to_msgs(turns)}, ensure_ascii=False) + "\n")
    with open(os.path.join(OUTDIR, "val_chat.jsonl"), "w", encoding="utf-8") as f:
        for i, turns in enumerate(chat_dialogues):
            if i in cval:
                f.write(json.dumps({"messages": to_msgs(turns)}, ensure_ascii=False) + "\n")

    for name in ["pretrain.txt", "val.txt", "chat.jsonl", "val_chat.jsonl"]:
        p = os.path.join(OUTDIR, name)
        print(f"{name}: {os.path.getsize(p) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
