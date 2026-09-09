"""Reference byte-level BPE tokenizer (used by training + fixture generation).

Must stay numerically identical to the TypeScript port in src/engine/tokenizer.ts.
Vocab: 0..255 raw bytes, 256..4092 merges, 4093 <bos>, 4094 <eos>, 4095 <pad>.
"""
import json
import os
import re

VOCAB_SIZE = 4096
BOS_ID = 4093
EOS_ID = 4094
PAD_ID = 4095


class BPETokenizer:
    def __init__(self, path=None):
        if path is None:
            path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "tokenizer.json")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        merges = data["merges"]
        self.vocab_size = data.get("vocab_size", VOCAB_SIZE)
        self.rank = {}
        self.vocab_bytes = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(merges):
            new_id = 256 + i
            self.rank[(a, b)] = i
            self.vocab_bytes[new_id] = self.vocab_bytes[a] + self.vocab_bytes[b]
        self.vocab_bytes[BOS_ID] = b""
        self.vocab_bytes[EOS_ID] = b""
        self.vocab_bytes[PAD_ID] = b""

    def _encode_piece(self, piece_bytes: bytes):
        ids = list(piece_bytes)
        if len(ids) < 2:
            return ids
        while True:
            best_rank = None
            best_pos = -1
            for i in range(len(ids) - 1):
                r = self.rank.get((ids[i], ids[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_pos = i
            if best_rank is None:
                break
            new_id = 256 + best_rank
            ids = ids[:best_pos] + [new_id] + ids[best_pos + 2:]
            if len(ids) < 2:
                break
        return ids

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False):
        out = []
        if add_bos:
            out.append(BOS_ID)
        for piece in re.findall(r"\S+|\s", text):
            out.extend(self._encode_piece(piece.encode("utf-8")))
        if add_eos:
            out.append(EOS_ID)
        return out

    def decode(self, ids, skip_specials: bool = True) -> str:
        buf = bytearray()
        for i in ids:
            if skip_specials and i >= BOS_ID:
                continue
            b = self.vocab_bytes.get(int(i))
            if b is None:
                continue
            buf.extend(b)
        return bytes(buf).decode("utf-8", errors="replace")

    def encode_chat(self, messages):
        """Format + encode a dialogue.

        <bos>User: ...\\nAssistant: ...\\nUser: ...\\nAssistant: ...<eos>
        Returns (ids, weights) where weights mark assistant-span tokens (for
        SFT loss masking).
        """
        ids = [BOS_ID]
        weights = [0.0]
        for m in messages:
            role = m.get("role", "user")
            prefix = "User: " if role == "user" else "Assistant: "
            chunk = prefix + m.get("content", "")
            tids = self.encode(chunk)
            w = 1.0 if role == "assistant" else 0.0
            # The \\n separator after a user turn belongs to no one; after an
            # assistant turn it terminates the response (train it too).
            ids.extend(tids)
            weights.extend([w] * len(tids))
            nl = self.encode("\n")
            ids.extend(nl)
            weights.extend([w] * len(nl))
        ids.append(EOS_ID)
        weights.append(1.0)
        return ids, weights

    def chat_prompt(self, user_text: str):
        """Token prefix the UI feeds the model for a chat turn."""
        return [BOS_ID] + self.encode("User: " + user_text + "\nAssistant: ")
