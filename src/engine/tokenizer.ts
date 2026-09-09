// Byte-level BPE tokenizer — exact port of training/tokenizer/bpe.py.
// Vocab: 0..255 raw bytes, 256..4092 merges, 4093 <bos>, 4094 <eos>, 4095 <pad>.
import type { TokenizerData } from './types';

export interface ChatMessage {
  role: string;
  content: string;
}

export class BPETokenizer {
  readonly vocabSize: number;
  readonly bos: number;
  readonly eos: number;
  readonly pad: number;
  private rank = new Map<number, number>();
  private vocabBytes: Uint8Array[] = [];
  private encoder = new TextEncoder();
  private decoder = new TextDecoder('utf-8', { fatal: false });

  constructor(data: TokenizerData) {
    this.vocabSize = data.vocab_size;
    this.bos = data.special.bos;
    this.eos = data.special.eos;
    this.pad = data.special.pad;
    for (let i = 0; i < 256; i++) this.vocabBytes[i] = Uint8Array.of(i);
    data.merges.forEach(([a, b], i) => {
      const id = 256 + i;
      this.rank.set(a * 4096 + b, i);
      const ba = this.vocabBytes[a];
      const bb = this.vocabBytes[b];
      const out = new Uint8Array(ba.length + bb.length);
      out.set(ba, 0);
      out.set(bb, ba.length);
      this.vocabBytes[id] = out;
    });
    this.vocabBytes[this.bos] = new Uint8Array(0);
    this.vocabBytes[this.eos] = new Uint8Array(0);
    this.vocabBytes[this.pad] = new Uint8Array(0);
  }

  private encodePiece(bytes: Uint8Array): number[] {
    let ids: number[] = Array.from(bytes);
    if (ids.length < 2) return ids;
    for (;;) {
      let bestRank: number | undefined;
      let bestPos = -1;
      for (let i = 0; i < ids.length - 1; i++) {
        const r = this.rank.get(ids[i] * 4096 + ids[i + 1]);
        if (r !== undefined && (bestRank === undefined || r < bestRank)) {
          bestRank = r;
          bestPos = i;
        }
      }
      if (bestRank === undefined) break;
      const merged = ids.slice(0, bestPos);
      merged.push(256 + bestRank);
      for (let i = bestPos + 2; i < ids.length; i++) merged.push(ids[i]);
      ids = merged;
      if (ids.length < 2) break;
    }
    return ids;
  }

  encode(text: string, addBos = false, addEos = false): number[] {
    const out: number[] = [];
    if (addBos) out.push(this.bos);
    const pieces = text.match(/\S+|\s/gu) ?? [];
    for (const p of pieces) {
      const ids = this.encodePiece(this.encoder.encode(p));
      for (const id of ids) out.push(id);
    }
    if (addEos) out.push(this.eos);
    return out;
  }

  decode(ids: ArrayLike<number>, skipSpecials = true): string {
    let total = 0;
    const parts: Uint8Array[] = [];
    for (let k = 0; k < ids.length; k++) {
      const id = ids[k];
      if (skipSpecials && id >= this.bos) continue;
      const b = this.vocabBytes[id];
      if (!b) continue;
      parts.push(b);
      total += b.length;
    }
    const buf = new Uint8Array(total);
    let o = 0;
    for (const p of parts) {
      buf.set(p, o);
      o += p.length;
    }
    return this.decoder.decode(buf);
  }

  /** ids of one decoded token (for debug display) */
  decodeToken(id: number): string {
    return this.decode([id]);
  }

  encodeChat(messages: ChatMessage[]): { ids: number[]; weights: number[] } {
    const ids: number[] = [this.bos];
    const weights: number[] = [0];
    for (const m of messages) {
      const prefix = m.role === 'user' ? 'User: ' : 'Assistant: ';
      const tids = this.encode(prefix + m.content);
      const w = m.role === 'assistant' ? 1 : 0;
      for (const t of tids) {
        ids.push(t);
        weights.push(w);
      }
      const nl = this.encode('\n');
      for (const t of nl) {
        ids.push(t);
        weights.push(w);
      }
    }
    ids.push(this.eos);
    weights.push(1);
    return { ids, weights };
  }

  chatPrompt(userText: string): number[] {
    return [this.bos, ...this.encode('User: ' + userText + '\nAssistant: ')];
  }

  /** Multi-turn prompt: full transcript + trailing assistant prefix (no eos). */
  promptFor(messages: { role: string; text: string }[]): number[] {
    const ids: number[] = [this.bos];
    for (const m of messages) {
      const prefix = m.role === 'user' ? 'User: ' : 'Assistant: ';
      const tids = this.encode(prefix + m.text + '\n');
      for (const t of tids) ids.push(t);
    }
    const tail = this.encode('Assistant: ');
    for (const t of tail) ids.push(t);
    return ids;
  }
}

/** Incremental UTF-8 decoder for streaming (handles multi-byte splits). */
export class StreamingDecoder {
  private dec = new TextDecoder('utf-8', { fatal: false });
  push(bytes: Uint8Array): string {
    return this.dec.decode(bytes, { stream: true });
  }
  flush(): string {
    return this.dec.decode();
  }
}
