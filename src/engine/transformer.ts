// Decoder-only Transformer forward pass (mirrors training/common/model.py exactly):
// RMSNorm -> RoPE MHA (causal) -> residual -> RMSNorm -> SwiGLU MLP -> residual,
// final RMSNorm, tied-embedding logits. Supports full-sequence prefill and
// single-step generation with a KV cache.
import type { LoadedModel, ModelConfig, WeightTensor } from './types';
import { f32MatVec, q8MatVec } from './quant';
import { addInto, mulInto, rmsNorm, siluInPlace } from './ops';

export function matVecW(w: WeightTensor, x: Float32Array, out: Float32Array): void {
  if (w.kind === 'f32') {
    f32MatVec(w.data, w.shape[0], w.shape[1], x, out);
  } else {
    q8MatVec(w, x, out);
  }
}

function needF32(m: LoadedModel, name: string): Float32Array {
  const t = m.weights.get(name);
  if (!t || t.kind !== 'f32') throw new Error(`missing f32 tensor ${name}`);
  return t.data;
}

function needW(m: LoadedModel, name: string): WeightTensor {
  const t = m.weights.get(name);
  if (!t) throw new Error(`missing tensor ${name}`);
  return t;
}

/** Rotate one head vector in place (NeoX/halves RoPE), position pos. */
function ropeHead(v: Float32Array, off: number, hd: number, pos: number, theta: number): void {
  const half = hd >>> 1;
  for (let i = 0; i < half; i++) {
    const ang = pos * theta ** ((2 * i) / -hd);
    const c = Math.cos(ang);
    const s = Math.sin(ang);
    const a = v[off + i];
    const b = v[off + half + i];
    v[off + i] = a * c - b * s;
    v[off + half + i] = b * c + a * s;
  }
}

export class KVCache {
  readonly pos = { n: 0 };
  readonly k: Float32Array[] = [];
  readonly v: Float32Array[] = [];
  constructor(public cfg: ModelConfig) {
    for (let l = 0; l < cfg.nLayer; l++) {
      this.k.push(new Float32Array(cfg.ctx * cfg.dModel));
      this.v.push(new Float32Array(cfg.ctx * cfg.dModel));
    }
  }
  reset(): void {
    this.pos.n = 0;
  }
  get len(): number {
    return this.pos.n;
  }
}

export interface FullLogits {
  logits: Float32Array; // [T*V] row-major
  T: number;
  V: number;
}

/** Full causal forward pass. Returns logits for every position. */
export function forwardFull(m: LoadedModel, ids: ArrayLike<number>): FullLogits {
  const cfg = m.cfg;
  const T = ids.length;
  const V = cfg.vocab;
  const d = cfg.dModel;
  const H = cfg.nHead;
  const hd = d / H;
  if (T === 0) throw new Error('empty input');
  if (T > cfg.ctx) throw new Error(`input length ${T} exceeds context ${cfg.ctx}`);

  const emb = needF32(m, 'tok_emb');
  const x = new Float32Array(T * d);
  for (let t = 0; t < T; t++) {
    const id = ids[t];
    if (id < 0 || id >= V) throw new Error(`token id ${id} out of range`);
    x.set(emb.subarray(id * d, id * d + d), t * d);
  }
  const h = new Float32Array(T * d);
  const q = new Float32Array(T * d);
  const k = new Float32Array(T * d);
  const v = new Float32Array(T * d);
  const o = new Float32Array(T * d);
  const tmp = new Float32Array(Math.max(T * d, cfg.dFF * T));
  const gate = new Float32Array(T * cfg.dFF);
  const up = new Float32Array(T * cfg.dFF);
  const scores = new Float32Array(H * T * T);
  const scale = 1 / Math.sqrt(hd);

  const wq: WeightTensor[] = [];
  const wk: WeightTensor[] = [];
  const wv: WeightTensor[] = [];
  const wo: WeightTensor[] = [];
  const wg: WeightTensor[] = [];
  const wu: WeightTensor[] = [];
  const wd: WeightTensor[] = [];
  const an: Float32Array[] = [];
  const mn: Float32Array[] = [];
  for (let l = 0; l < cfg.nLayer; l++) {
    wq.push(needW(m, `L${l}.wq`));
    wk.push(needW(m, `L${l}.wk`));
    wv.push(needW(m, `L${l}.wv`));
    wo.push(needW(m, `L${l}.wo`));
    wg.push(needW(m, `L${l}.wgate`));
    wu.push(needW(m, `L${l}.wup`));
    wd.push(needW(m, `L${l}.wdown`));
    an.push(needF32(m, `L${l}.attn_norm`));
    mn.push(needF32(m, `L${l}.mlp_norm`));
  }
  const xRow = new Float32Array(d);
  const hRow = new Float32Array(d);
  const qRow = new Float32Array(d);
  const kRow = new Float32Array(d);
  const vRow = new Float32Array(d);
  const oRow = new Float32Array(d);

  for (let l = 0; l < cfg.nLayer; l++) {
    for (let t = 0; t < T; t++) {
      xRow.set(x.subarray(t * d, t * d + d));
      rmsNorm(xRow, an[l], cfg.rmsEps, hRow);
      h.set(hRow, t * d);
      matVecW(wq[l], hRow, qRow);
      matVecW(wk[l], hRow, kRow);
      matVecW(wv[l], hRow, vRow);
      q.set(qRow, t * d);
      k.set(kRow, t * d);
      v.set(vRow, t * d);
    }
    // RoPE (positions 0..T-1), head-major layout inside each row
    for (let t = 0; t < T; t++) {
      for (let hh = 0; hh < H; hh++) {
        ropeHead(q, t * d + hh * hd, hd, t, cfg.ropeTheta);
        ropeHead(k, t * d + hh * hd, hd, t, cfg.ropeTheta);
      }
    }
    // Causal attention per head
    for (let hh = 0; hh < H; hh++) {
      const sb = hh * T * T;
      for (let t = 0; t < T; t++) {
        const qb = t * d + hh * hd;
        let rowMax = -Infinity;
        for (let s = 0; s <= t; s++) {
          const kb = s * d + hh * hd;
          let dot = 0;
          for (let i = 0; i < hd; i++) dot += q[qb + i] * k[kb + i];
          const sc = dot * scale;
          scores[sb + t * T + s] = sc;
          if (sc > rowMax) rowMax = sc;
        }
        let sum = 0;
        for (let s = 0; s <= t; s++) {
          const e = Math.exp(scores[sb + t * T + s] - rowMax);
          scores[sb + t * T + s] = e;
          sum += e;
        }
        const inv = 1 / sum;
        const ob = t * d + hh * hd;
        for (let i = 0; i < hd; i++) o[ob + i] = 0;
        for (let s = 0; s <= t; s++) {
          const wgt = scores[sb + t * T + s] * inv;
          const vb = s * d + hh * hd;
          for (let i = 0; i < hd; i++) o[ob + i] += wgt * v[vb + i];
        }
      }
    }
    for (let t = 0; t < T; t++) {
      oRow.set(o.subarray(t * d, t * d + d));
      matVecW(wo[l], oRow, hRow);
      for (let i = 0; i < d; i++) x[t * d + i] += hRow[i];
    }
    // MLP
    const ff = cfg.dFF;
    for (let t = 0; t < T; t++) {
      xRow.set(x.subarray(t * d, t * d + d));
      rmsNorm(xRow, mn[l], cfg.rmsEps, hRow);
      const gRow = gate.subarray(t * ff, t * ff + ff);
      const uRow = up.subarray(t * ff, t * ff + ff);
      matVecW(wg[l], hRow, gRow as Float32Array);
      matVecW(wu[l], hRow, uRow as Float32Array);
      siluInPlace(gRow as Float32Array);
      mulInto(gRow as Float32Array, uRow as Float32Array, tmp.subarray(0, ff) as Float32Array);
      const prod = tmp.subarray(0, ff);
      matVecW(wd[l], prod as Float32Array, hRow);
      for (let i = 0; i < d; i++) x[t * d + i] += hRow[i];
    }
  }
  const fin = needF32(m, 'final_norm');
  const logits = new Float32Array(T * V);
  const lRow = new Float32Array(V);
  for (let t = 0; t < T; t++) {
    xRow.set(x.subarray(t * d, t * d + d));
    rmsNorm(xRow, fin, cfg.rmsEps, hRow);
    f32MatVec(emb, V, d, hRow, lRow);
    logits.set(lRow, t * V);
  }
  return { logits, T, V };
}

/** One autoregressive step at cache position `pos`. Returns logits (V,). */
export function forwardStep(m: LoadedModel, id: number, pos: number, cache: KVCache): Float32Array {
  const cfg = m.cfg;
  const d = cfg.dModel;
  const H = cfg.nHead;
  const hd = d / H;
  const ff = cfg.dFF;
  if (pos >= cfg.ctx) throw new Error('context exhausted');
  const emb = needF32(m, 'tok_emb');
  const x = new Float32Array(d);
  x.set(emb.subarray(id * d, id * d + d));
  const hh = new Float32Array(d);
  const qv = new Float32Array(d);
  const kv = new Float32Array(d);
  const vv = new Float32Array(d);
  const ov = new Float32Array(d);
  const g = new Float32Array(ff);
  const u = new Float32Array(ff);
  const sc = new Float32Array(cfg.ctx);
  const scale = 1 / Math.sqrt(hd);

  for (let l = 0; l < cfg.nLayer; l++) {
    rmsNorm(x, needF32(m, `L${l}.attn_norm`), cfg.rmsEps, hh);
    matVecW(needW(m, `L${l}.wq`), hh, qv);
    matVecW(needW(m, `L${l}.wk`), hh, kv);
    matVecW(needW(m, `L${l}.wv`), hh, vv);
    for (let hIdx = 0; hIdx < H; hIdx++) {
      ropeHead(qv, hIdx * hd, hd, pos, cfg.ropeTheta);
      ropeHead(kv, hIdx * hd, hd, pos, cfg.ropeTheta);
    }
    cache.k[l].set(kv, pos * d);
    cache.v[l].set(vv, pos * d);
    const kc = cache.k[l];
    const vc = cache.v[l];
    for (let hIdx = 0; hIdx < H; hIdx++) {
      const qb = hIdx * hd;
      let rowMax = -Infinity;
      for (let s = 0; s <= pos; s++) {
        const kb = s * d + hIdx * hd;
        let dot = 0;
        for (let i = 0; i < hd; i++) dot += qv[qb + i] * kc[kb + i];
        const v2 = dot * scale;
        sc[s] = v2;
        if (v2 > rowMax) rowMax = v2;
      }
      let sum = 0;
      for (let s = 0; s <= pos; s++) {
        const e = Math.exp(sc[s] - rowMax);
        sc[s] = e;
        sum += e;
      }
      const inv = 1 / sum;
      const ob = hIdx * hd;
      for (let i = 0; i < hd; i++) ov[ob + i] = 0;
      for (let s = 0; s <= pos; s++) {
        const wgt = sc[s] * inv;
        const vb = s * d + hIdx * hd;
        for (let i = 0; i < hd; i++) ov[ob + i] += wgt * vc[vb + i];
      }
    }
    matVecW(needW(m, `L${l}.wo`), ov, hh);
    addInto(x, hh);
    rmsNorm(x, needF32(m, `L${l}.mlp_norm`), cfg.rmsEps, hh);
    matVecW(needW(m, `L${l}.wgate`), hh, g);
    matVecW(needW(m, `L${l}.wup`), hh, u);
    siluInPlace(g);
    mulInto(g, u, u);
    matVecW(needW(m, `L${l}.wdown`), u, hh);
    addInto(x, hh);
  }
  rmsNorm(x, needF32(m, 'final_norm'), cfg.rmsEps, hh);
  const logits = new Float32Array(cfg.vocab);
  f32MatVec(emb, cfg.vocab, d, hh, logits);
  cache.pos.n = pos + 1;
  return logits;
}

/** Prefill the KV cache with a prompt; returns logits of the last position. */
export function prefillCache(m: LoadedModel, ids: ArrayLike<number>, cache: KVCache): Float32Array {
  cache.reset();
  let logits = new Float32Array(m.cfg.vocab);
  for (let t = 0; t < ids.length; t++) {
    logits = forwardStep(m, ids[t], t, cache);
  }
  return logits;
}
