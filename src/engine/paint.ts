// Paint.exe inference: text-conditional DDIM diffusion sampler.
// Mirrors training/paint/unet.py exactly (NHWC, OIHW kernels, TF-SAME padding).
import { f32MatVec } from './quant';
import { siluInPlace, softmaxInPlace } from './ops';
import { mulberry32 } from './generate';

export interface PaintTensorMeta {
  name: string;
  shape: number[];
  dtype: 'f32';
  offset: number;
  length: number;
}

export interface PaintConfig {
  name: string;
  arch: string;
  img: number;
  channels: number[];
  nDown: number[];
  midBlocks: number;
  heads: number;
  byteDim: number;
  textDim: number;
  timeDim: number;
  maxText: number;
  tSteps: number;
  betaStart: number;
  betaEnd: number;
  groups: number;
  gnEps: number;
  paramCount: number;
  tensors: PaintTensorMeta[];
  labels: string[];
  training?: { steps: number; valLoss?: number };
}

export interface PaintWeights {
  cfg: PaintConfig;
  tensors: Map<string, { shape: number[]; data: Float32Array }>;
  paramCount: number;
  buffer: ArrayBuffer;
}

export function parsePaintWeights(buf: ArrayBuffer, cfg: PaintConfig): PaintWeights {
  const dv = new DataView(buf);
  const magic = String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3));
  if (magic !== 'V0P1') throw new Error(`bad paint magic: ${JSON.stringify(magic)}`);
  if (dv.getUint32(4, true) !== 1) throw new Error('bad paint version');
  const out = new Map<string, { shape: number[]; data: Float32Array }>();
  for (const t of cfg.tensors) {
    const off = 16 + t.offset;
    const n = t.shape.reduce((a, b) => a * b, 1);
    if (t.length !== n * 4) throw new Error(`paint tensor ${t.name} length mismatch`);
    out.set(t.name, { shape: t.shape, data: new Float32Array(buf, off, n) });
  }
  let paramCount = 0;
  for (const t of cfg.tensors) paramCount += t.shape.reduce((a, b) => a * b, 1);
  return { cfg, tensors: out, paramCount, buffer: buf };
}

function T(w: PaintWeights, name: string): { shape: number[]; data: Float32Array } {
  const t = w.tensors.get(name);
  if (!t) throw new Error(`missing paint tensor ${name}`);
  return t;
}

/** TF-SAME 2-D convolution, NHWC single image, OIHW kernel. */
export function conv2d(x: Float32Array, H: number, W: number, C: number, k: Float32Array, OC: number, stride: number, out: Float32Array): void {
  const K = 3;
  const oH = Math.ceil(H / stride);
  const oW = Math.ceil(W / stride);
  const padH = Math.max((oH - 1) * stride + K - H, 0);
  const padW = Math.max((oW - 1) * stride + K - W, 0);
  const padTop = Math.floor(padH / 2);
  const padLeft = Math.floor(padW / 2);
  for (let oy = 0; oy < oH; oy++) {
    for (let ox = 0; ox < oW; ox++) {
      for (let oc = 0; oc < OC; oc++) {
        let s = 0;
        for (let ky = 0; ky < K; ky++) {
          const iy = oy * stride + ky - padTop;
          if (iy < 0 || iy >= H) continue;
          for (let kx = 0; kx < K; kx++) {
            const ix = ox * stride + kx - padLeft;
            if (ix < 0 || ix >= W) continue;
            const xb = (iy * W + ix) * C;
            const kb = ((oc * C + 0) * K + ky) * K + kx;
            for (let ic = 0; ic < C; ic++) s += x[xb + ic] * k[kb + ic * K * K];
          }
        }
        out[(oy * oW + ox) * OC + oc] = s;
      }
    }
  }
}

/** 1x1 convolution (residual projection). */
export function conv1x1(x: Float32Array, n: number, C: number, k: Float32Array, OC: number, out: Float32Array): void {
  for (let p = 0; p < n; p++) {
    for (let oc = 0; oc < OC; oc++) {
      let s = 0;
      for (let ic = 0; ic < C; ic++) s += x[p * C + ic] * k[oc * C + ic];
      out[p * OC + oc] = s;
    }
  }
}

export function groupNorm(x: Float32Array, C: number, w: Float32Array, b: Float32Array, groups: number, eps: number): void {
  const n = x.length / C;
  const gd = C / groups;
  for (let g = 0; g < groups; g++) {
    let sum = 0;
    let sum2 = 0;
    const cnt = n * gd;
    for (let p = 0; p < n; p++) {
      const base = p * C + g * gd;
      for (let i = 0; i < gd; i++) {
        const v = x[base + i];
        sum += v;
        sum2 += v * v;
      }
    }
    const mu = sum / cnt;
    const va = sum2 / cnt - mu * mu;
    const s = 1 / Math.sqrt(Math.max(va, 0) + eps);
    for (let p = 0; p < n; p++) {
      const base = p * C + g * gd;
      for (let i = 0; i < gd; i++) {
        const c = g * gd + i;
        x[base + i] = (x[base + i] - mu) * s * w[c] + b[c];
      }
    }
  }
}

export function upsample2x(x: Float32Array, H: number, W: number, C: number): Float32Array {
  const out = new Float32Array(H * 2 * W * 2 * C);
  for (let y = 0; y < H; y++) {
    for (let xx = 0; xx < W; xx++) {
      const src = (y * W + xx) * C;
      for (let dy = 0; dy < 2; dy++) {
        for (let dx = 0; dx < 2; dx++) {
          const dst = ((y * 2 + dy) * W * 2 + (xx * 2 + dx)) * C;
          for (let c = 0; c < C; c++) out[dst + c] = x[src + c];
        }
      }
    }
  }
  return out;
}

export function timeEmbed(t: number, dim = 64): Float32Array {
  const half = dim / 2;
  const out = new Float32Array(dim);
  for (let i = 0; i < half; i++) {
    const div = Math.exp(i * (-(Math.log(10000) / half)));
    const a = t * div;
    out[i] = Math.sin(a);
    out[half + i] = Math.cos(a);
  }
  return out;
}

const textEncoder = new TextEncoder();

export function textToBytes(prompt: string, maxLen: number): { ids: Uint8Array; mask: Float32Array } {
  const raw = textEncoder.encode(prompt).slice(0, maxLen);
  const ids = new Uint8Array(maxLen);
  const mask = new Float32Array(maxLen);
  ids.set(raw, 0);
  for (let i = 0; i < raw.length; i++) mask[i] = 1;
  return { ids, mask };
}

export function encodeText(w: PaintWeights, prompt: string): Float32Array {
  const cfg = w.cfg;
  const { ids, mask } = textToBytes(prompt, cfg.maxText);
  const emb = T(w, 'byte_emb').data;
  const pooled = new Float32Array(cfg.byteDim);
  let cnt = 0;
  for (let i = 0; i < cfg.maxText; i++) {
    if (mask[i] === 0) continue;
    cnt++;
    const b = ids[i] * cfg.byteDim;
    for (let j = 0; j < cfg.byteDim; j++) pooled[j] += emb[b + j];
  }
  const inv = 1 / Math.max(cnt, 1);
  for (let j = 0; j < pooled.length; j++) pooled[j] *= inv;
  const m1 = T(w, 'txt_mlp1');
  const h = new Float32Array(cfg.textDim);
  f32MatVec(m1.data, m1.shape[0], m1.shape[1], pooled, h);
  const b1 = T(w, 'txt_mlp1_b').data;
  for (let j = 0; j < h.length; j++) h[j] += b1[j];
  siluInPlace(h);
  const m2 = T(w, 'txt_mlp2');
  const out = new Float32Array(cfg.textDim);
  f32MatVec(m2.data, m2.shape[0], m2.shape[1], h, out);
  const b2 = T(w, 'txt_mlp2_b').data;
  for (let j = 0; j < out.length; j++) out[j] += b2[j];
  return out;
}

function linear(w: PaintWeights, name: string, x: Float32Array, biasName?: string): Float32Array {
  const m = T(w, name);
  const out = new Float32Array(m.shape[0]);
  f32MatVec(m.data, m.shape[0], m.shape[1], x, out);
  const b = w.tensors.get(biasName ?? name + '_b');
  if (b) for (let i = 0; i < out.length; i++) out[i] += b.data[i];
  return out;
}

function resBlock(w: PaintWeights, prefix: string, x: Float32Array, H: number, Wd: number, C: number, temb: Float32Array): Float32Array {
  const cfg = w.cfg;
  const h = new Float32Array(x);
  groupNorm(h, C, T(w, prefix + '.gn1_w').data, T(w, prefix + '.gn1_b').data, cfg.groups, cfg.gnEps);
  siluInPlace(h);
  const c1 = T(w, prefix + '.conv1');
  const h2 = new Float32Array(H * Wd * c1.shape[0]);
  conv2d(h, H, Wd, C, c1.data, c1.shape[0], 1, h2);
  const C2 = c1.shape[0];
  groupNorm(h2, C2, T(w, prefix + '.gn2_w').data, T(w, prefix + '.gn2_b').data, cfg.groups, cfg.gnEps);
  const film = linear(w, prefix + '.film_w', temb);
  for (let p = 0; p < H * Wd; p++) {
    for (let c = 0; c < C2; c++) {
      h2[p * C2 + c] = h2[p * C2 + c] * (1 + film[c]) + film[C2 + c];
    }
  }
  siluInPlace(h2);
  const c2 = T(w, prefix + '.conv2');
  const h3 = new Float32Array(H * Wd * C2);
  conv2d(h2, H, Wd, C2, c2.data, C2, 1, h3);
  const res = w.tensors.get(prefix + '.res');
  if (res) {
    const r = new Float32Array(H * Wd * C2);
    conv1x1(x, H * Wd, C, res.data, C2, r);
    for (let i = 0; i < r.length; i++) h3[i] += r[i];
  } else {
    for (let i = 0; i < h3.length; i++) h3[i] += x[i];
  }
  return h3;
}

function attnBlock(w: PaintWeights, x: Float32Array): Float32Array {
  const cfg = w.cfg;
  const C = cfg.channels[2];
  const N = 64;
  const heads = cfg.heads;
  const hd = C / heads;
  const h = new Float32Array(x);
  groupNorm(h, C, T(w, 'mid.attn.gn_w').data, T(w, 'mid.attn.gn_b').data, cfg.groups, cfg.gnEps);
  const pos = T(w, 'mid.attn.pos').data;
  for (let i = 0; i < h.length; i++) h[i] += pos[i];
  const q = new Float32Array(N * C);
  const k = new Float32Array(N * C);
  const v = new Float32Array(N * C);
  const Wq = T(w, 'mid.attn.wq');
  const Wk = T(w, 'mid.attn.wk');
  const Wv = T(w, 'mid.attn.wv');
  const tmp = new Float32Array(C);
  const row = new Float32Array(C);
  for (let p = 0; p < N; p++) {
    row.set(h.subarray(p * C, p * C + C));
    f32MatVec(Wq.data, C, C, row, tmp);
    q.set(tmp, p * C);
    f32MatVec(Wk.data, C, C, row, tmp);
    k.set(tmp, p * C);
    f32MatVec(Wv.data, C, C, row, tmp);
    v.set(tmp, p * C);
  }
  const o = new Float32Array(N * C);
  const sc = new Float32Array(N);
  const scale = 1 / Math.sqrt(hd);
  for (let hh = 0; hh < heads; hh++) {
    for (let t = 0; t < N; t++) {
      for (let s = 0; s < N; s++) {
        let dot = 0;
        for (let i = 0; i < hd; i++) dot += q[(t * heads + hh) * hd + i] * k[(s * heads + hh) * hd + i];
        sc[s] = dot * scale;
      }
      softmaxInPlace(sc);
      for (let i = 0; i < hd; i++) {
        let acc = 0;
        for (let s = 0; s < N; s++) acc += sc[s] * v[(s * heads + hh) * hd + i];
        o[(t * heads + hh) * hd + i] = acc;
      }
    }
  }
  // NOTE: q/k/v above are laid out [N, heads, hd] flattened; o matches.
  const Wo = T(w, 'mid.attn.wo');
  const out = new Float32Array(N * C);
  for (let p = 0; p < N; p++) {
    row.set(o.subarray(p * C, p * C + C));
    f32MatVec(Wo.data, C, C, row, tmp);
    out.set(tmp, p * C);
  }
  for (let i = 0; i < out.length; i++) out[i] += x[i];
  return out;
}

/** UNet forward: x_t [H*W*3] -> predicted noise. cond is the 128-d text vector. */
export function applyUnet(w: PaintWeights, x: Float32Array, t: number, cond: Float32Array): Float32Array {
  const cfg = w.cfg;
  const [c0, c1, c2] = cfg.channels;
  let te = linear(w, 't_mlp1', timeEmbed(t));
  siluInPlace(te);
  te = linear(w, 't_mlp2', te);
  const temb = new Float32Array(te.length + cond.length);
  temb.set(te, 0);
  temb.set(cond, te.length);

  const stem = T(w, 'stem');
  let h = new Float32Array(32 * 32 * c0);
  conv2d(x, 32, 32, 3, stem.data, c0, 1, h);
  const skips: Float32Array[] = [];
  for (let i = 0; i < cfg.nDown[0]; i++) h = resBlock(w, `dn0.b${i}`, h, 32, 32, c0, temb);
  skips.push(h);
  const d0 = T(w, 'dn0.down');
  const h16 = new Float32Array(16 * 16 * c1);
  conv2d(h, 32, 32, c0, d0.data, c1, 2, h16);
  h = h16;
  for (let i = 0; i < cfg.nDown[1]; i++) h = resBlock(w, `dn1.b${i}`, h, 16, 16, c1, temb);
  skips.push(h);
  const d1 = T(w, 'dn1.down');
  const h8 = new Float32Array(8 * 8 * c2);
  conv2d(h, 16, 16, c1, d1.data, c2, 2, h8);
  h = h8;
  for (let i = 0; i < cfg.midBlocks; i++) h = resBlock(w, `mid.b${i}`, h, 8, 8, c2, temb);
  h = attnBlock(w, h);
  for (let i = 0; i < cfg.midBlocks; i++) h = resBlock(w, `mid.c${i}`, h, 8, 8, c2, temb);
  h = upsample2x(h, 8, 8, c2);
  h = concatCh(h, skips.pop()!, c2, c1);
  h = resBlock(w, 'up1.b0', h, 16, 16, c2 + c1, temb);
  h = upsample2x(h, 16, 16, c1);
  h = concatCh(h, skips.pop()!, c1, c0);
  h = resBlock(w, 'up0.b0', h, 32, 32, c1 + c0, temb);
  groupNorm(h, c0, T(w, 'out_gn_w').data, T(w, 'out_gn_b').data, cfg.groups, cfg.gnEps);
  siluInPlace(h);
  const oc = T(w, 'out_conv');
  const out = new Float32Array(32 * 32 * 3);
  conv2d(h, 32, 32, c0, oc.data, 3, 1, out);
  return out;
}

function concatCh(a: Float32Array, b: Float32Array, ca: number, cb: number): Float32Array {
  const n = a.length / ca;
  const out = new Float32Array(n * (ca + cb));
  for (let p = 0; p < n; p++) {
    out.set(a.subarray(p * ca, p * ca + ca), p * (ca + cb));
    out.set(b.subarray(p * cb, p * cb + cb), p * (ca + cb) + ca);
  }
  return out;
}

export function diffusionSchedule(cfg: PaintConfig): { abars: Float64Array } {
  const T = cfg.tSteps;
  const abars = new Float64Array(T);
  let cum = 1;
  for (let i = 0; i < T; i++) {
    const b = cfg.betaStart + ((cfg.betaEnd - cfg.betaStart) * i) / (T - 1);
    cum *= 1 - b;
    abars[i] = cum;
  }
  return { abars };
}

export function ddimTimesteps(steps: number, tSteps: number): number[] {
  const ts: number[] = [];
  for (let i = 0; i < steps; i++) ts.push(Math.round(((tSteps - 1) * i) / (steps - 1)));
  return ts;
}

export function randn(rng: () => number, n: number): Float32Array {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 2) {
    let u1 = 0;
    while (u1 === 0) u1 = rng();
    const u2 = rng();
    const r = Math.sqrt(-2 * Math.log(u1));
    const th = 2 * Math.PI * u2;
    out[i] = r * Math.cos(th);
    if (i + 1 < n) out[i + 1] = r * Math.sin(th);
  }
  return out;
}

export interface DdimCallbacks {
  onStep?: (done: number, total: number, x0preview: Float32Array) => void;
  shouldStop?: () => boolean;
  yieldNow?: () => Promise<void>;
}

/** Run DDIM from explicit noise/timesteps (parity-friendly core). */
export async function ddimRun(
  w: PaintWeights,
  xInit: Float32Array,
  cond: Float32Array | null,
  timesteps: number[],
  guidance: number,
  cb: DdimCallbacks = {},
): Promise<{ image: Float32Array; stopped: boolean }> {
  const { abars } = diffusionSchedule(w.cfg);
  const nullCond = T(w, 'null_cond').data;
  const c = cond ?? nullCond;
  let x = new Float32Array(xInit);
  const useCfg = guidance !== 1 && cond !== null;
  for (let i = timesteps.length - 1; i >= 0; i--) {
    if (cb.shouldStop?.()) return { image: toImage(x), stopped: true };
    const t = timesteps[i];
    const tPrev = i > 0 ? timesteps[i - 1] : -1;
    const abT = abars[t];
    const abPrev = tPrev >= 0 ? abars[tPrev] : 1;
    const eCond = applyUnet(w, x, t, c);
    let eps = eCond;
    if (useCfg) {
      const eUn = applyUnet(w, x, t, nullCond);
      eps = new Float32Array(eCond.length);
      for (let j = 0; j < eps.length; j++) eps[j] = eUn[j] + guidance * (eCond[j] - eUn[j]);
    }
    const s1 = Math.sqrt(1 - abT);
    const s2 = Math.sqrt(abT);
    const s3 = Math.sqrt(abPrev);
    const s4 = Math.sqrt(Math.max(0, 1 - abPrev));
    const x0 = new Float32Array(x.length);
    for (let j = 0; j < x.length; j++) {
      x0[j] = (x[j] - s1 * eps[j]) / s2;
      x[j] = s3 * x0[j] + s4 * eps[j];
    }
    cb.onStep?.(timesteps.length - i, timesteps.length, toImage(x0));
    if (cb.yieldNow) await cb.yieldNow();
  }
  return { image: toImage(x), stopped: false };
}

function toImage(x: Float32Array): Float32Array {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const v = (x[i] + 1) / 2;
    out[i] = v < 0 ? 0 : v > 1 ? 1 : v;
  }
  return out;
}

/** Full sampling from a text prompt + seed. */
export async function ddimSample(
  w: PaintWeights,
  prompt: string,
  opts: { steps: number; seed: number; guidance: number },
  cb: DdimCallbacks = {},
): Promise<{ image: Float32Array; stopped: boolean }> {
  const cond = encodeText(w, prompt);
  const rng = mulberry32(opts.seed >>> 0);
  const xInit = randn(rng, 32 * 32 * 3);
  return ddimRun(w, xInit, cond, ddimTimesteps(opts.steps, w.cfg.tSteps), opts.guidance, cb);
}
