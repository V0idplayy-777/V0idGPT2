// Q8_0-style weight quantization (llama.cpp-compatible blocking) + GEMV kernels.
//
// Format (weights.q8.bin):
//   bytes 0..3   magic "V0Q8"
//   bytes 4..7   version u32 LE (=1)
//   bytes 8..15  reserved
//   then tensor payloads, each starting at a 16-byte aligned offset:
//     f32 tensor: rows*cols float32 LE
//     q8  tensor: rows*stride int8, then rows*(stride/32) fp16 scales LE,
//                where stride = ceil(cols/32)*32 (zero-padded)
// Tensor table (name/shape/dtype/offset/length) lives in config.json.
//
// LAYOUT CONVENTION: all 2-D linear weights are stored OUTPUT-MAJOR, i.e.
// shape [out, in] (the exporter transposes JAX [in, out] matrices). Every
// GEMV in this engine therefore computes plain y = Wx. The embedding table
// is [vocab, d] and works with the same y = Wx kernel.
import type { ModelConfig, Q8Tensor, WeightTensor } from './types';

export const Q8_BLOCK = 32;

export function f16ToF32(bits: number): number {
  const s = (bits & 0x8000) >> 15;
  const e = (bits & 0x7c00) >> 10;
  const f = bits & 0x03ff;
  if (e === 0) {
    if (f === 0) return s ? -0 : 0;
    let mant = f;
    let exp = -14;
    while ((mant & 0x0400) === 0) {
      mant <<= 1;
      exp -= 1;
    }
    mant &= 0x03ff;
    return (s ? -1 : 1) * (1 + mant / 1024) * 2 ** exp;
  }
  if (e === 31) return f === 0 ? (s ? -Infinity : Infinity) : NaN;
  return (s ? -1 : 1) * (1 + f / 1024) * 2 ** (e - 15);
}

export function f32ToF16(v: number): number {
  // Round-to-nearest-even-ish fp16 encoder (used by tests/reference only).
  if (Number.isNaN(v)) return 0x7e00;
  const s = v < 0 || Object.is(v, -0) ? 0x8000 : 0;
  const a = Math.abs(v);
  if (a === Infinity) return s | 0x7c00;
  if (a < 2 ** -24) return s; // underflow -> signed zero
  let e: number;
  let m: number;
  if (a < 2 ** -14) {
    // subnormal
    m = Math.round(a / 2 ** -24);
    return s | m;
  }
  e = Math.floor(Math.log2(a));
  if (e > 15) return s | 0x7c00;
  m = Math.round((a / 2 ** e - 1) * 1024);
  if (m === 1024) {
    m = 0;
    e += 1;
    if (e > 15) return s | 0x7c00;
  }
  return s | ((e + 15) << 10) | m;
}

export function parseWeights(buf: ArrayBuffer, cfg: ModelConfig): Map<string, WeightTensor> {
  const dv = new DataView(buf);
  const magic = String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3));
  if (magic !== 'V0Q8') throw new Error(`bad weights magic: ${JSON.stringify(magic)}`);
  const version = dv.getUint32(4, true);
  if (version !== 1) throw new Error(`unsupported weights version: ${version}`);
  const DATA_START = 16;
  const out = new Map<string, WeightTensor>();
  for (const t of cfg.tensors) {
    const off = DATA_START + t.offset;
    if (off % 16 !== 0) throw new Error(`tensor ${t.name} misaligned`);
    if (t.dtype === 'f32') {
      const n = t.shape.reduce((a, b) => a * b, 1);
      if (t.length !== n * 4) throw new Error(`tensor ${t.name} length mismatch`);
      out.set(t.name, { kind: 'f32', shape: t.shape, data: new Float32Array(buf, off, n) });
    } else if (t.dtype === 'q8') {
      if (t.shape.length !== 2) throw new Error(`tensor ${t.name} must be 2-D for q8`);
      const rows = t.shape[0];
      const cols = t.shape[1];
      const stride = Math.ceil(cols / Q8_BLOCK) * Q8_BLOCK;
      const nblocks = stride / Q8_BLOCK;
      const qlen = rows * stride;
      const slen = rows * nblocks;
      if (t.length !== qlen + slen * 2) throw new Error(`tensor ${t.name} length mismatch`);
      const q = new Int8Array(buf, off, qlen);
      const s16 = new Uint16Array(buf, off + qlen, slen);
      const scales = new Float32Array(slen);
      for (let i = 0; i < slen; i++) scales[i] = f16ToF32(s16[i]);
      out.set(t.name, { kind: 'q8', shape: t.shape, rows, cols, stride, q, scales });
    } else {
      throw new Error(`tensor ${t.name} has unknown dtype ${(t as { dtype: string }).dtype}`);
    }
  }
  return out;
}

/** y = Wx for a Q8 matrix (dequantized on the fly, never materialized). */
export function q8MatVec(t: Q8Tensor, x: Float32Array, out: Float32Array): void {
  const { rows, cols, stride, q, scales } = t;
  const nblocks = stride / Q8_BLOCK;
  for (let r = 0; r < rows; r++) {
    const qb = r * stride;
    const sb = r * nblocks;
    let s = 0;
    for (let b = 0; b < nblocks; b++) {
      const start = b * Q8_BLOCK;
      if (start >= cols) break;
      const end = Math.min(Q8_BLOCK, cols - start);
      const base = qb + start;
      let acc = 0;
      for (let j = 0; j < end; j++) acc += x[start + j] * q[base + j];
      s += acc * scales[sb + b];
    }
    out[r] = s;
  }
}

/** y = Wx for a row-major fp32 matrix. */
export function f32MatVec(data: Float32Array, rows: number, cols: number, x: Float32Array, out: Float32Array): void {
  for (let r = 0; r < rows; r++) {
    const b = r * cols;
    let s = 0;
    for (let i = 0; i < cols; i++) s += data[b + i] * x[i];
    out[r] = s;
  }
}

/** Reference quantizer (mirrors the Python exporter; used by tests). */
export function quantizeQ8(data: Float32Array, rows: number, cols: number): Q8Tensor {
  const stride = Math.ceil(cols / Q8_BLOCK) * Q8_BLOCK;
  const nblocks = stride / Q8_BLOCK;
  const q = new Int8Array(rows * stride);
  const scales = new Float32Array(rows * nblocks);
  for (let r = 0; r < rows; r++) {
    for (let b = 0; b < nblocks; b++) {
      const start = b * Q8_BLOCK;
      let amax = 0;
      const end = Math.min(Q8_BLOCK, cols - start);
      for (let j = 0; j < end; j++) {
        const a = Math.abs(data[r * cols + start + j]);
        if (a > amax) amax = a;
      }
      const sc = amax === 0 ? 0 : amax / 127;
      scales[r * nblocks + b] = sc;
      for (let j = 0; j < Q8_BLOCK; j++) {
        const i = start + j;
        let v = 0;
        if (i < cols) {
          const w = data[r * cols + i];
          v = sc === 0 ? 0 : Math.max(-127, Math.min(127, Math.round(w / sc)));
        }
        q[r * stride + i] = v;
      }
    }
  }
  return { kind: 'q8', shape: [rows, cols], rows, cols, stride, q, scales };
}

/** Exact parameter count from tensor shapes (true shapes, padding excluded). */
export function countParams(tensors: Iterable<WeightTensor>): number {
  let n = 0;
  for (const t of tensors) n += t.shape.reduce((a, b) => a * b, 1);
  return n;
}
