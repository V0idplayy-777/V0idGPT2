import { describe, expect, it } from 'vitest';
import { countParams, f16ToF32, f32MatVec, f32ToF16, q8MatVec, quantizeQ8 } from './quant';

/** Deterministic LCG for test data. */
function lcg(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (Math.imul(a, 1664525) + 1013904223) >>> 0;
    return a / 4294967296;
  };
}

describe('fp16 conversion', () => {
  it('decodes known values', () => {
    expect(f16ToF32(0x3c00)).toBe(1);
    expect(f16ToF32(0xbc00)).toBe(-1);
    expect(f16ToF32(0x4000)).toBe(2);
    expect(f16ToF32(0x0000)).toBe(0);
    expect(f16ToF32(0x7c00)).toBe(Infinity);
    expect(f16ToF32(0xfc00)).toBe(-Infinity);
    expect(f16ToF32(0x7e00)).toBeNaN();
    expect(f16ToF32(0x0001)).toBeCloseTo(2 ** -24, 12);
    expect(f16ToF32(0x3555)).toBeCloseTo(0.333251953125, 9);
  });
  it('roundtrips through the encoder', () => {
    for (const v of [0, 1, -1, 0.5, -0.001, 100, 3.14159, 1e-5]) {
      expect(f16ToF32(f32ToF16(v))).toBeCloseTo(v, 2);
    }
  });
});

describe('Q8 quantization', () => {
  it('q8MatVec matches f32MatVec (no padding)', () => {
    const rnd = lcg(7);
    const rows = 48;
    const cols = 96;
    const data = new Float32Array(rows * cols);
    for (let i = 0; i < data.length; i++) data[i] = (rnd() - 0.5) * 0.4;
    const x = new Float32Array(cols);
    for (let i = 0; i < cols; i++) x[i] = (rnd() - 0.5) * 2;
    const a = new Float32Array(rows);
    const b = new Float32Array(rows);
    f32MatVec(data, rows, cols, x, a);
    q8MatVec(quantizeQ8(data, rows, cols), x, b);
    let mx = 0;
    for (let i = 0; i < rows; i++) mx = Math.max(mx, Math.abs(a[i] - b[i]));
    expect(mx).toBeLessThan(0.05);
  });
  it('q8MatVec matches f32MatVec (with column padding)', () => {
    const rnd = lcg(11);
    const rows = 40;
    const cols = 100; // stride 128
    const data = new Float32Array(rows * cols);
    for (let i = 0; i < data.length; i++) data[i] = (rnd() - 0.5) * 0.4;
    const x = new Float32Array(cols);
    for (let i = 0; i < cols; i++) x[i] = (rnd() - 0.5) * 2;
    const a = new Float32Array(rows);
    const b = new Float32Array(rows);
    f32MatVec(data, rows, cols, x, a);
    const q = quantizeQ8(data, rows, cols);
    expect(q.stride).toBe(128);
    q8MatVec(q, x, b);
    let mx = 0;
    for (let i = 0; i < rows; i++) mx = Math.max(mx, Math.abs(a[i] - b[i]));
    expect(mx).toBeLessThan(0.05);
  });
  it('counts parameters from shapes (padding excluded)', () => {
    const q = quantizeQ8(new Float32Array(10 * 100), 10, 100);
    expect(countParams([q])).toBe(1000);
    expect(countParams([{ kind: 'f32', shape: [64, 16], data: new Float32Array(1024) }])).toBe(1024);
  });
});
