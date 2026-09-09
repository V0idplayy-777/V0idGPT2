import { describe, expect, it } from 'vitest';
import { addInto, mulInto, rmsNorm, siluInPlace, softmaxInPlace } from './ops';

describe('ops', () => {
  it('rmsNorm normalizes', () => {
    const out = new Float32Array(4);
    rmsNorm(new Float32Array([1, 2, 3, 4]), new Float32Array([1, 1, 1, 1]), 0, out);
    const s = Math.sqrt((1 + 4 + 9 + 16) / 4);
    expect(Array.from(out)).toEqual([1 / s, 2 / s, 3 / s, 4 / s].map((v) => expect.closeTo(v, 6)));
  });
  it('softmax sums to 1 and is shift-stable', () => {
    const x = new Float32Array([1000, 1001, 999]);
    softmaxInPlace(x);
    const sum = x[0] + x[1] + x[2];
    expect(sum).toBeCloseTo(1, 6);
    expect(x[1]).toBeGreaterThan(x[0]);
    expect(x[0]).toBeGreaterThan(x[2]);
  });
  it('silu matches known values', () => {
    const x = new Float32Array([0, 1, -1]);
    siluInPlace(x);
    expect(x[0]).toBe(0);
    expect(x[1]).toBeCloseTo(0.7310585786, 6);
    expect(x[2]).toBeCloseTo(-0.2689414214, 6);
  });
  it('addInto/mulInto work', () => {
    const a = new Float32Array([1, 2]);
    addInto(a, new Float32Array([3, 4]));
    expect(Array.from(a)).toEqual([4, 6]);
    const o = new Float32Array(2);
    mulInto(a, new Float32Array([2, 0.5]), o);
    expect(Array.from(o)).toEqual([8, 3]);
  });
});
