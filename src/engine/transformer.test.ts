import { describe, expect, it } from 'vitest';
import { KVCache, forwardFull, forwardStep, prefillCache } from './transformer';
import { makeRandomModel, maxAbsDiff } from './testutil';

describe('transformer', () => {
  it('is deterministic', () => {
    const m = makeRandomModel();
    const a = forwardFull(m, [1, 2, 3, 4]);
    const b = forwardFull(m, [1, 2, 3, 4]);
    expect(maxAbsDiff(a.logits, b.logits)).toBe(0);
  });
  it('produces finite logits', () => {
    const m = makeRandomModel();
    const a = forwardFull(m, [1, 2, 3, 4]);
    for (let i = 0; i < a.logits.length; i++) expect(Number.isFinite(a.logits[i])).toBe(true);
  });
  it('has working causal masking (future tokens do not affect the past)', () => {
    const m = makeRandomModel();
    const a = forwardFull(m, [1, 2, 3, 4, 5]);
    const b = forwardFull(m, [1, 2, 3, 4, 50]);
    for (let t = 0; t < 4; t++) {
      const ra = a.logits.subarray(t * 64, t * 64 + 64);
      const rb = b.logits.subarray(t * 64, t * 64 + 64);
      expect(maxAbsDiff(ra, rb)).toBeLessThan(1e-6);
    }
    const la = a.logits.subarray(4 * 64, 5 * 64);
    const lb = b.logits.subarray(4 * 64, 5 * 64);
    expect(maxAbsDiff(la, lb)).toBeGreaterThan(1e-6);
  });
  it('attention actually matters (different context -> different output)', () => {
    const m = makeRandomModel();
    const a = forwardFull(m, [1, 2, 3]);
    const b = forwardFull(m, [9, 2, 3]);
    expect(maxAbsDiff(a.logits, b.logits)).toBeGreaterThan(1e-6);
  });
  it('KV-cache stepping matches full forward', () => {
    const m = makeRandomModel();
    const ids = [5, 11, 23, 7, 42];
    const full = forwardFull(m, ids);
    const cache = new KVCache(m.cfg);
    const last = prefillCache(m, ids, cache);
    expect(cache.len).toBe(ids.length);
    const row = full.logits.subarray(4 * 64, 5 * 64);
    expect(maxAbsDiff(last, new Float32Array(row))).toBeLessThan(1e-4);
    const next = forwardStep(m, 3, cache.len, cache);
    const full2 = forwardFull(m, [...ids, 3]);
    const row2 = full2.logits.subarray(5 * 64, 6 * 64);
    expect(maxAbsDiff(next, new Float32Array(row2))).toBeLessThan(1e-4);
  });
});
