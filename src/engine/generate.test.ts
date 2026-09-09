import { describe, expect, it } from 'vitest';
import { DEFAULT_SETTINGS } from './types';
import { generate, mulberry32, sampleToken } from './generate';
import { makeRandomModel } from './testutil';

describe('sampler', () => {
  it('greedy (temp=0) picks argmax', () => {
    const logits = new Float32Array([1, 5, 3, 2]);
    const r = sampleToken(logits, { ...DEFAULT_SETTINGS, temperature: 0 }, [], mulberry32(1));
    expect(r.id).toBe(1);
    expect(r.top[0].id).toBe(1);
  });
  it('is deterministic under a fixed seed', () => {
    const logits = new Float32Array([1, 2, 3, 4, 0.5]);
    const s = { ...DEFAULT_SETTINGS, temperature: 1, topK: 0, topP: 1, repeatPenalty: 1 };
    const a = sampleToken(logits, s, [], mulberry32(99));
    const b = sampleToken(logits, s, [], mulberry32(99));
    expect(a.id).toBe(b.id);
  });
  it('topK=1 equals greedy', () => {
    const logits = new Float32Array([0.1, 4, 3, 2, 1]);
    const s = { ...DEFAULT_SETTINGS, temperature: 1, topK: 1, topP: 1, repeatPenalty: 1 };
    const r = sampleToken(logits, s, [], mulberry32(3));
    expect(r.id).toBe(1);
  });
  it('repetition penalty demotes seen tokens', () => {
    const logits = new Float32Array([0, 0, 10, 9.9]);
    const base = { ...DEFAULT_SETTINGS, temperature: 0, topK: 0, topP: 1 };
    expect(sampleToken(logits, { ...base, repeatPenalty: 1 }, [2], mulberry32(1)).id).toBe(2);
    expect(sampleToken(logits, { ...base, repeatPenalty: 2 }, [2], mulberry32(1)).id).toBe(3);
  });
  it('topP truncation keeps the head of the distribution', () => {
    const logits = new Float32Array([10, 9, -100, -100]);
    const s = { ...DEFAULT_SETTINGS, temperature: 0, topK: 0, topP: 0.01, repeatPenalty: 1 };
    expect(sampleToken(logits, s, [], mulberry32(1)).id).toBe(0);
  });
});

describe('generate loop', () => {
  it('streams tokens with a KV cache and respects maxTokens', async () => {
    const m = makeRandomModel();
    const seen: number[] = [];
    const { ids, stats } = await generate(
      m,
      [1, 2, 3],
      { ...DEFAULT_SETTINGS, temperature: 0, maxTokens: 8 },
      { onToken: (id) => seen.push(id), shouldStop: () => false },
    );
    expect(ids.length).toBeLessThanOrEqual(8);
    expect(seen).toEqual(ids);
    expect(stats.tokens).toBe(ids.length);
  });
  it('is deterministic end-to-end under a fixed seed', async () => {
    const m = makeRandomModel();
    const run = () =>
      generate(m, [1, 2, 3], { ...DEFAULT_SETTINGS, maxTokens: 6, seed: 7 }, {
        onToken: () => {},
        shouldStop: () => false,
      });
    const a = await run();
    const b = await run();
    expect(a.ids).toEqual(b.ids);
  });
  it('supports user stop', async () => {
    const m = makeRandomModel();
    let n = 0;
    const { stats } = await generate(m, [1, 2, 3], { ...DEFAULT_SETTINGS, maxTokens: 8 }, {
      onToken: () => {
        n++;
      },
      shouldStop: () => n >= 3,
    });
    expect(stats.stopped).toBe('user');
  });
});
