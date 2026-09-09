import { describe, expect, it } from 'vitest';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { MODEL_NAMES, loadModelFromDir } from './model';
import { KVCache, forwardFull, prefillCache } from './transformer';
import { maxAbsDiff } from './testutil';

const dir = fileURLToPath(new URL('../../public/models', import.meta.url));

const EXPECTED_PARAMS: Record<string, number> = {
  potato: 1498496,
  toaster: 5103168,
  microwave: 11968416,
  blender: 28420992,
  nuclearfridge: 52840960,
};

function argmax(a: Float32Array): number {
  let bi = 0;
  for (let i = 1; i < a.length; i++) if (a[i] > a[bi]) bi = i;
  return bi;
}

describe.each(MODEL_NAMES)('model %s', (name) => {
  it('loads weights and counts exact parameters', async () => {
    const m = await loadModelFromDir(name, dir);
    expect(m.paramCount).toBe(EXPECTED_PARAMS[name]);
    expect(m.cfg.paramCount).toBe(EXPECTED_PARAMS[name]);
    expect(m.cfg.vocab).toBe(4096);
  });

  it('matches JAX reference logits (Q8 parity)', async () => {
    const m = await loadModelFromDir(name, dir);
    const parity = JSON.parse(fs.readFileSync(path.join(dir, name, 'parity.json'), 'utf-8')) as {
      prompt_ids: number[];
      logits_fp32: number[];
      logits_q8: number[];
    };
    const T = parity.prompt_ids.length;
    const full = forwardFull(m, parity.prompt_ids);
    const row = full.logits.subarray((T - 1) * m.cfg.vocab, T * m.cfg.vocab);
    const ref = new Float32Array(parity.logits_q8);
    expect(maxAbsDiff(new Float32Array(row), ref)).toBeLessThan(0.05);
    // quantization preserves the top prediction
    expect(argmax(new Float32Array(row))).toBe(argmax(new Float32Array(parity.logits_fp32)));
    // cached prefill agrees with the full forward pass on real weights
    const cache = new KVCache(m.cfg);
    const last = prefillCache(m, parity.prompt_ids, cache);
    expect(maxAbsDiff(last, new Float32Array(row))).toBeLessThan(1e-3);
  }, 180000);
});
