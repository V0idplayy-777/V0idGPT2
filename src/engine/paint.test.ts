import { describe, expect, it } from 'vitest';
import fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { applyUnet, ddimRun, ddimSample, encodeText, parsePaintWeights, type PaintConfig } from './paint';
import { maxAbsDiff } from './testutil';

const dir = fileURLToPath(new URL('../../public/models/paintexe', import.meta.url));

function load() {
  const cfg = JSON.parse(fs.readFileSync(path.join(dir, 'config.json'), 'utf-8')) as PaintConfig;
  const raw = fs.readFileSync(path.join(dir, 'weights.f32.bin'));
  const buf = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength) as ArrayBuffer;
  return parsePaintWeights(buf, cfg);
}

function parity() {
  return JSON.parse(fs.readFileSync(path.join(dir, 'parity.json'), 'utf-8')) as {
    prompt: string;
    cond: number[];
    x_t: number[];
    t_single: number;
    eps_pred: number[];
    x_init: number[];
    timesteps: number[];
    ddim8: number[];
  };
}

describe('paint', () => {
  it('loads weights and counts parameters', () => {
    const w = load();
    expect(w.paramCount).toBe(w.cfg.paramCount);
    expect(w.paramCount).toBeGreaterThan(100000);
  });
  it('matches JAX text conditioning', () => {
    const w = load();
    const p = parity();
    const cond = encodeText(w, p.prompt);
    expect(maxAbsDiff(cond, new Float32Array(p.cond))).toBeLessThan(1e-4);
  });
  it('matches one JAX UNet forward', () => {
    const w = load();
    const p = parity();
    const eps = applyUnet(w, new Float32Array(p.x_t), p.t_single, new Float32Array(p.cond));
    expect(maxAbsDiff(eps, new Float32Array(p.eps_pred))).toBeLessThan(1e-2);
  });
  it('matches an 8-step JAX DDIM run', async () => {
    const w = load();
    const p = parity();
    const { image } = await ddimRun(w, new Float32Array(p.x_init), new Float32Array(p.cond), p.timesteps, 1.0);
    expect(maxAbsDiff(image, new Float32Array(p.ddim8))).toBeLessThan(2e-2);
  }, 120000);
  it('is deterministic under a fixed seed', async () => {
    const w = load();
    const a = await ddimSample(w, 'a photo of a dog', { steps: 3, seed: 11, guidance: 1 });
    const b = await ddimSample(w, 'a photo of a dog', { steps: 3, seed: 11, guidance: 1 });
    expect(maxAbsDiff(a.image, b.image)).toBe(0);
    for (let i = 0; i < a.image.length; i++) {
      expect(a.image[i]).toBeGreaterThanOrEqual(0);
      expect(a.image[i]).toBeLessThanOrEqual(1);
    }
  }, 120000);
});
