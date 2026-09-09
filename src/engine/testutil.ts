// Shared test helpers (not part of the shipped engine).
import type { LoadedModel, ModelConfig, WeightTensor } from './types';

export function lcg(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (Math.imul(a, 1664525) + 1013904223) >>> 0;
    return a / 4294967296;
  };
}

/**
 * Tiny random fp32 model for structural tests.
 * NOTE: 2-D linear weights are stored output-major ([out, in]), exactly like
 * the real exported weights (the exporter transposes JAX [in, out] -> [out, in]).
 */
export function makeRandomModel(seed = 42): LoadedModel {
  const rnd = lcg(seed);
  const cfg: ModelConfig = {
    name: 'tiny',
    arch: 'test',
    vocab: 64,
    dModel: 16,
    nLayer: 2,
    nHead: 2,
    dFF: 32,
    ctx: 16,
    ropeTheta: 10000,
    rmsEps: 1e-6,
    tied: true,
    bos: 61,
    eos: 62,
    pad: 63,
    format: 'f32',
    paramCount: 0,
    tensors: [],
  };
  const rand = (r: number, c: number): Float32Array => {
    const a = new Float32Array(r * c);
    for (let i = 0; i < a.length; i++) a[i] = (rnd() - 0.5) * 0.2;
    return a;
  };
  const ones = (n: number): Float32Array => new Float32Array(n).fill(1);
  const f32 = (shape: number[], data: Float32Array): WeightTensor => ({ kind: 'f32', shape, data });
  const w = new Map<string, WeightTensor>();
  w.set('tok_emb', f32([64, 16], rand(64, 16)));
  w.set('final_norm', f32([16], ones(16)));
  for (let l = 0; l < 2; l++) {
    w.set(`L${l}.attn_norm`, f32([16], ones(16)));
    w.set(`L${l}.mlp_norm`, f32([16], ones(16)));
    w.set(`L${l}.wq`, f32([16, 16], rand(16, 16)));
    w.set(`L${l}.wk`, f32([16, 16], rand(16, 16)));
    w.set(`L${l}.wv`, f32([16, 16], rand(16, 16)));
    w.set(`L${l}.wo`, f32([16, 16], rand(16, 16)));
    w.set(`L${l}.wgate`, f32([32, 16], rand(32, 16)));
    w.set(`L${l}.wup`, f32([32, 16], rand(32, 16)));
    w.set(`L${l}.wdown`, f32([16, 32], rand(16, 32)));
  }
  return { cfg, weights: w, paramCount: 0, buffer: new ArrayBuffer(8) };
}

export function maxAbsDiff(a: Float32Array, b: Float32Array): number {
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}
