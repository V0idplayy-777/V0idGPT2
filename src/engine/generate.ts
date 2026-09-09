// Autoregressive sampling: temperature / top-k / top-p / repetition penalty,
// seeded RNG, streaming generation with KV cache.
import type { GenerationSettings, LoadedModel } from './types';
import type { BPETokenizer } from './tokenizer';
import { KVCache, forwardStep, prefillCache } from './transformer';
import { softmaxInPlace } from './ops';

/** Deterministic seeded PRNG (mulberry32). */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface SampleResult {
  id: number;
  /** top-5 candidates (id, prob) for the debug panel */
  top: { id: number; prob: number }[];
}

/** Sample one token id from logits (mutates a copy, never the input). */
export function sampleToken(
  logits: Float32Array,
  settings: GenerationSettings,
  history: ArrayLike<number>,
  rng: () => number,
): SampleResult {
  const V = logits.length;
  const work = new Float32Array(logits);
  const { temperature, topK, topP, repeatPenalty } = settings;

  if (repeatPenalty !== 1 && history.length > 0) {
    const seen = new Set<number>();
    for (let i = 0; i < history.length; i++) seen.add(history[i]);
    for (const id of seen) {
      if (id < 0 || id >= V) continue;
      if (work[id] > 0) work[id] /= repeatPenalty;
      else work[id] *= repeatPenalty;
    }
  }

  // Rank all candidates once (V=4096: full sort is cheap).
  const order = new Array<number>(V);
  for (let i = 0; i < V; i++) order[i] = i;
  order.sort((a, b) => work[b] - work[a]);

  let k = V;
  if (topK > 0) k = Math.min(k, topK);
  if (topP < 1) {
    // nucleus cutoff on temperature-scaled probs
    const t = temperature > 0 ? temperature : 1;
    let m = -Infinity;
    for (let i = 0; i < k; i++) if (work[order[i]] / t > m) m = work[order[i]] / t;
    let cum = 0;
    let total = 0;
    const probs = new Float64Array(k);
    for (let i = 0; i < k; i++) {
      const p = Math.exp(work[order[i]] / t - m);
      probs[i] = p;
      total += p;
    }
    let keep = k;
    for (let i = 0; i < k; i++) {
      cum += probs[i] / total;
      if (cum >= topP) {
        keep = i + 1;
        break;
      }
    }
    k = Math.max(1, keep);
  }

  let id: number;
  if (temperature <= 0) {
    id = order[0];
  } else {
    let m = -Infinity;
    for (let i = 0; i < k; i++) {
      const v = work[order[i]] / temperature;
      if (v > m) m = v;
    }
    let total = 0;
    const probs = new Float64Array(k);
    for (let i = 0; i < k; i++) {
      const p = Math.exp(work[order[i]] / temperature - m);
      probs[i] = p;
      total += p;
    }
    let r = rng() * total;
    id = order[k - 1];
    for (let i = 0; i < k; i++) {
      r -= probs[i];
      if (r <= 0) {
        id = order[i];
        break;
      }
    }
  }

  // Debug top-5 (full-vocab softmax at temperature 1 for interpretability).
  const dbg = new Float32Array(work);
  softmaxInPlace(dbg);
  const top = order.slice(0, 5).map((cid) => ({ id: cid, prob: dbg[cid] }));
  return { id, top };
}

export interface GenerateCallbacks {
  onToken: (id: number, top: { id: number; prob: number }[]) => void;
  shouldStop: () => boolean;
  /** yield to the event loop (lets a worker receive stop messages) */
  yieldNow?: () => Promise<void>;
}

export interface GenerateStats {
  prefillMs: number;
  genMs: number;
  tokens: number;
  stopped: 'eos' | 'maxlen' | 'ctx' | 'user';
}

/** Stream tokens until eos / maxTokens / context end / user stop. */
export async function generate(
  model: LoadedModel,
  promptIds: number[],
  settings: GenerationSettings,
  cb: GenerateCallbacks,
): Promise<{ ids: number[]; stats: GenerateStats }> {
  const cache = new KVCache(model.cfg);
  const rng = mulberry32(settings.seed >>> 0);
  const ids: number[] = [];
  const t0 = performance.now();
  let logits = prefillCache(model, promptIds, cache);
  const prefillMs = performance.now() - t0;
  const history = [...promptIds];
  const t1 = performance.now();
  let stopped: GenerateStats['stopped'] = 'maxlen';
  for (let n = 0; n < settings.maxTokens; n++) {
    if (cb.shouldStop()) {
      stopped = 'user';
      break;
    }
    if (cache.len >= model.cfg.ctx) {
      stopped = 'ctx';
      break;
    }
    const { id, top } = sampleToken(logits, settings, history, rng);
    ids.push(id);
    history.push(id);
    cb.onToken(id, top);
    if (id === model.cfg.eos) {
      stopped = 'eos';
      break;
    }
    if (cb.yieldNow) await cb.yieldNow();
    logits = forwardStep(model, id, cache.len, cache);
  }
  const genMs = performance.now() - t1;
  return { ids, stats: { prefillMs, genMs, tokens: ids.length, stopped } };
}

export type { BPETokenizer };
