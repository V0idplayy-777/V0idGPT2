// Model + tokenizer loading (fetch from the static site, no backend).
import type { LoadedModel, ModelConfig, TokenizerData } from './types';
import { countParams, parseWeights } from './quant';
import { BPETokenizer } from './tokenizer';

export const MODEL_NAMES = ['potato', 'toaster', 'microwave', 'blender', 'nuclearfridge'] as const;
export type ModelName = (typeof MODEL_NAMES)[number];

export const MODEL_DISPLAY: Record<ModelName, { title: string; blurb: string }> = {
  potato: { title: 'Potato', blurb: 'The tiniest brain. Fast, simple, occasionally wise.' },
  toaster: { title: 'Toaster', blurb: 'Warming up. Short sentences, breakfast topics.' },
  microwave: { title: 'Microwave', blurb: 'Heats up leftovers and conversations.' },
  blender: { title: 'Blender', blurb: 'Blends words into mostly-smooth sentences.' },
  nuclearfridge: { title: 'NuclearFridge', blurb: 'The flagship. Cold storage, hot takes.' },
};

async function fetchArrayBuffer(url: string, onProgress?: (loaded: number, total: number) => void): Promise<ArrayBuffer> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch failed: ${url} (${res.status})`);
  const total = Number(res.headers.get('content-length') ?? 0);
  if (!res.body || !onProgress) {
    const buf = await res.arrayBuffer();
    onProgress?.(buf.byteLength, total || buf.byteLength);
    return buf;
  }
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress(loaded, total);
  }
  const buf = new Uint8Array(loaded);
  let o = 0;
  for (const c of chunks) {
    buf.set(c, o);
    o += c.length;
  }
  return buf.buffer;
}

export async function loadModel(
  name: string,
  baseUrl: string,
  onProgress?: (loaded: number, total: number) => void,
): Promise<LoadedModel> {
  const cfgRes = await fetch(`${baseUrl}models/${name}/config.json`);
  if (!cfgRes.ok) throw new Error(`missing config for model ${name}`);
  const cfg = (await cfgRes.json()) as ModelConfig;
  const buf = await fetchArrayBuffer(`${baseUrl}models/${name}/weights.q8.bin`, onProgress);
  const weights = parseWeights(buf, cfg);
  const paramCount = countParams(weights.values());
  if (paramCount !== cfg.paramCount) {
    throw new Error(`param mismatch for ${name}: counted ${paramCount}, config says ${cfg.paramCount}`);
  }
  return { cfg, weights, paramCount, buffer: buf };
}

let cachedTokenizer: BPETokenizer | null = null;

export async function loadTokenizer(baseUrl: string): Promise<BPETokenizer> {
  if (cachedTokenizer) return cachedTokenizer;
  const res = await fetch(`${baseUrl}models/tokenizer.json`);
  if (!res.ok) throw new Error('missing tokenizer.json');
  const data = (await res.json()) as TokenizerData;
  cachedTokenizer = new BPETokenizer(data);
  return cachedTokenizer;
}

/** Synchronous loader for Node tests (reads from the repo's public dir). */
export async function loadModelFromDir(name: string, dir: string): Promise<LoadedModel> {
  const fs = await import('node:fs');
  const path = await import('node:path');
  const cfg = JSON.parse(fs.readFileSync(path.join(dir, name, 'config.json'), 'utf-8')) as ModelConfig;
  const raw = fs.readFileSync(path.join(dir, name, 'weights.q8.bin'));
  const buf = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength) as ArrayBuffer;
  const weights = parseWeights(buf, cfg);
  const paramCount = countParams(weights.values());
  return { cfg, weights, paramCount, buffer: buf };
}
