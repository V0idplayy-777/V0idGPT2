// Shared types for the V0idGPT2 browser inference engine.

export interface TensorMeta {
  name: string;
  shape: number[];
  dtype: 'f32' | 'q8';
  offset: number; // bytes, relative to start of weight data (after 16-byte file header)
  length: number; // bytes on disk (excluding alignment padding)
}

export interface ModelConfig {
  name: string;
  arch: string;
  vocab: number;
  dModel: number;
  nLayer: number;
  nHead: number;
  dFF: number;
  ctx: number;
  ropeTheta: number;
  rmsEps: number;
  tied: boolean;
  bos: number;
  eos: number;
  pad: number;
  format: string;
  paramCount: number;
  tensors: TensorMeta[];
  training?: {
    tokens: number;
    steps: number;
    stage: string;
    valLoss?: number;
  };
}

export interface F32Tensor {
  kind: 'f32';
  shape: number[];
  data: Float32Array;
}

export interface Q8Tensor {
  kind: 'q8';
  shape: number[]; // true [rows, cols]
  rows: number;
  cols: number;
  stride: number; // cols rounded up to a multiple of 32
  q: Int8Array; // rows*stride
  scales: Float32Array; // rows*(stride/32), converted from fp16 at load
}

export type WeightTensor = F32Tensor | Q8Tensor;

export interface LoadedModel {
  cfg: ModelConfig;
  weights: Map<string, WeightTensor>;
  /** exact parameter count computed from the loaded tensors */
  paramCount: number;
  /** keep the raw buffer alive (tensor views point into it) */
  buffer: ArrayBuffer;
}

export interface TokenizerData {
  vocab_size: number;
  merges: [number, number][];
  special: { bos: number; eos: number; pad: number };
}

export interface GenerationSettings {
  temperature: number;
  topK: number;
  topP: number;
  maxTokens: number;
  repeatPenalty: number;
  seed: number;
}

export const DEFAULT_SETTINGS: GenerationSettings = {
  temperature: 0.8,
  topK: 40,
  topP: 0.9,
  maxTokens: 128,
  repeatPenalty: 1.1,
  seed: 1234,
};
