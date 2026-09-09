// Small fp32 vector ops for the inference engine.

/** out = x / sqrt(mean(x^2) + eps) * w */
export function rmsNorm(x: Float32Array, w: Float32Array, eps: number, out: Float32Array): void {
  const n = x.length;
  let ss = 0;
  for (let i = 0; i < n; i++) ss += x[i] * x[i];
  const s = 1 / Math.sqrt(ss / n + eps);
  for (let i = 0; i < n; i++) out[i] = x[i] * s * w[i];
}

/** In-place softmax (max-subtracted, fp64 accumulation). */
export function softmaxInPlace(x: Float32Array): void {
  const n = x.length;
  let m = -Infinity;
  for (let i = 0; i < n; i++) if (x[i] > m) m = x[i];
  let s = 0;
  for (let i = 0; i < n; i++) {
    const e = Math.exp(x[i] - m);
    x[i] = e;
    s += e;
  }
  const inv = 1 / s;
  for (let i = 0; i < n; i++) x[i] *= inv;
}

/** In-place SiLU: x * sigmoid(x). */
export function siluInPlace(x: Float32Array): void {
  for (let i = 0; i < x.length; i++) {
    const v = x[i];
    x[i] = v / (1 + Math.exp(-v));
  }
}

/** a += b (elementwise). */
export function addInto(a: Float32Array, b: Float32Array): void {
  for (let i = 0; i < a.length; i++) a[i] += b[i];
}

/** out = a * b (elementwise). */
export function mulInto(a: Float32Array, b: Float32Array, out: Float32Array): void {
  for (let i = 0; i < a.length; i++) out[i] = a[i] * b[i];
}
