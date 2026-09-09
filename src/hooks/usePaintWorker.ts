import { useCallback, useEffect, useRef, useState } from 'react';

export interface SampleCallbacks {
  onStep: (done: number, total: number, preview: Float32Array) => void;
  onDone: (image: Float32Array, ms: number, stopped: boolean) => void;
}

export function usePaintWorker() {
  const workerRef = useRef<Worker | null>(null);
  const cbRef = useRef<SampleCallbacks | null>(null);
  const idRef = useRef(0);
  const [loaded, setLoaded] = useState(false);
  const [paramCount, setParamCount] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [sampling, setSampling] = useState(false);
  const [progress, setProgress] = useState<{ done: number; total: number } | null>(null);

  useEffect(() => {
    const w = new Worker(new URL('../workerpaint.ts', import.meta.url), { type: 'module' });
    workerRef.current = w;
    w.onmessage = (ev: MessageEvent) => {
      const m = ev.data as { type: string; [k: string]: unknown };
      switch (m.type) {
        case 'loaded':
          setParamCount(m.paramCount as number);
          setLoaded(true);
          break;
        case 'step':
          setProgress({ done: m.done as number, total: m.total as number });
          cbRef.current?.onStep(m.done as number, m.total as number, m.preview as Float32Array);
          break;
        case 'done':
          setSampling(false);
          setProgress(null);
          cbRef.current?.onDone(m.image as Float32Array, m.ms as number, m.stopped as boolean);
          cbRef.current = null;
          break;
        case 'error':
          setError(String(m.message ?? 'unknown paint error'));
          setSampling(false);
          cbRef.current = null;
          break;
      }
    };
    w.postMessage({ type: 'load', baseUrl: import.meta.env.BASE_URL });
    return () => {
      w.terminate();
      workerRef.current = null;
    };
  }, []);

  const sample = useCallback((prompt: string, steps: number, seed: number, guidance: number, cbs: SampleCallbacks) => {
    const id = ++idRef.current;
    cbRef.current = cbs;
    setSampling(true);
    setError(null);
    setProgress({ done: 0, total: steps });
    workerRef.current?.postMessage({ type: 'sample', id, prompt, steps, seed, guidance });
  }, []);

  const stop = useCallback(() => {
    workerRef.current?.postMessage({ type: 'stop' });
  }, []);

  return { loaded, paramCount, error, sampling, progress, sample, stop };
}
