import { useCallback, useEffect, useRef, useState } from 'react';
import type { GenerationSettings } from '../engine/types';

export interface LoadedInfo {
  name: string;
  paramCount: number;
  arch: { dModel: number; nLayer: number; nHead: number; dFF: number; ctx: number; vocab: number };
  ms: number;
}

export type LoadStatus = 'idle' | 'loading' | 'ready' | 'error';

export interface TokenEvent {
  tokenId: number;
  top?: { id: number; prob: number }[];
}

export interface DoneEvent {
  ids: number[];
  stats: { prefillMs: number; genMs: number; tokens: number; stopped: string };
}

export interface GenCallbacks {
  onStart: (promptTokens: number) => void;
  onToken: (t: TokenEvent) => void;
  onDone: (d: DoneEvent) => void;
}

export function useChatWorker() {
  const workerRef = useRef<Worker | null>(null);
  const genRef = useRef<GenCallbacks | null>(null);
  const idRef = useRef(0);
  const [status, setStatus] = useState<LoadStatus>('idle');
  const [progress, setProgress] = useState<{ loaded: number; total: number } | null>(null);
  const [info, setInfo] = useState<LoadedInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [generating, setGenerating] = useState(false);

  useEffect(() => {
    const w = new Worker(new URL('../worker.ts', import.meta.url), { type: 'module' });
    workerRef.current = w;
    w.onmessage = (ev: MessageEvent) => {
      const m = ev.data as { type: string; [k: string]: unknown };
      switch (m.type) {
        case 'progress':
          setProgress({ loaded: m.loaded as number, total: m.total as number });
          break;
        case 'loaded':
          setInfo({
            name: m.model as string,
            paramCount: m.paramCount as number,
            arch: m.cfg as LoadedInfo['arch'],
            ms: m.ms as number,
          });
          setStatus('ready');
          setProgress(null);
          break;
        case 'start':
          genRef.current?.onStart(m.promptTokens as number);
          break;
        case 'token':
          genRef.current?.onToken({ tokenId: m.tokenId as number, top: m.top as TokenEvent['top'] });
          break;
        case 'done':
          setGenerating(false);
          genRef.current?.onDone({ ids: m.ids as number[], stats: m.stats as DoneEvent['stats'] });
          genRef.current = null;
          break;
        case 'error':
          setError(String(m.message ?? 'unknown worker error'));
          setGenerating(false);
          genRef.current = null;
          setStatus((s) => (s === 'loading' ? 'error' : s));
          break;
      }
    };
    return () => {
      w.terminate();
      workerRef.current = null;
    };
  }, []);

  const load = useCallback((model: string) => {
    setStatus('loading');
    setError(null);
    setInfo(null);
    setProgress(null);
    workerRef.current?.postMessage({ type: 'load', model, baseUrl: import.meta.env.BASE_URL });
  }, []);

  const generate = useCallback((prompt: string, settings: GenerationSettings, debug: boolean, cbs: GenCallbacks) => {
    const id = ++idRef.current;
    genRef.current = cbs;
    setGenerating(true);
    setError(null);
    workerRef.current?.postMessage({ type: 'generate', id, prompt, settings, debug });
  }, []);

  const stop = useCallback(() => {
    workerRef.current?.postMessage({ type: 'stop' });
  }, []);

  return { status, progress, info, error, generating, load, generate, stop };
}
