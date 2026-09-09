// Paint.exe inference Web Worker: DDIM sampling off the UI thread.
import { ddimRun, ddimTimesteps, encodeText, parsePaintWeights, randn, type PaintConfig, type PaintWeights } from './engine/paint';
import { mulberry32 } from './engine/generate';

type InMsg =
  | { type: 'load'; baseUrl: string }
  | { type: 'sample'; id: number; prompt: string; steps: number; seed: number; guidance: number }
  | { type: 'stop' };

let w: PaintWeights | null = null;
let stopFlag = false;
let busy = false;

function post(msg: unknown): void {
  self.postMessage(msg);
}

self.onmessage = async (ev: MessageEvent<InMsg>) => {
  const msg = ev.data;
  try {
    if (msg.type === 'load') {
      if (busy) {
        post({ type: 'error', message: 'worker busy' });
        return;
      }
      busy = true;
      const res = await fetch(`${msg.baseUrl}models/paintexe/config.json`);
      if (!res.ok) throw new Error('missing paintexe config');
      const cfg = (await res.json()) as PaintConfig;
      const bin = await (await fetch(`${msg.baseUrl}models/paintexe/weights.f32.bin`)).arrayBuffer();
      w = parsePaintWeights(bin, cfg);
      post({ type: 'loaded', paramCount: w.paramCount });
      busy = false;
    } else if (msg.type === 'sample') {
      if (!w) {
        post({ type: 'error', id: msg.id, message: 'paint model not loaded' });
        return;
      }
      if (busy) {
        post({ type: 'error', id: msg.id, message: 'already sampling' });
        return;
      }
      busy = true;
      stopFlag = false;
      const weights = w;
      const cond = encodeText(weights, msg.prompt);
      const rng = mulberry32(msg.seed >>> 0);
      const xInit = randn(rng, 32 * 32 * 3);
      const t0 = performance.now();
      const { image, stopped } = await ddimRun(weights, xInit, cond, ddimTimesteps(msg.steps, weights.cfg.tSteps), msg.guidance, {
        onStep: (done, total, preview) => post({ type: 'step', id: msg.id, done, total, preview }),
        shouldStop: () => stopFlag,
        yieldNow: () => new Promise((r) => setTimeout(r, 0)),
      });
      post({ type: 'done', id: msg.id, image, ms: performance.now() - t0, stopped });
      busy = false;
    } else if (msg.type === 'stop') {
      stopFlag = true;
    }
  } catch (e) {
    busy = false;
    post({ type: 'error', message: e instanceof Error ? e.message : String(e) });
  }
};
