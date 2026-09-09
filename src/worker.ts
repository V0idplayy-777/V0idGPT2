// Chat inference Web Worker: loads models and streams tokens off the UI thread.
import type { GenerationSettings } from './engine/types';
import { loadModel, loadTokenizer } from './engine/model';
import type { LoadedModel } from './engine/types';
import type { BPETokenizer } from './engine/tokenizer';
import { generate } from './engine/generate';

interface LoadMsg {
  type: 'load';
  model: string;
  baseUrl: string;
}
interface GenMsg {
  type: 'generate';
  id: number;
  prompt: string;
  settings: GenerationSettings;
  debug: boolean;
}
interface StopMsg {
  type: 'stop';
}
type InMsg = LoadMsg | GenMsg | StopMsg;

let model: LoadedModel | null = null;
let tokenizer: BPETokenizer | null = null;
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
      stopFlag = false;
      const t0 = performance.now();
      tokenizer = await loadTokenizer(msg.baseUrl);
      model = await loadModel(msg.model, msg.baseUrl, (loaded, total) =>
        post({ type: 'progress', loaded, total }),
      );
      const ms = performance.now() - t0;
      post({
        type: 'loaded',
        model: msg.model,
        paramCount: model.paramCount,
        cfg: {
          dModel: model.cfg.dModel,
          nLayer: model.cfg.nLayer,
          nHead: model.cfg.nHead,
          dFF: model.cfg.dFF,
          ctx: model.cfg.ctx,
          vocab: model.cfg.vocab,
        },
        ms,
      });
      busy = false;
    } else if (msg.type === 'generate') {
      if (!model || !tokenizer) {
        post({ type: 'error', id: msg.id, message: 'no model loaded' });
        return;
      }
      if (busy) {
        post({ type: 'error', id: msg.id, message: 'already generating' });
        return;
      }
      busy = true;
      stopFlag = false;
      const tok = tokenizer;
      const m = model;
      let promptIds = tok.chatPrompt(msg.prompt);
      if (promptIds.length > m.cfg.ctx - 8) {
        // keep the tail (most recent context) — honest truncation, reported to UI
        promptIds = promptIds.slice(promptIds.length - (m.cfg.ctx - 8));
      }
      post({ type: 'start', id: msg.id, promptTokens: promptIds.length });
      const { ids, stats } = await generate(m, promptIds, msg.settings, {
        onToken: (id, top) => {
          post({
            type: 'token',
            id: msg.id,
            tokenId: id,
            text: id === m.cfg.eos ? '' : tok.decodeToken(id),
            top: msg.debug ? top : undefined,
          });
        },
        shouldStop: () => stopFlag,
        yieldNow: () => new Promise((r) => setTimeout(r, 0)),
      });
      post({ type: 'done', id: msg.id, ids, stats });
      busy = false;
    } else if (msg.type === 'stop') {
      stopFlag = true;
    }
  } catch (e) {
    busy = false;
    post({ type: 'error', message: e instanceof Error ? e.message : String(e) });
  }
};
