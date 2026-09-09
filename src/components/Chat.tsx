import { useEffect, useRef, useState } from 'react';
import type { BPETokenizer } from '../engine/tokenizer';
import type { LoadStatus } from '../hooks/useChatWorker';

export interface AssistantMessage {
  role: 'assistant';
  text: string;
  ids: number[];
  tops?: { id: number; prob: number }[][];
  stats?: { prefillMs: number; genMs: number; tokens: number; stopped: string };
  promptTokens?: number;
  truncated?: boolean;
}

export interface UserMessage {
  role: 'user';
  text: string;
}

export type Msg = UserMessage | AssistantMessage;

export interface StreamingState {
  ids: number[];
  tops: { id: number; prob: number }[][];
  promptTokens: number;
  truncated: boolean;
}

function DebugDetails({ msg, tok }: { msg: AssistantMessage; tok: BPETokenizer | null }) {
  if (!tok) return null;
  const tps = msg.stats && msg.stats.genMs > 0 ? (msg.stats.tokens / (msg.stats.genMs / 1000)).toFixed(1) : '?';
  return (
    <details className="debug">
      <summary>
        {msg.ids.length} tokens · {tps} tok/s · prefill {msg.stats?.prefillMs.toFixed(0)}ms · stop: {msg.stats?.stopped} ·
        prompt {msg.promptTokens} tokens{msg.truncated ? ' · ⚠ truncated' : ''}
      </summary>
      <div className="debug-body">
        <div className="debug-ids">
          {msg.ids.map((id, i) => (
            <span key={i} className="tok" title={`id ${id}`}>
              {JSON.stringify(tok.decodeToken(id))}
            </span>
          ))}
        </div>
        {msg.tops && (
          <table className="debug-table">
            <thead>
              <tr>
                <th>#</th>
                <th>picked</th>
                <th colSpan={5}>top candidates (id: prob)</th>
              </tr>
            </thead>
            <tbody>
              {msg.tops.map((top, i) => (
                <tr key={i}>
                  <td>{i}</td>
                  <td className="mono">{msg.ids[i]}</td>
                  {top.map((c) => (
                    <td key={c.id} className="mono">
                      {c.id}: {c.prob.toFixed(3)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </details>
  );
}

export default function Chat({
  messages,
  streaming,
  debug,
  tokenizer,
  generating,
  status,
  progress,
  error,
  modelTitle,
  onSend,
  onStop,
}: {
  messages: Msg[];
  streaming: StreamingState | null;
  debug: boolean;
  tokenizer: BPETokenizer | null;
  generating: boolean;
  status: LoadStatus;
  progress: { loaded: number; total: number } | null;
  error: string | null;
  modelTitle: string;
  onSend: (text: string) => void;
  onStop: () => void;
}) {
  const [draft, setDraft] = useState('');
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = listRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, streaming]);

  const send = () => {
    const t = draft.trim();
    if (!t || generating || status !== 'ready') return;
    setDraft('');
    onSend(t);
  };

  const streamingText = streaming && tokenizer ? tokenizer.decode(streaming.ids) : '';

  return (
    <div className="chat">
      <div className="messages" ref={listRef}>
        {messages.length === 0 && !streaming && (
          <div className="empty">
            <h2>{modelTitle} is listening.</h2>
            <p>
              Every token is computed live by the model&apos;s own weights, right here in your browser. No servers, no
              APIs, no canned answers.
            </p>
            <div className="examples">
              {['Hello! How are you?', 'Tell me a short story about a cat.', 'Why is the sky blue?'].map((ex) => (
                <button key={ex} className="example" disabled={generating || status !== 'ready'} onClick={() => onSend(ex)}>
                  {ex}
                </button>
              ))}
            </div>
          </div>
        )}
        {messages.map((m, i) =>
          m.role === 'user' ? (
            <div key={i} className="msg user">
              <div className="bubble">{m.text}</div>
            </div>
          ) : (
            <div key={i} className="msg assistant">
              <div className="bubble">{m.text || <span className="muted">…</span>}</div>
              {debug && m.ids.length > 0 && <DebugDetails msg={m} tok={tokenizer} />}
            </div>
          ),
        )}
        {streaming && (
          <div className="msg assistant">
            <div className="bubble streaming">
              {streamingText}
              <span className="caret">▍</span>
            </div>
            {debug && (
              <div className="muted small">
                streaming {streaming.ids.length} tokens · prompt {streaming.promptTokens} tokens
                {streaming.truncated ? ' · ⚠ prompt truncated to fit context' : ''}
              </div>
            )}
          </div>
        )}
        {status === 'loading' && (
          <div className="loading">
            <div>Loading {modelTitle} weights…</div>
            <div className="progress">
              <div
                className="progress-bar"
                style={{ width: progress && progress.total > 0 ? `${(100 * progress.loaded / progress.total).toFixed(1)}%` : '8%' }}
              />
            </div>
            <div className="muted small">
              {progress ? `${(progress.loaded / 1e6).toFixed(1)} / ${(progress.total / 1e6).toFixed(1)} MB` : 'connecting…'}
            </div>
          </div>
        )}
        {error && <div className="error">⚠ {error}</div>}
      </div>
      <div className="composer">
        <textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
              e.preventDefault();
              send();
            }
          }}
          placeholder={status === 'ready' ? `Message ${modelTitle}… (Enter to send)` : 'Loading model…'}
          rows={2}
          disabled={generating || status !== 'ready'}
        />
        {generating ? (
          <button className="btn stop" onClick={onStop}>
            Stop
          </button>
        ) : (
          <button className="btn send" onClick={send} disabled={status !== 'ready' || !draft.trim()}>
            Send
          </button>
        )}
      </div>
    </div>
  );
}
