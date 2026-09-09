import { useEffect, useRef, useState } from 'react';
import ModelTabs, { type Selection } from './components/ModelTabs';
import Chat, { type Msg, type StreamingState } from './components/Chat';
import SettingsPanel from './components/SettingsPanel';
import InfoPanel from './components/InfoPanel';
import PaintPanel from './components/PaintPanel';
import { useChatWorker } from './hooks/useChatWorker';
import { MODEL_DISPLAY, loadTokenizer, type ModelName } from './engine/model';
import type { BPETokenizer } from './engine/tokenizer';
import { DEFAULT_SETTINGS, type GenerationSettings } from './engine/types';

const EMPTY_STREAM: StreamingState = { ids: [], tops: [], promptTokens: 0, truncated: false };

export default function App() {
  const [selection, setSelection] = useState<Selection>('potato');
  const [convos, setConvos] = useState<Record<string, Msg[]>>({});
  const [streaming, setStreaming] = useState<StreamingState | null>(null);
  const [settings, setSettings] = useState<GenerationSettings>(DEFAULT_SETTINGS);
  const [debug, setDebug] = useState(false);
  const [tokenizer, setTokenizer] = useState<BPETokenizer | null>(null);
  const worker = useChatWorker();
  // Ref mirror of the streaming state so onDone can finalize without
  // side effects inside a state updater (StrictMode-safe).
  const streamRef = useRef<StreamingState>({ ...EMPTY_STREAM });

  useEffect(() => {
    loadTokenizer(import.meta.env.BASE_URL)
      .then(setTokenizer)
      .catch((e) => console.error('tokenizer load failed', e));
  }, []);

  useEffect(() => {
    if (selection !== 'paintexe') worker.load(selection);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selection]);

  // A worker error mid-generation leaves no 'done' event: clear the stuck stream.
  useEffect(() => {
    if (worker.error) {
      streamRef.current = { ...EMPTY_STREAM };
      setStreaming(null);
    }
  }, [worker.error]);

  const messages = convos[selection] ?? [];

  const send = (text: string) => {
    if (!tokenizer || worker.generating || worker.status !== 'ready') return;
    const model = selection;
    const updated: Msg[] = [...(convos[model] ?? []), { role: 'user', text }];
    setConvos((c) => ({ ...c, [model]: updated }));
    streamRef.current = { ...EMPTY_STREAM };
    setStreaming({ ...EMPTY_STREAM });
    const history = updated.map((m) => ({ role: m.role, text: m.text }));
    worker.generate(history, settings, debug, {
      onStart: (promptTokens, truncated) => {
        streamRef.current = { ...streamRef.current, promptTokens, truncated };
        setStreaming((s) => (s ? { ...s, promptTokens, truncated } : s));
      },
      onToken: (t) => {
        streamRef.current = {
          ...streamRef.current,
          ids: [...streamRef.current.ids, t.tokenId],
          tops: [...streamRef.current.tops, t.top ?? []],
        };
        setStreaming((s) =>
          s ? { ...s, ids: [...s.ids, t.tokenId], tops: [...s.tops, t.top ?? []] } : s,
        );
      },
      onDone: (d) => {
        const s = streamRef.current;
        const reply: Msg = {
          role: 'assistant',
          text: tokenizer.decode(d.ids),
          ids: d.ids,
          tops: debug ? s.tops : undefined,
          stats: d.stats,
          promptTokens: s.promptTokens,
          truncated: s.truncated,
        };
        streamRef.current = { ...EMPTY_STREAM };
        setStreaming(null);
        setConvos((c) => ({ ...c, [model]: [...(c[model] ?? []), reply] }));
      },
    });
  };

  const isPaint = selection === 'paintexe';
  const title = isPaint ? 'Paint.exe' : MODEL_DISPLAY[selection as ModelName].title;
  const tabsDisabled = worker.generating || worker.status === 'loading';

  return (
    <div className="app">
      <header>
        <div className="brand">
          <h1>V0idGPT2</h1>
          <p className="tagline">Six real neural networks. Zero servers. 100% local inference.</p>
        </div>
        <a className="gh" href="https://github.com/V0idplayy-777/V0idGPT2" target="_blank" rel="noreferrer">
          GitHub
        </a>
      </header>
      <ModelTabs selected={selection} onSelect={setSelection} disabled={tabsDisabled} />
      {isPaint ? (
        <PaintPanel />
      ) : (
        <main className="layout">
          <Chat
            messages={messages}
            streaming={streaming}
            debug={debug}
            tokenizer={tokenizer}
            generating={worker.generating}
            status={worker.status}
            progress={worker.progress}
            error={worker.error}
            modelTitle={title}
            onSend={send}
            onStop={worker.stop}
          />
          <aside className="side">
            <InfoPanel model={selection} counted={worker.info?.paramCount ?? null} />
            <SettingsPanel
              settings={settings}
              onChange={setSettings}
              debug={debug}
              onDebug={setDebug}
              disabled={worker.generating}
            />
          </aside>
        </main>
      )}
      <footer className="muted small">
        Every response is generated token-by-token from the selected model&apos;s own learned weights. There are no canned
        responses, no templates, and no API calls — verify with debug mode.
      </footer>
    </div>
  );
}
