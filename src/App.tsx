import { useEffect, useState } from 'react';
import ModelTabs, { type Selection } from './components/ModelTabs';
import Chat, { type Msg, type StreamingState } from './components/Chat';
import SettingsPanel from './components/SettingsPanel';
import InfoPanel from './components/InfoPanel';
import PaintPanel from './components/PaintPanel';
import { useChatWorker } from './hooks/useChatWorker';
import { MODEL_DISPLAY, loadTokenizer, type ModelName } from './engine/model';
import type { BPETokenizer } from './engine/tokenizer';
import { DEFAULT_SETTINGS, type GenerationSettings } from './engine/types';

export default function App() {
  const [selection, setSelection] = useState<Selection>('potato');
  const [convos, setConvos] = useState<Record<string, Msg[]>>({});
  const [streaming, setStreaming] = useState<StreamingState | null>(null);
  const [settings, setSettings] = useState<GenerationSettings>(DEFAULT_SETTINGS);
  const [debug, setDebug] = useState(false);
  const [tokenizer, setTokenizer] = useState<BPETokenizer | null>(null);
  const worker = useChatWorker();

  useEffect(() => {
    loadTokenizer(import.meta.env.BASE_URL)
      .then(setTokenizer)
      .catch((e) => console.error('tokenizer load failed', e));
  }, []);

  useEffect(() => {
    if (selection !== 'paintexe') worker.load(selection);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selection]);

  const messages = convos[selection] ?? [];

  const send = (text: string) => {
    if (!tokenizer || worker.generating || worker.status !== 'ready') return;
    const model = selection;
    setConvos((c) => ({ ...c, [model]: [...(c[model] ?? []), { role: 'user', text }] }));
    setStreaming({ ids: [], tops: [], promptTokens: 0 });
    worker.generate(text, settings, debug, {
      onStart: (promptTokens) => setStreaming((s) => (s ? { ...s, promptTokens } : s)),
      onToken: (t) =>
        setStreaming((s) => (s ? { ...s, ids: [...s.ids, t.tokenId], tops: [...s.tops, t.top ?? []] } : s)),
      onDone: (d) => {
        setStreaming((s) => {
          const tops = debug ? (s?.tops ?? []) : undefined;
          const promptTokens = s?.promptTokens ?? 0;
          const reply: Msg = {
            role: 'assistant',
            text: tokenizer.decode(d.ids),
            ids: d.ids,
            tops,
            stats: d.stats,
            promptTokens,
          };
          setConvos((c) => ({ ...c, [model]: [...(c[model] ?? []), reply] }));
          return null;
        });
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
