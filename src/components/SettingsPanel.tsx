import type { GenerationSettings } from '../engine/types';

function Row({ label, value, children }: { label: string; value: string; children: React.ReactNode }) {
  return (
    <label className="setting">
      <span className="setting-head">
        <span>{label}</span>
        <span className="setting-val mono">{value}</span>
      </span>
      {children}
    </label>
  );
}

export default function SettingsPanel({
  settings,
  onChange,
  debug,
  onDebug,
  disabled,
}: {
  settings: GenerationSettings;
  onChange: (s: GenerationSettings) => void;
  debug: boolean;
  onDebug: (d: boolean) => void;
  disabled: boolean;
}) {
  const set = (patch: Partial<GenerationSettings>) => onChange({ ...settings, ...patch });
  return (
    <div className="panel">
      <h3>Generation settings</h3>
      <Row label="Temperature" value={settings.temperature.toFixed(2)}>
        <input
          type="range"
          min={0}
          max={2}
          step={0.05}
          value={settings.temperature}
          disabled={disabled}
          onChange={(e) => set({ temperature: Number(e.target.value) })}
        />
      </Row>
      <Row label="Top-k" value={settings.topK === 0 ? 'off' : String(settings.topK)}>
        <input
          type="range"
          min={0}
          max={100}
          step={1}
          value={settings.topK}
          disabled={disabled}
          onChange={(e) => set({ topK: Number(e.target.value) })}
        />
      </Row>
      <Row label="Top-p" value={settings.topP.toFixed(2)}>
        <input
          type="range"
          min={0.05}
          max={1}
          step={0.05}
          value={settings.topP}
          disabled={disabled}
          onChange={(e) => set({ topP: Number(e.target.value) })}
        />
      </Row>
      <Row label="Max tokens" value={String(settings.maxTokens)}>
        <input
          type="range"
          min={8}
          max={248}
          step={8}
          value={settings.maxTokens}
          disabled={disabled}
          onChange={(e) => set({ maxTokens: Number(e.target.value) })}
        />
      </Row>
      <Row label="Repetition penalty" value={settings.repeatPenalty.toFixed(2)}>
        <input
          type="range"
          min={1}
          max={1.5}
          step={0.01}
          value={settings.repeatPenalty}
          disabled={disabled}
          onChange={(e) => set({ repeatPenalty: Number(e.target.value) })}
        />
      </Row>
      <Row label="Seed" value={String(settings.seed)}>
        <span className="seed-row">
          <input
            type="number"
            value={settings.seed}
            disabled={disabled}
            onChange={(e) => set({ seed: Number(e.target.value) >>> 0 })}
          />
          <button
            className="btn small"
            disabled={disabled}
            title="Random seed"
            onClick={() => set({ seed: (Math.random() * 2 ** 31) >>> 0 })}
          >
            🎲
          </button>
        </span>
      </Row>
      <label className="setting check">
        <input type="checkbox" checked={debug} onChange={(e) => onDebug(e.target.checked)} />
        <span>Debug mode (token IDs, candidates, timings)</span>
      </label>
    </div>
  );
}
