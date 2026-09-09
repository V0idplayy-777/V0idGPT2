import { useEffect, useState } from 'react';
import type { ModelConfig } from '../engine/types';

function fmt(n: number): string {
  return n.toLocaleString('en-US');
}

export default function InfoPanel({ model, counted }: { model: string; counted: number | null }) {
  const [cfg, setCfg] = useState<ModelConfig | null>(null);
  useEffect(() => {
    let live = true;
    setCfg(null);
    fetch(`${import.meta.env.BASE_URL}models/${model}/config.json`)
      .then((r) => (r.ok ? r.json() : null))
      .then((j) => {
        if (live) setCfg(j as ModelConfig | null);
      })
      .catch(() => {});
    return () => {
      live = false;
    };
  }, [model]);
  if (!cfg) return <div className="panel">Loading model info…</div>;
  return (
    <div className="panel">
      <h3>Model information</h3>
      <dl className="info">
        <dt>Model</dt>
        <dd className="mono">{cfg.name}</dd>
        <dt>Parameters</dt>
        <dd className="mono">
          {fmt(counted ?? cfg.paramCount)} (~{((counted ?? cfg.paramCount) / 1e6).toFixed(2)}M)
        </dd>
        <dt>Counted from</dt>
        <dd>{counted === cfg.paramCount ? 'loaded tensors ✓' : 'config (loading…)'} </dd>
        <dt>Architecture</dt>
        <dd>Decoder-only Transformer</dd>
        <dt>Layers / heads</dt>
        <dd className="mono">
          {cfg.nLayer} / {cfg.nHead} (d={cfg.dModel}, ff={cfg.dFF})
        </dd>
        <dt>Context</dt>
        <dd className="mono">{cfg.ctx} tokens</dd>
        <dt>Tokenizer</dt>
        <dd className="mono">byte-BPE, vocab {cfg.vocab}</dd>
        <dt>Generation</dt>
        <dd>Autoregressive, local</dd>
        <dt>Weights</dt>
        <dd className="mono">Q8_0 quantized</dd>
        {cfg.training && (
          <>
            <dt>Trained on</dt>
            <dd className="mono">{fmt(cfg.training.tokens)} tokens</dd>
            <dt>Val loss</dt>
            <dd className="mono">{cfg.training.valLoss?.toFixed(3) ?? '—'}</dd>
          </>
        )}
      </dl>
    </div>
  );
}
