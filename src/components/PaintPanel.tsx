import { useEffect, useRef, useState } from 'react';
import { usePaintWorker } from '../hooks/usePaintWorker';

const QUICK = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'];

function drawImage(canvas: HTMLCanvasElement, img: Float32Array): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const data = ctx.createImageData(32, 32);
  for (let i = 0; i < 32 * 32; i++) {
    data.data[i * 4] = Math.round(img[i * 3] * 255);
    data.data[i * 4 + 1] = Math.round(img[i * 3 + 1] * 255);
    data.data[i * 4 + 2] = Math.round(img[i * 3 + 2] * 255);
    data.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(data, 0, 0);
}

export default function PaintPanel() {
  const { loaded, paramCount, error, sampling, progress, sample, stop } = usePaintWorker();
  const [prompt, setPrompt] = useState('a photo of a cat');
  const [steps, setSteps] = useState(16);
  const [seed, setSeed] = useState(3);
  const [guidance, setGuidance] = useState(1);
  const [info, setInfo] = useState('');
  const [hasImage, setHasImage] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const c = canvasRef.current;
    if (c && !hasImage) {
      const ctx = c.getContext('2d');
      if (ctx) {
        ctx.fillStyle = '#21262d';
        ctx.fillRect(0, 0, 32, 32);
      }
    }
  }, [hasImage]);

  const run = () => {
    if (!loaded || sampling || !prompt.trim()) return;
    sample(prompt.trim(), steps, seed >>> 0, guidance, {
      onStep: (_d, _t, preview) => {
        if (canvasRef.current) drawImage(canvasRef.current, preview);
        setHasImage(true);
      },
      onDone: (image, ms, stopped) => {
        if (canvasRef.current) drawImage(canvasRef.current, image);
        setHasImage(true);
        setInfo(`${stopped ? 'stopped' : 'done'} in ${(ms / 1000).toFixed(1)}s · ${steps} steps · seed ${seed} · guidance ${guidance}`);
      },
    });
  };

  return (
    <div className="paint-grid">
      <div className="paint-main panel">
        <div className="paint-canvas-wrap">
          <canvas ref={canvasRef} width={32} height={32} className="paint-canvas" />
        </div>
        {sampling && progress && (
          <div className="progress">
            <div className="progress-bar" style={{ width: `${(100 * progress.done / progress.total).toFixed(0)}%` }} />
          </div>
        )}
        <div className="muted small">{sampling ? `denoising step ${progress?.done ?? 0}/${progress?.total ?? steps}…` : info || 'Idle. Type a prompt and hit Paint.'}</div>
        {error && <div className="error">⚠ {error}</div>}
        {!loaded && !error && <div className="muted">Loading Paint.exe weights…</div>}
      </div>
      <div className="panel">
        <h3>Paint.exe — tiny text-to-image diffusion</h3>
        <label className="setting">
          <span className="setting-head">
            <span>Prompt</span>
          </span>
          <input
            type="text"
            className="paint-prompt"
            value={prompt}
            disabled={sampling}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') run();
            }}
          />
        </label>
        <div className="quick">
          {QUICK.map((q) => (
            <button key={q} className="example" disabled={sampling} onClick={() => setPrompt(`a photo of a ${q}`)}>
              {q}
            </button>
          ))}
        </div>
        <label className="setting">
          <span className="setting-head">
            <span>DDIM steps</span>
            <span className="setting-val mono">{steps}</span>
          </span>
          <input type="range" min={4} max={50} step={1} value={steps} disabled={sampling} onChange={(e) => setSteps(Number(e.target.value))} />
        </label>
        <label className="setting">
          <span className="setting-head">
            <span>Guidance</span>
            <span className="setting-val mono">{guidance.toFixed(1)}</span>
          </span>
          <input
            type="range"
            min={1}
            max={5}
            step={0.5}
            value={guidance}
            disabled={sampling}
            onChange={(e) => setGuidance(Number(e.target.value))}
          />
        </label>
        <label className="setting">
          <span className="setting-head">
            <span>Seed</span>
          </span>
          <span className="seed-row">
            <input type="number" value={seed} disabled={sampling} onChange={(e) => setSeed(Number(e.target.value) >>> 0)} />
            <button className="btn small" disabled={sampling} title="Random seed" onClick={() => setSeed((Math.random() * 2 ** 31) >>> 0)}>
              🎲
            </button>
          </span>
        </label>
        <div className="paint-actions">
          {sampling ? (
            <button className="btn stop" onClick={stop}>
              Stop
            </button>
          ) : (
            <button className="btn send" onClick={run} disabled={!loaded || !prompt.trim()}>
              🎨 Paint
            </button>
          )}
          <button
            className="btn small"
            disabled={!hasImage || sampling}
            onClick={() => {
              const c = canvasRef.current;
              if (!c) return;
              const a = document.createElement('a');
              a.href = c.toDataURL('image/png');
              a.download = 'paintexe.png';
              a.click();
            }}
          >
            ⬇ PNG
          </button>
        </div>
        <dl className="info">
          <dt>Parameters</dt>
          <dd className="mono">{paramCount !== null ? paramCount.toLocaleString('en-US') : '…'}</dd>
          <dt>Model</dt>
          <dd>conditional DDPM U-Net</dd>
          <dt>Resolution</dt>
          <dd className="mono">32×32 (shown upscaled)</dd>
          <dt>Inference</dt>
          <dd>local DDIM, this tab</dd>
        </dl>
        <p className="muted small">
          Honest limits: this is a 1.3M-parameter model trained on CIFAR-10. Expect blurry 32×32 images, mostly of
          the 10 object classes it knows. No stock images, no APIs — every pixel is denoised from pure noise by the
          model&apos;s own weights.
        </p>
      </div>
    </div>
  );
}
