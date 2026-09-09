# V0idGPT2

**Five genuine pretrained Transformer language models + one genuine diffusion image model, all running 100% locally in your browser. No servers, no APIs, no canned responses.**

Live site (after enabling Pages): `https://V0idplayy-777.github.io/V0idGPT2/`

Every token you see is computed, one at a time, from the selected model's own learned weights: prompt → tokens → embeddings → Transformer blocks → logits → sampling → next token → repeat → decode. There is not a single hardcoded response, template, keyword rule, or API call anywhere in the project. Debug mode (in the UI) shows you the token IDs, per-token candidates, and timings to prove it.

## The models

| Model | Parameters (exact) | d_model | Layers | Heads | FFN | Context |
|---|---|---|---|---|---|---|
| Potato | 1,498,496 (~1.49M) | 128 | 5 | 4 | 336 | 256 |
| Toaster | 5,103,168 (~5M) | 192 | 9 | 6 | 576 | 256 |
| Microwave | 11,968,416 (~12M) | 288 | 10 | 9 | 864 | 256 |
| Blender | 28,420,992 (~28M) | 384 | 14 | 12 | 1152 | 256 |
| NuclearFridge | 52,840,960 (~52M) | 512 | 18 | 16 | 1152 | 256 |
| Paint.exe | see `public/models/paintexe/config.json` (~0.7M) | — | — | — | — | — |

Parameter counts are computed from the actual weight tensors (`countParams` in `src/engine/quant.ts`, cross-checked by `training/configs.py`) and validated at load time: the app refuses to run if the counted parameters don't match the config. Per-model training statistics (tokens seen, steps, validation loss) ship inside each `public/models/<name>/config.json` and are shown in the app's Model Information panel.

## Language-model architecture

All five LMs are decoder-only Transformers (LLaMA-style), implemented twice with identical math:

- `training/common/model.py` — the JAX reference used for training
- `src/engine/transformer.ts` — the TypeScript implementation used in the browser

Components: byte-level BPE token embeddings with **tied** input/output embeddings, RoPE positional encoding (θ=10 000, NeoX/halves style), pre-norm blocks (RMSNorm → causal multi-head self-attention → residual → RMSNorm → SwiGLU MLP → residual), final RMSNorm, logits via the embedding transpose. No biases, no dropout. Context length 256.

`src/engine/model.test.ts` verifies every shipped model against JAX reference logits (both fp32 and dequantized-Q8) computed by the exporter, so the browser math is proven — not assumed — to match training.

## Tokenizer

Byte-level BPE (GPT-2 style), vocabulary 4096: bytes 0–255, 3837 learned merges, `<bos>`/`<eos>`/`<pad>`. Pretokenization splits text into non-whitespace runs and single whitespace characters; merges apply within each piece. The merge table (`public/models/tokenizer.json`, 46 KB) is shared by all models.

- Training: `training/tokenizer/train_bpe.py`
- Python reference: `training/tokenizer/bpe.py`
- Browser port: `src/engine/tokenizer.ts` (tested for byte-exact agreement with Python on fixtures in `src/engine/fixtures/`)

## Paint.exe (image generation)

A real text-conditional diffusion model: a small U-Net (~0.7M params) trained from scratch on CIFAR-10 with DDPM (T=1000, ε-prediction) and sampled in the browser with DDIM (4–50 steps, default 16) plus optional classifier-free guidance.

- Channels (32, 64, 128); ResNet blocks + 4-head attention at 8×8; GroupNorm + SiLU + FiLM conditioning
- Text encoder: byte embeddings → mean pool → MLP (128-d), trained jointly; understands mainly the 10 CIFAR class words
- Honest limits: 32×32 output (displayed upscaled), blurry, best on single-object prompts like “a photo of a cat”
- Training: `training/paint/` · Browser: `src/engine/paint.ts` (parity-tested against JAX)

## Training

All training is genuine next-token prediction (LMs) / denoising (Paint.exe), implemented in JAX on CPU. Nothing is faked: loss curves, validation losses, and sample grids are produced by the training scripts.

**Pipeline (LMs):** `prepare_corpus.py` (build corpora) → `tokenizer/train_bpe.py` → `pretrain.py --pretokenize` → `pretrain.py --stage pretrain` → `pretrain.py --stage sft` (chat fine-tune with assistant-span loss masking) → `export_weights.py`.

**Hyperparameters:** AdamW (β1=0.9, β2=0.95, weight decay 0.1 on 2-D weights), gradient clip 1.0, warmup + cosine decay, batch sizes 48/16/8/4/2 × 256 tokens (potato → nuclearfridge). SFT uses LR 8e-5 with padded dialogue batches. Seeds and exact step counts are recorded in each model's `config.json`.

**Data** (all public, fetched from GitHub mirrors; see `training/prepare_corpus.py` for the exact recipe):

- Pretraining (~67 MB): NLTK corpora (Brown, Reuters, Gutenberg, movie reviews, webtext, Twitter samples, Europarl English, ABC news, state-of-the-union/inaugural addresses, sentences/subjectivity sets, Shakespeare, Genesis, NPS chat, switchboard, CoNLL-2000) + WikiText-2 + TinyShakespeare
- Chat SFT (~38 MB, ~100k dialogues): DailyDialog, Cornell Movie-Dialogs, NPS chat, switchboard turns
- Images: CIFAR-10 (50k train / 10k test, 32×32 RGB)

## Weight format

`public/models/<name>/weights.q8.bin`: 16-byte header (`V0Q8`, version 1) + 16-byte-aligned tensor payloads. Embeddings and norms are fp32; all 2-D linear weights are **Q8_0** (int8 blocks of 32 + fp16 scale, llama.cpp-compatible blocking), stored output-major (`[out, in]`). The browser GEMV dequantizes on the fly — quantized weights are never expanded in memory. Paint.exe uses an fp32 variant (`V0P1`). Full spec: header comment of `src/engine/quant.ts`.

## Browser inference

- `src/engine/`: `quant.ts` (Q8 loader + GEMV), `ops.ts` (RMSNorm/softmax/SiLU), `transformer.ts` (prefill + KV-cached stepping), `generate.ts` (temperature/top-k/top-p/repetition-penalty/seeded sampling), `tokenizer.ts`, `model.ts` (fetch + validation), `paint.ts` (DDIM)
- `src/worker.ts` / `src/workerpaint.ts`: inference runs in Web Workers; the UI streams tokens via messages and stays responsive (Stop button included)
- Generation settings, per-model conversations, model info, and debug mode (token IDs, top candidates, tok/s) are in the UI

## Project structure

```
src/
  engine/        inference: quant, ops, transformer, generate, tokenizer,
                 model loader, paint (+ *.test.ts, fixtures/, testutil.ts)
  components/    Chat, ModelTabs, SettingsPanel, InfoPanel, PaintPanel
  hooks/         useChatWorker, usePaintWorker
  worker.ts workerpaint.ts   Web Workers
  App.tsx main.tsx styles.css
public/models/   tokenizer.json + per-model {config, weights, parity}.json
training/
  common/model.py configs.py  JAX transformer + the five configurations
  tokenizer/     BPE trainer + Python reference encoder
  prepare_corpus.py pretrain.py sample.py export_weights.py bench.py
  paint/         cifar.py unet.py train_paint.py sample_paint.py export_paint.py
  requirements.txt
.github/workflows/  ci.yml (typecheck + tests + build), deploy.yml (Pages)
```

## Development

```bash
npm install
npm run dev        # local dev server
npm test           # vitest (includes real-weight parity tests)
npm run build      # typecheck + static build into dist/
```

Python training:

```bash
python -m venv .venv && .venv/bin/pip install -r training/requirements.txt
.venv/bin/python training/prepare_corpus.py   # needs the datasets (see script header)
.venv/bin/python training/pretrain.py --pretokenize
.venv/bin/python training/pretrain.py --model potato --stage pretrain --steps 2000 --batch 48
.venv/bin/python training/pretrain.py --model potato --stage sft --steps 800 --batch 24 --lr 8e-5
.venv/bin/python training/sample.py --model potato --ckpt /home/user/checkpoints/potato_sft/step_800.npz
.venv/bin/python training/export_weights.py --model potato --ckpt /home/user/checkpoints/potato_sft/step_800.npz
```

## Deploy to GitHub Pages

`vite.config.ts` sets `base: '/V0idGPT2/'` and `npm run build` emits a fully static `dist/` (the `public/models/` weights are copied in). Push to `main` and the `deploy.yml` workflow builds + deploys via official Pages actions — then enable Pages in repo Settings (Source: GitHub Actions). No backend, no secrets, no environment variables.

## Tests — what they prove

- `tokenizer.test.ts`: TS encoder matches Python byte-for-byte (fixtures), roundtrips tricky text
- `quant.test.ts`: fp16 codec, Q8 GEMV ≈ fp32 GEMV, exact param counting
- `ops.test.ts`: RMSNorm/softmax/SiLU correctness
- `transformer.test.ts`: determinism, finite logits, causal masking (future can't affect past), attention matters, KV-cache ≡ full forward
- `generate.test.ts`: greedy/top-k/top-p/penalty/seed semantics, streaming loop, stop
- `model.test.ts`: all five real models load, exact param counts, TS logits match JAX (Q8 parity), top-1 preserved by quantization
- `paint.test.ts`: text-cond, UNet forward, and an 8-step DDIM run match JAX; seed determinism
- CI additionally typechecks and verifies the production Pages build

## Limitations (read this)

- These are tiny models trained on ~2 CPU cores with far fewer tokens than Chinchilla-optimal. They write simple, sometimes repetitive English; the bigger ones are smoother but none reason, follow complex instructions reliably, or know facts beyond their small corpus. They will happily make things up — like all LMs, only smaller and more honest about it.
- Context is 256 tokens; long chats are tail-truncated (reported in debug mode).
- Q8 quantization slightly perturbs logits (top-1 agreement is tested).
- Paint.exe is a sub-1M-parameter CIFAR model: blurry 32×32 images of 10 object classes. Anything else is out of distribution.
- First load downloads the selected model's weights (1.7–58 MB); switching models downloads the newly selected one.

## License / attribution

Code in this repository is original. Training data: NLTK corpus collection (various research-friendly licenses), WikiText-2 (CC-BY-SA 3.0), DailyDialog (research use), Cornell Movie-Dialogs Corpus (research use), TinyShakespeare (public domain), CIFAR-10 (please cite Krizhevsky & Hinton 2009). Model weights are artifacts of training on that data.
