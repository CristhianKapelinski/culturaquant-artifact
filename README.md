# Tool: CulturaQuant — differential cultural-knowledge erosion under SLM quantization

CulturaQuant measures whether **Brazilian cultural knowledge** in small language models
(0.5B–4B) erodes faster under post-training quantization than matched **generic world
knowledge**, on identical inputs. It ships a macro-region-stratified 5-way MCQ probe
(proverbs, regional facts, cuisine, geography) and a **format-matched** generic control
(matched in format, not difficulty),
scores each model at FP16, bitsandbytes int8, and NF4 4-bit with deterministic constrained
log-likelihood, and reports Wilson CIs, McNemar paired tests, a paired bootstrap, and a
**knowledge-type x bit-width interaction term** so a uniform drop yields a null instead of
a manufactured effect. Headline: on the 70 freshly-authored localized items (the
contribution), the int8 interaction is significantly negative (**−0.230**, 95% CI
[−0.439, −0.040], excludes 0): cultural knowledge erodes faster than the control. Pooled
over all cultural items the interaction is near-null (**−0.191**, [−0.861, 0.259]),
diluted by the reused near-random proverb stratum and a near-ceiling control. Under int8
cultural accuracy drops **3.1 pp** (95% CI [1.1, 5.2], excludes 0) while the control moves
**−0.4 pp** (CI includes 0); the strongest single-model effect is **Qwen2.5-3B int8,
cultural −7.3 pp, McNemar p=0.019** (does not survive Holm correction).

> Paper: ENIAC 2026 (under review). **This README is the only document needed for artifact
> review;** `docs/` and the paper-side notes are complementary.

## README structure
| Section | What |
|---|---|
| [Seals](#considered-seals) | Why each badge holds |
| [Basic info](#basic-information) | One reference machine |
| [Dependencies](#dependencies) | Pinned, `uv`-managed |
| [Security](#security-concerns) | Runs locally |
| [Installation](#installation) | Clone + `uv sync` |
| [Minimal test](#minimal-test) | One command |
| [Experiments](#experiments) | Claims + commands |
| [License](#license) | MIT |

## Considered seals
- **Disponível (SeloD):** public repo, MIT-licensed, anonymized content.
- **Funcional (SeloF):** one command runs the real scoring + analysis pipeline and emits the
  results macros and per-stratum accuracy.
- **Sustentável (SeloS):** `src/` layout, typed modules, unit tests, pinned `uv.lock`, single
  pinned Docker image; no hardcoded paths.
- **Reprodutível (SeloR):** deterministic scoring (no sampling, fixed seed), pinned image and
  lock, content-tagged inputs; the quick path reproduces the headline-shaped result in minutes
  and the full grid regenerates every macro.

## Basic information
| Item | Value |
|---|---|
| OS | Ubuntu 22.04 (Docker base `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`) |
| Runtime | Python 3.11, torch 2.8.0+cu128, bitsandbytes 0.48.2 |
| GPU | NVIDIA RTX 3060 (12 GB, Ampere sm_86) — any sm_80–sm_89 GPU with ≥12 GB works |
| RAM / disk | 16 GB RAM; ~25 GB disk for model cache + image |
| Quick path | runs on one 12 GB GPU in ~5 min |

One reference machine only. Heavy/GPU work runs the Docker image on a GPU host; growing data
(model cache, predictions) goes to a bind-mounted directory, never inside the image.

## Dependencies
Managed with **`uv`** (`pyproject.toml` + committed `uv.lock`); the reviewer installs with
`uv sync` and runs via `uv run`. Key pins: `torch>=2.5,<2.9`, `transformers>=4.44,<4.57`,
`bitsandbytes>=0.43,<0.49`, `datasets`, `accelerate`. The proverbs stratum (BRoverbs) is
fetched from the HuggingFace Hub at runtime by the loader; the cultural strata and generic
control are author-written JSONL shipped in `data/`. No dataset needs manual download.

## Security concerns
- Everything runs locally; no external API, no monetary cost.
- The only network access is read-only model/dataset downloads from the HuggingFace Hub.
- No credentials are needed for the default model set (all public). Gemma-2/Llama-3.2 would
  need an HF token and are out of the default scope.
- Bind-mounted data dir holds only model cache and prediction JSONL; no secrets.

## Installation
```bash
git clone <repo-url> culturaquant && cd culturaquant/artifact
curl -LsSf https://astral.sh/uv/0.11.2/install.sh | sh        # install uv (~10 s)
uv sync --extra dev                                            # resolve from uv.lock (~3 min)
```
Docker path (recommended for the GPU runs), on a GPU host:
```bash
docker build -t culturaquant:1.0 .                            # ~3 min (cached layers faster)
```

## Minimal test
One command, no GPU, ~5 s — runs the real statistics pipeline (Wilson CIs, McNemar, paired
bootstrap, the saturated interaction, the model-clustered bootstrap, DerSimonian–Laird and
Holm) and asserts the differential-erosion logic:
```bash
uv run pytest tests/ -q
```
Expected: `15 passed`. This exercises that the interaction term is negative under genuine
differential erosion and near zero under a uniform drop (the anti-trophy guarantee), and that
the clustered/random-effects/Holm helpers behave.

## Experiments
The MAIN claim is **C2 (differential erosion under int8)**. Default to the quick path; the
full grid is an explicit opt-in. You may also inspect the pre-computed results in
`data/results/` instead of re-running.

### Regenerate every paper number from the committed grid (no GPU, ~30 s) — recommended first
This is the fastest, GPU-free way to confirm the headline. It re-derives **every** LaTeX macro
the paper uses (PILOT, E1, E2, E3, the localized/clustered/random-effects interaction, the
per-stratum and Holm-corrected numbers) from the committed per-item predictions in
`data/results/grid/`, with zero hand-transcription:
```bash
uv run python -m culturaquant.analyze --out-dir data/results/grid --macro-out /tmp/macros.tex --run-date 2026-06-07 && diff <(sort /tmp/macros.tex) <(sort data/results/results_macros.tex) && echo OK_MACROS_REPRODUCED
```
- **Expected time:** ~30 s. **Resources:** <1 GB RAM, no GPU.
- **Expected result:** prints `OK_MACROS_REPRODUCED`; the regenerated macros are byte-identical
  to the committed `data/results/results_macros.tex`, including localized int8 interaction
  **−0.230** (95% CI [−0.439, −0.040]) and pooled **−0.191** ([−0.861, 0.259]).

The optional human-readable cross-check (per-test McNemar/Holm table, gate sensitivity):
```bash
uv run python scripts/compute_round2_clustered.py --out-dir data/results/grid
```

### Main claim — differential cultural-vs-control erosion (quick: ~5 min, one 12 GB GPU)
- **Description:** score 2 band-clearing models at FP16 and NF4 on the 120-item probe+control
  and emit the macros; confirms cultural erodes and the interaction is computed.
- **Execution:**
  ```bash
  CQ_DATA=$HOME/cq_data QUICK=1 ./scripts/run_grid.sh && \
  CQ_DATA=$HOME/cq_data ./scripts/emit_macros.sh grid_quick
  ```
- **Expected time:** ~5 min (models cached after first run). **Resources:** ~12 GB VRAM, ~10 GB disk.
- **Expected result:** `$CQ_DATA/out/grid_quick/results_macros.tex` with `\cqHeadNBandClearers`=2
  and a per-precision cultural delta; Qwen2.5-3B shows a clear FP16→NF4 cultural drop.

### Full reproduction — all 8 models × 3 precisions, 200 items (full: ~30 min)
- **Description:** the run of record behind every macro in the paper.
- **Execution:**
  ```bash
  CQ_DATA=$HOME/cq_data ./scripts/run_grid.sh && \
  CQ_DATA=$HOME/cq_data ./scripts/emit_macros.sh grid
  ```
- **Expected time:** ~30 min on one 12 GB GPU (plus model downloads on first run).
- **Expected result:** `results_macros.tex` matching `data/results/results_macros.tex`;
  pooled int8 cultural **−3.1 pp** vs control **−0.4 pp**, localized-item interaction
  **−0.230** (95% CI [−0.439, −0.040]), Qwen2.5-3B int8 cultural **−7.3 pp** (McNemar
  p=0.019). Pre-computed copies are in `data/results/`.

Claims map: **C1** the released probe (`data/cultural_strata.jsonl`, `data/control_generic.jsonl`,
the BRoverbs loader); **C2** the differential, primary on the localized items and a
sensitivity over all cultural items (`\cqHead*`, `\cqLocInt*`, `\cqEOne*`); **size scaling**
(`\cqETwo*`); **E3 regional** N+NE vs SE+S (`\cqEThree*`); **PILOT gate** (`\cqPilot*`, `\cqHeadBandClearers`).
The method/calibration sweep, option-order shuffle, free-generation cross-check, and the
rarity/leakage controls are future work (see `docs/` / paper notes), not run here.

## License
MIT — see `LICENSE`.
