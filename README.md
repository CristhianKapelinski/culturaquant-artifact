# CulturaQuant — Brazilian cultural-knowledge erosion under SLM quantization

CulturaQuant measures whether **Brazilian cultural knowledge** in small language models
(0.5B–9B) erodes faster under post-training quantization than equally rare **generic world
knowledge**, on identical inputs. It ships a 5-way multiple-choice probe of rare Brazilian
cultural facts drawn from Wikidata, each item paired with a non-Brazilian fact matched on
rarity (so a difference reflects cultural content, not rarity), scores each model at FP16,
bitsandbytes int8, and NF4 4-bit with deterministic constrained log-likelihood, and reports
**differential erosion** (the cultural-accuracy drop minus the matched-control drop, in
percentage points) with item-clustered bootstrap confidence intervals.

**Headline (all-band grid, 11 models, 6 measurable above chance).** Cultural knowledge does
**not** erode faster than the rarity-matched control: the pooled 8-bit differential is
**+0.2 pp** (95% CI [−0.4, 0.7]), every per-model interval crosses zero, and the design
excludes a differential beyond about **1.8 pp** (MDE). At 4-bit the picture is model-specific
and unstable rather than a larger cultural loss (the three Qwen2.5 models erode cultural
knowledge faster by 3.1–3.5 pp, Qwen3-1.7B slower by 3.6 pp, Qwen3-4B/Phi-3.5 not at all;
pooled +0.7 pp [−0.1, 1.6], signs disagree). The five models that sit at the five-way chance
floor at full precision (Qwen2.5-0.5B, Qwen3-0.6B, Mistral-7B, and both Tucano models) are
reported as unmeasurable, not as nulls.

> Paper: ENIAC 2026 (under review). **This README is the only document needed for artifact
> review;** `docs/` and the paper-side notes are complementary.

## README structure
| Section | What |
|---|---|
| [Seals](#considered-seals) | Why each badge holds |
| [Basic info](#basic-information) | Reference machines |
| [Dependencies](#dependencies) | Pinned, `uv`-managed |
| [Security](#security-concerns) | Runs locally |
| [Installation](#installation) | Clone + `uv sync` |
| [Minimal test](#minimal-test) | One command, no GPU |
| [Experiments](#experiments) | Claims + commands |
| [License](#license) | MIT |

## Considered seals
- **Disponível (SeloD):** public repo, MIT-licensed, anonymized content.
- **Funcional (SeloF):** one GPU-free command recomputes the differential-erosion CIs and the
  position-bias RStd from the committed per-item predictions and regenerates every paper macro.
- **Sustentável (SeloS):** `src/` layout, typed modules, unit tests, pinned `uv.lock`, single
  pinned Docker image; no hardcoded paths (analysis paths default to the committed data and are
  overridable by environment variable).
- **Reprodutível (SeloR):** deterministic scoring (no sampling, fixed seed 20260609); the
  committed run of record (11 models × 3 precisions × 700 items × 5 rotations) regenerates every
  macro byte-for-byte without a GPU, and the from-scratch grid reproduces the predictions on a
  16 GB GPU.

## Basic information
| Item | Value |
|---|---|
| OS | Ubuntu 22.04 (Docker base `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`) |
| Runtime | Python 3.11, torch 2.8.0+cu128, bitsandbytes 0.48.2 |
| GPU (scoring) | NVIDIA RTX 5080 (16 GB) for the all-band grid; any ≥16 GB GPU works |
| GPU (reproduction) | **none** — the committed-predictions path is CPU-only |
| RAM / disk | 8 GB RAM; ~70 MB for the committed predictions, ~25 GB more for model cache if re-scoring |
| Reproduce path | recomputes all macros from committed data in <1 min, no GPU |

Heavy/GPU work runs the Docker image on a GPU host; growing data (model cache, predictions)
goes to a bind-mounted directory, never inside the image. The committed run of record lives in
`v2/allband_predictions/` and needs no GPU to re-analyze.

## Dependencies
Managed with **`uv`** (`pyproject.toml` + committed `uv.lock`); the reviewer installs with
`uv sync` and runs via `uv run`. The GPU-free reproduction needs only the Python standard
library. Key pins for the scoring path: `torch>=2.5,<2.9`, `transformers>=4.44,<4.57`,
`bitsandbytes>=0.43,<0.49`, `datasets`, `accelerate`. The cultural probe and the rarity-matched
control are author-built JSONL shipped in the repo; no dataset needs manual download.

## Security concerns
- Everything runs locally; no external API, no monetary cost.
- The only network access is read-only model/dataset downloads from the HuggingFace Hub, and
  only when re-scoring from scratch; the committed-predictions reproduction is offline.
- No credentials are needed for the default model set (all public). Gated models would need an
  HF token and are out of the default scope.
- Bind-mounted data dir holds only model cache and prediction JSONL; no secrets.

## Installation
```bash
git clone <repo-url> culturaquant && cd culturaquant/artifact
curl -LsSf https://astral.sh/uv/0.11.2/install.sh | sh        # install uv (~10 s)
uv sync --extra dev                                            # resolve from uv.lock (~3 min)
```
Docker path (recommended only for the from-scratch GPU runs), on a GPU host:
```bash
docker build -t culturaquant:1.0 .                            # ~3 min (cached layers faster)
```

## Minimal test
One command, **no GPU**, <1 min — recomputes the headline from the committed run of record and
asserts the regenerated macros are byte-identical to the committed reference:
```bash
v2/build/reproduce_allband.sh
```
Expected tail: the per-model differential-erosion table (pooled int8 `+0.2 pp [-0.4, 0.7]`,
pooled nf4 `+0.7 pp [-0.1, 1.6]`, MDE `1.8 pp`), the fp16 RStd range `0.120 to 0.198`, and
`OK_MACROS_REPRODUCED`. The unit tests (`./scripts/test.sh`, `44 passed`) separately exercise
the scoring, option-shuffle, and statistics helpers.

## Experiments
The MAIN claim is **the 8-bit differential-erosion null on the measurable models**. Default to
the GPU-free path; the from-scratch grid is an explicit opt-in.

### Regenerate every paper number from the committed grid (no GPU, <1 min) — recommended
Re-derives the differential-erosion CIs, the per-band and per-region drops, and the
position-bias RStd from the committed per-item predictions in `v2/allband_predictions/`, then
regenerates `v2/allband/results_macros.tex` and checks it byte-for-byte:
```bash
v2/build/reproduce_allband.sh
```
- **Resources:** <1 GB RAM, no GPU, no network.
- **Expected result:** prints `OK_MACROS_REPRODUCED`; pooled int8 differential **+0.2 pp**
  (95% CI [−0.4, 0.7]), pooled nf4 **+0.7 pp** ([−0.1, 1.6]), MDE **1.8 pp**, 6 of 11 models
  measurable. The individual analyses can also be run directly:
  ```bash
  python3 v2/build/cq_ci.py            # per-model + pooled differential erosion with CIs
  python3 v2/build/cq_rstd.py          # position-choice spread (RStd) per measurable model
  python3 v2/build/cq_full_analysis.py # measurable-set table + per-band/region drops
  ```
  All three default to the committed predictions; set `CQ_OUT` (plus `CQ_SUB_CULT`,
  `CQ_SUB_CTRL`, `CQ_CULT_ITEMS`, `CQ_CTRL_ITEMS`) to point them at a fresh grid instead.

### Full reproduction — score the 11-model grid from scratch (GPU, several hours)
- **Description:** the run of record behind every macro: 11 models × {FP16, int8, NF4} ×
  700 items × 5 cyclic rotations, scored by deterministic constrained log-likelihood.
- **Execution (on a GPU host):**
  ```bash
  CQ_DATA=$HOME/cq_data ./scripts/run_grid.sh        # writes per-item prediction JSONL
  CQ_OUT=$CQ_DATA/out CQ_SUB_CULT=<cult-dir> CQ_SUB_CTRL=<ctrl-dir> \
    python3 v2/build/cq_ci.py                         # analyze the fresh grid
  ```
- **Expected time:** several hours on one 16 GB GPU (plus model downloads on first run).
- **Expected result:** prediction JSONL matching `v2/allband_predictions/` and, after analysis,
  the same pooled int8 **+0.2 pp** / nf4 **+0.7 pp** / MDE **1.8 pp**.

Claims map: the released **probe and rarity-matched control** (`v2/allband_predictions/cq_cult_items.jsonl`,
`cq_ctrl_items.jsonl`, with `matched_br_id` linking each pair); the **measurable set** (Wilson
lower bound above the 20% chance floor, `cq_full_analysis.py`); the **8-bit null and 4-bit
instability** (`cq_ci.py`); **no growth with rarity** and **no region erodes first** (per-band
and per-region drops in `cq_ci.py`); **position-bias check** (`cq_rstd.py`, provenance in
`v2/allband/rstd_measurable.json`). The calibration-based quantization methods and the upper
deployment band are future work, not run here.

## License
MIT — see `LICENSE`.
