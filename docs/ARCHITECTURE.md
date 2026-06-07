# Architecture

Complementary detail for reviewers who want it. The top-level `README.md` is
self-contained and is all that is needed to grant the seals.

## Pipeline

```
data/cultural_strata.jsonl  ┐
data/control_generic.jsonl  �├─► culturaquant.data ──► culturaquant.scorer ──► preds__*.jsonl
BRoverbs (HF, at runtime)   ┘     (harmonized Item)     (constrained LL, GPU)   + manifest.json
                                                                                      │
                                                          culturaquant.analyze ◄──────┘
                                                          (pure-Python, no GPU)
                                                                  │
                                                          results_macros.tex
```

The two halves are deliberately decoupled:

- **Scoring** (`run.py` + `scorer.py`) is the only GPU-bound part. It loads each
  model at a chosen precision and writes one JSONL of per-item predictions per
  `(model, precision)`. It is the run of record and produces `data/results/grid/`.
- **Analysis** (`analyze.py` + `stats.py`) is pure standard library. It reads the
  committed prediction grid and emits every LaTeX macro the paper uses. This is why
  the macros can be regenerated on any machine with no GPU and no heavy dependencies,
  which is what makes the result auditable.

## Module responsibilities (SOLID)

| Module | Single responsibility |
|---|---|
| `data.py` | Load each item source and harmonize to one `Item` schema. |
| `scorer.py` | Load a model at fp16/int8/nf4 and score one MCQ item by constrained log-likelihood. |
| `run.py` | Drive the model × precision grid; record the manifest + reference machine. |
| `stats.py` | Wilson CI, McNemar, paired bootstrap, the saturated interaction, the model-clustered bootstrap, DerSimonian–Laird, Holm. No I/O. |
| `analyze.py` | Read the grid; compute PILOT / E1 / E2 / E3 and the round-2 robustness numbers; emit the macros. No statistics math of its own. |

## Determinism

Scoring uses no sampling (argmax over option log-likelihoods), so the per-item
FP16-vs-quantized comparison is a clean paired measurement and reruns are exact.
Every random procedure (BRoverbs subsampling, all bootstraps) is seeded with
`20260607`. The reference machine, seed, model ids, precisions and item counts are
recorded in `data/results/grid/manifest.json`.

## Why the analysis emits the interaction, not raw drops

A uniform compression hit that lowers cultural and control accuracy equally yields
an interaction coefficient near zero (unit-tested), so the design cannot manufacture
a "cultural erodes first" headline out of a uniform drop. The interaction is the
difference-in-differences of the four cell log-odds (cultural/control × fp/low),
Haldane–Anscombe corrected; the headline CI is a model-clustered bootstrap so
model-level non-independence widens the interval honestly.
