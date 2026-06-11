# Architecture

Complementary detail for reviewers who want it. The top-level `README.md` is
self-contained and is all that is needed to grant the seals.

## Pipeline

```
v2/allband_predictions/cq_cult_items.jsonl ┐
v2/allband_predictions/cq_ctrl_items.jsonl ┘
        │  (cultural probe + rarity-matched control)
        ▼
culturaquant.run + culturaquant.scorer ──► cyclic__<model>__<precision>.jsonl
   (constrained log-likelihood, GPU)         (per-item predictions, 5 cyclic rotations)
        │
        ▼
v2/build/cq_ci.py · cq_rstd.py · cq_full_analysis.py ──► v2/build/gen_allband_macros.py
   (pure-Python, no GPU: differential erosion,                 │
    item-clustered bootstrap CIs, RStd)                        ▼
                                                  v2/allband/results_macros.tex
```

The two halves are deliberately decoupled:

- **Scoring** (`run.py` + `scorer.py`) is the only GPU-bound part. It loads each model
  at a chosen precision and writes one JSONL of per-item predictions per
  `(model, precision)`. It is the run of record and produces
  `v2/allband_predictions/{cult,ctrl}/`.
- **Analysis** (`v2/build/cq_*.py`) is pure standard library. It reads the committed
  prediction grid and recomputes every number the paper reports, then
  `gen_allband_macros.py` emits the LaTeX macros. This is why the macros can be
  regenerated on any machine with no GPU and no heavy dependencies, which is what makes
  the result auditable. `v2/build/reproduce_allband.sh` runs the whole no-GPU path and
  asserts the macros are byte-identical to the committed reference.

## Module responsibilities (SOLID)

| Module | Single responsibility |
|---|---|
| `data.py` | Load each item source and harmonize to one `Item` schema. |
| `scorer.py` | Load a model at fp16/int8/nf4 and score one MCQ item by constrained log-likelihood; apply the per-item option permutation. |
| `run.py` | Drive the model × precision grid over cyclic option rotations; record the manifest + reference machine. |
| `v2/build/cq_ci.py` | Per-model and pooled differential erosion with item-clustered bootstrap CIs; per-band and per-region cultural drops. |
| `v2/build/cq_rstd.py` | Position-choice spread (RStd) per measurable model and precision. |
| `v2/build/cq_full_analysis.py` | Measurable-set determination (Wilson lower bound vs the chance floor) and the full per-model table. |
| `v2/build/gen_allband_macros.py` | Emit every paper macro from the analysis outputs. |

## Determinism

Scoring uses no sampling (argmax over option log-likelihoods), so the per-item
FP16-vs-quantized comparison is a clean paired measurement and reruns are exact. Options
are deterministically permuted per item (`perm_options_v2`, seeded by item id) so the gold
answer is not pinned at index 0, and the same permutation is applied across precisions.
The bootstraps are seeded with `20260609`. The reference machine, seed, model ids,
precisions and item counts are recorded in the prediction manifests.

## Why the analysis reports differential erosion

The primary quantity is **differential erosion**: the drop in cultural accuracy from full
precision to a quantized precision, minus the same drop for the rarity-matched control,
per matched item pair. A uniform compression hit that lowers cultural and control accuracy
equally yields a differential near zero, so the design cannot manufacture a "cultural
erodes first" headline out of a uniform drop. The 95% confidence interval is an
item-clustered bootstrap that resamples items (not rotations), so within-item correlation
across rotations does not understate the interval. Erosion is reported only for the
measurable models whose full-precision cultural accuracy clears the five-way chance floor
(Wilson lower bound above 20%).
