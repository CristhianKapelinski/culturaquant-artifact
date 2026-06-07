# Dataset internals

Complementary detail. The released probe is the C1 contribution of the paper.

## Item schema (JSONL, one object per line)

```json
{"id": "reg-001", "stratum": "regional_facts", "region": "NE",
 "input": "<question in Brazilian Portuguese>",
 "alternatives": ["opt A", "opt B", "opt C", "opt D", "opt E"],
 "correct_index": 0}
```

Every item is a single-correct 5-way MCQ, so the random floor is 0.20. Gold answers
are author-fixed and never taken from a model under test.

## What ships in this repo

| File | Group | Items | Strata | Regions | Notes |
|---|---|---|---|---|---|
| `data/cultural_strata.jsonl` | cultural | 70 | regional_facts 38, cuisine 20, geography 12 | N 15, NE 15, CO 13, SE 13, S 14 | Freshly authored to avoid BLUEX/ENEM contamination. |
| `data/control_generic.jsonl` | control | 50 | generic | — | World geography/history/science, format-matched 5-way MCQ. |

These 120 author-written items are released under the repo's MIT license.

## What is NOT redistributed

The proverbs stratum is **not** shipped as data. BRoverbs has no dataset license on
the Hub (the card shows license "coming soon"), so the loader pulls it from the
HuggingFace Hub at runtime (`Tropic-AI/BRoverbs`, `proverb_to_history` split) and
subsamples deterministically (seed `20260607`). Only the stratum/region tag and the
item id are author-owned. Set `--n-proverbs 0` to run without any Hub access.

## Macro-region balance

The cultural strata are tagged by Brazilian macro-region (N, NE, CO, SE, S). The
pre-registered coarse contrast is N+NE vs SE+S; the Centro-Oeste (CO) items are
reported but held out of the coarse split because finer per-region claims are
underpowered at ~13–15 items/region (E3 in the paper).

## Honest scope of the released probe

The probe is the first Brazilian state-level cultural-knowledge MCQ probe with a
matched generic control. It is **not** yet inter-annotator validated nor
leakage-audited against an exact-count index; both are stated as future work in the
paper and are not claimed here. The freshly-authored items reduce, but do not prove
the absence of, pretraining leakage.
