# Dataset internals

Complementary detail. The released probe is the central contribution of the paper;
this file documents its schema, balance, and scope.

## Item schema (JSONL, one object per line)

```json
{"id": "v2-mb-001", "set": "midband_br", "relation": "figure_to_state",
 "qid": "Q65161183", "statement": "<Wikidata claim the item encodes>",
 "source_url": "https://www.wikidata.org/wiki/Q65161183", "gold_verified": true,
 "region": "N",
 "input": "<question in Brazilian Portuguese>",
 "alternatives": ["opt A", "opt B", "opt C", "opt D", "opt E"],
 "alt_qids": ["Q...", "..."], "correct_index": 0, "distractor_qids": ["..."],
 "rarity": {"sitelinks": 2, "log_pageviews": 0.0, "band": "sl2-3"},
 "stratum": "figure_to_state", "group": "cultural"}
```

Every item is a single-correct 5-way MCQ, so the random floor is 0.20. Gold answers are
verified against Wikidata (`gold_verified`) and never taken from a model under test. Each
item carries its Wikidata QID and source URL for full provenance, and a `rarity` block
(sitelink count, log pageviews, and the rarity band the item falls in).

## What ships in this repo

| File | Group | Items | Tagging | Notes |
|---|---|---|---|---|
| `data/items/cultural.jsonl` | cultural | 700 | macro-region + rarity band | Rare Brazilian facts drawn from Wikidata: which state a person was born in or a festival happens in. |
| `data/items/control.jsonl` | control | 700 | country + rarity band, `matched_br_id` | Equally rare non-Brazilian facts, one paired to each cultural item on rarity. |

Both files are released under the repo's MIT license. The cultural items span the five
Brazilian macro-regions (N 105, NE 166, CO 113, SE 179, S 137) and seven Wikidata rarity
bands (`sl0-1` 153, `sl2-3` 150, `sl4-7` 146, `sl8-15` 123, `sl16-30` 78, `sl31-50` 36,
`sl51-80` 14). The control draws mostly from a few European and South American countries
(Spain, Germany, the United Kingdom, Argentina, Italy).

## The rarity-matched control

Each cultural item is paired with a non-Brazilian item of comparable rarity, linked by
`matched_br_id` on the control side (the control item's `matched_br_id` is the cultural
item's `id`). Rarity is proxied by Wikidata sitelink count and pageviews, not measured
corpus frequency, which no public tool gives for these models; residual frequency
differences are a stated limitation. Because the two groups are matched on rarity, a
difference in erosion between them reflects cultural content rather than rarity.

## Macro-region balance

The cultural items are tagged by Brazilian macro-region (N, NE, CO, SE, S). They are not
evenly spread (Southeast and Northeast dominate, Centro-Oeste is the smallest), so the
paper reports a coarse North-plus-Northeast vs Southeast-plus-South contrast rather than
per-region claims, which are underpowered at a few dozen items per region.

## Honest scope of the released probe

The probe is a Brazilian state-level cultural-knowledge MCQ probe with an item-by-item
rarity-matched generic control. It is **not** yet inter-annotator validated nor
leakage-audited against an exact-count frequency index; both are stated as future work in
the paper and are not claimed here. Wikidata gold verification reduces, but does not prove
the absence of, pretraining leakage.
