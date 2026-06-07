"""Dataset loading and harmonization for CulturaQuant.

All item sets are reduced to a single internal schema so the scorer is agnostic
to the source. A harmonized item is:
    {id, group, stratum, region, question, options[5], gold_index}
where ``group`` is one of {"cultural", "control"} and ``stratum`` tags the
knowledge type (proverbs, regional_facts, cuisine, geography, generic).

BRoverbs is pulled from the HuggingFace Hub at runtime (it is NOT redistributed
inside this repo); only its stratum/region tag and item id are author-owned. The
cultural strata and the generic control are author-written and shipped as JSONL.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

REGIONS = ("N", "NE", "CO", "SE", "S")
GROUPS = ("cultural", "control")


@dataclass(frozen=True)
class Item:
    """A single auto-gradable 5-way MCQ item."""

    id: str
    group: str
    stratum: str
    region: str
    question: str
    options: tuple[str, ...]
    gold_index: int

    def as_dict(self) -> dict:
        d = asdict(self)
        d["options"] = list(self.options)
        return d


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_cultural_strata(data_dir: Path) -> list[Item]:
    rows = _read_jsonl(data_dir / "cultural_strata.jsonl")
    return [
        Item(
            id=r["id"],
            group="cultural",
            stratum=r["stratum"],
            region=r["region"],
            question=r["input"],
            options=tuple(r["alternatives"]),
            gold_index=int(r["correct_index"]),
        )
        for r in rows
    ]


def load_control(data_dir: Path) -> list[Item]:
    rows = _read_jsonl(data_dir / "control_generic.jsonl")
    return [
        Item(
            id=r["id"],
            group="control",
            stratum="generic",
            region="-",
            question=r["input"],
            options=tuple(r["alternatives"]),
            gold_index=int(r["correct_index"]),
        )
        for r in rows
    ]


def load_broverbs_proverbs(max_items: int, seed: int) -> list[Item]:
    """Load the BRoverbs proverb stratum from the HuggingFace Hub.

    We use the ``proverb_to_history`` split (given a proverb, pick the matching
    short story among 5 options) as the proverbs cultural stratum, subsampled
    deterministically. Items are tagged region="-" because proverbs are not
    macro-region localized in BRoverbs.
    """
    from datasets import load_dataset
    import random

    ds = load_dataset("Tropic-AI/BRoverbs", split="proverb_to_history")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    idx = idx[:max_items]
    items: list[Item] = []
    for i in idx:
        row = ds[i]
        items.append(
            Item(
                id=f"prov-{i:04d}",
                group="cultural",
                stratum="proverbs",
                region="-",
                question=row["input"],
                options=tuple(row["alternatives"]),
                gold_index=int(row["correct_index"]),
            )
        )
    return items
