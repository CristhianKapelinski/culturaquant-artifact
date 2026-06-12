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


def load_items(path: Path, group: str) -> list[Item]:
    """Load a JSONL item file into the harmonized schema.

    ``group`` is "cultural" or "control". Each row carries the question (``input``),
    the five ``alternatives``, and the gold ``correct_index``; ``stratum`` and
    ``region`` are optional (control rows default to stratum "generic", region "-").
    """
    rows = _read_jsonl(path)
    return [
        Item(
            id=r["id"],
            group=group,
            stratum=r.get("stratum", "generic" if group == "control" else "cultural"),
            region=r.get("region", "-"),
            question=r["input"],
            options=tuple(r["alternatives"]),
            gold_index=int(r["correct_index"]),
        )
        for r in rows
    ]
