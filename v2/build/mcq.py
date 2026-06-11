"""5-option MCQ assembly with sibling distractors and balanced gold position.

A distractor pool is the set of sibling entities of the same type as the gold
(other Brazilian states for cultural items; other first-level admin divisions of the
same country for control items). Distractors are guaranteed wrong-for-this-subject
because the gold is the unique KB object of the relation, and we exclude it from the
pool. Gold position is assigned round-robin A..E across the batch so it is balanced by
construction (the design's cyclic-permutation eval controls position fully at score
time; the stored correct_index is only the canonical layout).
"""

from __future__ import annotations

import random

LETTERS = ["A", "B", "C", "D", "E"]


def build_mcq(
    gold_label: str,
    gold_qid: str,
    pool: list[tuple[str, str]],
    forced_index: int,
    rng: random.Random,
) -> dict | None:
    """Assemble one 5-option MCQ.

    pool: list of (qid, label) sibling candidates, MUST exclude the gold.
    forced_index: target position (0..4) for the gold, for batch balancing.
    Returns dict with alternatives, alt_qids, correct_index, or None if <4 distractors.
    """
    distract = [(q, lb) for q, lb in pool if q != gold_qid and lb != gold_label]
    # Deduplicate by label (sibling pools can carry alias duplicates).
    seen: set[str] = set()
    uniq: list[tuple[str, str]] = []
    for q, lb in distract:
        if lb in seen:
            continue
        seen.add(lb)
        uniq.append((q, lb))
    if len(uniq) < 4:
        return None
    chosen = rng.sample(uniq, 4)

    alts: list[str] = [lb for _, lb in chosen]
    qids: list[str] = [q for q, _ in chosen]
    idx = forced_index % 5
    alts.insert(idx, gold_label)
    qids.insert(idx, gold_qid)
    return {
        "alternatives": alts,
        "alt_qids": qids,
        "correct_index": idx,
        "distractor_qids": [q for q, _ in chosen],
    }
