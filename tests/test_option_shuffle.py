"""Tests for the per-item option permutation that de-biases gate scoring.

The author-written cultural/control items ship with the correct answer pinned at
index 0. Scoring them in that canonical order makes the gold answer always "A",
so any positional bias toward the first option (notably NF4's collapse toward
"A") inflates accuracy. ``option_permutation`` shuffles the options
deterministically per item so the gold position is balanced; these tests pin its
invariants and the balance property that the fix relies on.
"""

import collections

from culturaquant.scorer import apply_permutation, option_permutation


def test_permutation_is_a_valid_permutation():
    perm = option_permutation("reg-005", 5, seed=20260607)
    assert sorted(perm) == [0, 1, 2, 3, 4]


def test_permutation_is_deterministic():
    a = option_permutation("reg-005", 5, seed=20260607)
    b = option_permutation("reg-005", 5, seed=20260607)
    assert a == b


def test_apply_permutation_remaps_gold():
    options = ["correct", "b", "c", "d", "e"]
    perm = option_permutation("cui-001", 5, seed=20260607)
    new_opts, new_gold = apply_permutation(options, 0, perm)
    # gold text follows its option through the permutation
    assert new_opts[new_gold] == "correct"
    # and every option is preserved
    assert sorted(new_opts) == sorted(options)


def test_gold_position_is_balanced_not_pinned_to_zero():
    """Across the canonical (gold=0) item set the permuted gold must spread over
    all 5 positions, not stay at index 0 (the bug). We require gold to land at
    index 0 on no more than ~40% of items, i.e. far below the 100% of the bug."""
    ids = [f"reg-{i:03d}" for i in range(70)] + [f"ctl-{i:03d}" for i in range(50)]
    golds = []
    for iid in ids:
        perm = option_permutation(iid, 5, seed=20260607)
        _, g = apply_permutation(["x"] * 5, 0, perm)
        golds.append(g)
    dist = collections.Counter(golds)
    assert set(dist) == {0, 1, 2, 3, 4}, dist
    assert dist[0] / len(ids) < 0.40, dist
