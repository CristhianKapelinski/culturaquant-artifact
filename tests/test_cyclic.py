"""The cyclic-rotation probe must place the gold answer at each of the five
positions exactly once and never drop or duplicate an option."""

from culturaquant.score_cyclic import cyclic_rotations


def test_gold_visits_every_position_once():
    options = ["a", "b", "c", "d", "e"]
    gold_positions = []
    for r, new_opts, new_gold in cyclic_rotations(options, gold_index=2):
        assert new_opts[new_gold] == options[2]      # gold value preserved
        assert sorted(new_opts) == sorted(options)   # a permutation, nothing lost
        assert new_gold == r
        gold_positions.append(new_gold)
    assert sorted(gold_positions) == [0, 1, 2, 3, 4]  # each position once


def test_rotation_preserves_cyclic_order():
    options = ["a", "b", "c", "d", "e"]
    # every rotation is a pure cyclic shift of the canonical order
    canonical = options + options
    for _, new_opts, _ in cyclic_rotations(options, gold_index=0):
        s = "".join(o for o in new_opts)
        assert s in "".join(canonical)
