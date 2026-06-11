"""Unit tests for the QDIS gate math (no network, no GPU).

FLIP/JSD/MARGIN are checked on hand-computed synthetic log-prob vectors; the
cross-fitting is checked to hit the target retrieval budget within tolerance.
"""

import math

from culturaquant.gate import (
    apply_gate,
    fit_tau_jsd,
    fit_tau_margin,
    flip,
    gate_decisions,
    jsd,
    margin_min,
    oracle_decisions,
    qdis_components,
)


def test_flip_hard_disagreement():
    # int8 picks option 0, nf4 picks option 1 -> flip
    assert flip([2.0, 1.0, 0.0, 0.0, 0.0], [1.0, 2.0, 0.0, 0.0, 0.0]) is True
    # same argmax -> no flip
    assert flip([2.0, 1.0, 0.0, 0.0, 0.0], [3.0, 1.0, 0.0, 0.0, 0.0]) is False


def test_jsd_identical_is_zero():
    v = [1.0, 0.5, 0.0, -0.5, -1.0]
    assert abs(jsd(v, v)) < 1e-12


def test_jsd_bounded_by_ln2():
    # near-disjoint posteriors approach the ln2 ceiling but never exceed it
    a = [50.0, -50.0, -50.0, -50.0, -50.0]
    b = [-50.0, 50.0, -50.0, -50.0, -50.0]
    j = jsd(a, b)
    assert 0.0 <= j <= math.log(2.0) + 1e-12
    assert j > 0.6  # close to ln2 ~ 0.693


def test_jsd_known_two_point_value():
    # collapse to a 2-option Bernoulli: int8 = (1,0,..), nf4 = (0,1,..)
    # both one-hot in different positions -> JS divergence = ln 2
    a = [100.0, -100.0, -100.0, -100.0, -100.0]
    b = [-100.0, 100.0, -100.0, -100.0, -100.0]
    assert abs(jsd(a, b) - math.log(2.0)) < 1e-3


def test_margin_min_is_min_over_precisions():
    # int8 margin = 2.0 (3-1), nf4 margin = 0.5 (1-0.5) -> min = 0.5
    l_int8 = [3.0, 1.0, 0.0, 0.0, 0.0]
    l_nf4 = [1.0, 0.5, 0.0, 0.0, 0.0]
    assert abs(margin_min(l_int8, l_nf4) - 0.5) < 1e-12


def test_components_bundle():
    c = qdis_components([2.0, 1.0, 0, 0, 0], [1.0, 2.0, 0, 0, 0])
    assert c["flip"] == 1
    assert c["argmax_int8"] == 0 and c["argmax_nf4"] == 1
    assert c["jsd"] > 0


def _synthetic_items(n_flip: int, n_total: int) -> list[dict]:
    """n_flip flips (always retrieve) + the rest non-flip with descending JSD so
    a quantile cut is well defined."""
    items = []
    for i in range(n_flip):
        items.append({"flip": 1, "jsd": 0.0, "margin_int8": 0.01 * i})
    for i in range(n_total - n_flip):
        # non-flip JSDs spread over (0, 0.5); margins spread over (0, 5)
        items.append({"flip": 0, "jsd": 0.5 * (1 - i / (n_total - n_flip)),
                      "margin_int8": 5.0 * (i / (n_total - n_flip))})
    return items


def test_cross_fitting_hits_budget_qdis():
    items = _synthetic_items(n_flip=4, n_total=100)
    for budget in (0.10, 0.20, 0.30, 0.40, 0.50):
        res = fit_tau_jsd(items, budget, folds=5, seed=1)
        realized = sum(res["retrieve"]) / len(items)
        # cross-fitting cannot be exact; allow a small tolerance
        assert abs(realized - budget) <= 0.06, (budget, realized)


def test_cross_fitting_flip_floor():
    # if FLIP rate already exceeds the budget, everyone flipping still retrieves
    items = [{"flip": 1, "jsd": 0.0, "margin_int8": 0.0} for _ in range(50)]
    items += [{"flip": 0, "jsd": 0.1, "margin_int8": 1.0} for _ in range(50)]
    res = fit_tau_jsd(items, 0.10, folds=5, seed=2)
    # all 50 flips must retrieve regardless of budget
    assert sum(res["retrieve"]) >= 50


def test_cross_fitting_hits_budget_margin():
    items = _synthetic_items(n_flip=0, n_total=100)
    for budget in (0.10, 0.30, 0.50):
        res = fit_tau_margin(items, budget, folds=5, seed=3)
        realized = sum(res["retrieve"]) / len(items)
        assert abs(realized - budget) <= 0.06, (budget, realized)


def test_gate_decisions_dispatch():
    items = _synthetic_items(n_flip=2, n_total=40)
    q = gate_decisions(items, "quant_aware", 0.3, seed=4)
    g = gate_decisions(items, "generic", 0.3, seed=4)
    assert len(q) == len(g) == 40
    assert isinstance(q[0], bool)


def test_oracle_and_apply_gate():
    closed = [0, 1, 0, 1]
    rag = [1, 1, 0, 0]
    oracle = oracle_decisions(closed, rag)
    # item 0: closed wrong, rag right -> retrieve; item 3: closed right, rag wrong -> no
    assert oracle == [True, False, False, False]
    applied = apply_gate(closed, rag, oracle)
    # oracle recovers item 0, keeps the rest closed-book
    assert applied == [1, 1, 0, 1]
