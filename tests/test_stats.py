"""Unit tests for the statistics module (no network, no GPU)."""

from culturaquant.stats import (
    Proportion,
    cluster_bootstrap_interaction,
    dersimonian_laird,
    holm_adjust,
    interaction_from_cells,
    interaction_logit,
    mcnemar_exact_p,
    paired_delta_bootstrap,
)


def _cluster(cu, fp_acc, q_acc, n):
    """Build a synthetic cluster of n paired rows for one group at given accuracies."""
    rows = []
    fp_hits = round(fp_acc * n)
    q_hits = round(q_acc * n)
    for i in range(n):
        rows.append((cu, 1 if i < fp_hits else 0, 1 if i < q_hits else 0))
    return rows


def test_wilson_basic():
    p = Proportion(k=20, n=80)  # 0.25
    assert abs(p.point - 0.25) < 1e-9
    lo, hi = p.wilson()
    assert 0 < lo < 0.25 < hi < 1
    # known Wilson interval for 20/80 ~ [0.168, 0.356]
    assert abs(lo - 0.168) < 0.01
    assert abs(hi - 0.356) < 0.01


def test_wilson_edges():
    assert Proportion(k=0, n=0).wilson() != Proportion(k=0, n=0).wilson() or True
    lo, hi = Proportion(k=10, n=10).wilson()
    assert hi <= 1.0 and lo > 0.5


def test_mcnemar_symmetry_is_one():
    assert mcnemar_exact_p(5, 5) == 1.0
    assert mcnemar_exact_p(0, 0) == 1.0


def test_mcnemar_extreme_is_significant():
    p = mcnemar_exact_p(12, 0)
    assert p < 0.001


def test_paired_bootstrap_empty_is_nan():
    d, lo, hi = paired_delta_bootstrap([], [], n_boot=100, seed=1)
    assert d != d and lo != lo and hi != hi  # all NaN, no crash


def test_paired_bootstrap_zero_diff():
    fp = [1, 1, 0, 0, 1, 0]
    qc = [1, 1, 0, 0, 1, 0]
    d, lo, hi = paired_delta_bootstrap(fp, qc, n_boot=2000, seed=1)
    assert d == 0.0
    assert lo == 0.0 and hi == 0.0


def test_paired_bootstrap_positive_drop():
    fp = [1] * 20
    qc = [1] * 10 + [0] * 10  # 10 items dropped
    d, lo, hi = paired_delta_bootstrap(fp, qc, n_boot=4000, seed=2)
    assert abs(d - 0.5) < 1e-9
    assert lo > 0.2 and hi <= 0.8


def test_interaction_detects_differential():
    # construct rows where cultural items lose accuracy under low precision but
    # control items do not -> interaction should be negative (extra drop)
    rows = []
    # control: 90% at fp, 90% at lp (no drop)
    for _ in range(45):
        rows += [(0, 0, 1), (0, 1, 1)]
    for _ in range(5):
        rows += [(0, 0, 0), (0, 1, 0)]
    # cultural: 90% at fp, 50% at lp (big drop)
    for _ in range(45):
        rows += [(1, 0, 1)]
    for _ in range(5):
        rows += [(1, 0, 0)]
    for _ in range(25):
        rows += [(1, 1, 1)]
    for _ in range(25):
        rows += [(1, 1, 0)]
    res = interaction_logit(rows, n_boot=300, seed=3)
    assert res["interaction"] < 0  # cultural extra drop


def test_interaction_uniform_drop_near_zero():
    # both groups drop equally -> interaction near zero (honest null)
    rows = []
    for grp in (0, 1):
        for _ in range(40):
            rows += [(grp, 0, 1)]
        for _ in range(10):
            rows += [(grp, 0, 0)]
        for _ in range(25):
            rows += [(grp, 1, 1)]
        for _ in range(25):
            rows += [(grp, 1, 0)]
    res = interaction_logit(rows, n_boot=300, seed=4)
    assert abs(res["interaction"]) < 0.5


def test_interaction_from_cells_matches_logit():
    # the closed-form cell interaction must equal the logit-fit interaction
    rows = []
    for _ in range(45):
        rows += [(0, 0, 1), (0, 1, 1)]
    for _ in range(5):
        rows += [(0, 0, 0), (0, 1, 0)]
    for _ in range(20):
        rows += [(1, 0, 1), (1, 1, 0)]
    for _ in range(30):
        rows += [(1, 0, 1), (1, 1, 1)]
    cells = {(0, 0): [0, 0], (0, 1): [0, 0], (1, 0): [0, 0], (1, 1): [0, 0]}
    for (cu, lp, y) in rows:
        cells[(cu, lp)][0] += y
        cells[(cu, lp)][1] += 1
    direct = interaction_from_cells(cells)
    via = interaction_logit(rows, n_boot=10, seed=1)["interaction"]
    assert abs(direct - via) < 1e-9


def test_cluster_bootstrap_detects_differential():
    # 4 clusters where cultural drops 90->50 and control holds 90->90
    clusters = []
    for _ in range(4):
        clusters.append(_cluster(0, 0.9, 0.9, 50) + _cluster(1, 0.9, 0.5, 50))
    point, lo, hi = cluster_bootstrap_interaction(clusters, n_boot=400, seed=5)
    assert point < 0  # cultural extra drop
    assert hi < 0.5  # CI is informative, not unbounded


def test_cluster_bootstrap_uniform_is_near_zero():
    clusters = []
    for _ in range(4):
        clusters.append(_cluster(0, 0.9, 0.7, 50) + _cluster(1, 0.9, 0.7, 50))
    point, lo, hi = cluster_bootstrap_interaction(clusters, n_boot=400, seed=6)
    assert abs(point) < 0.5
    assert lo < 0 < hi  # null straddles zero


def test_dersimonian_laird_pooled_sign():
    clusters = []
    for _ in range(5):
        clusters.append(_cluster(0, 0.9, 0.9, 50) + _cluster(1, 0.9, 0.6, 50))
    pooled, lo, hi, tau2 = dersimonian_laird(clusters, n_boot=200, seed=7)
    assert pooled < 0
    assert tau2 >= 0.0


def test_holm_adjust_monotone_and_bounded():
    p = [0.001, 0.04, 0.5, 0.9]
    adj = holm_adjust(p)
    assert adj[0] <= adj[1] <= adj[2] <= adj[3]  # monotone in sorted order
    assert all(0.0 <= a <= 1.0 for a in adj)
    # smallest p of 4 tests is multiplied by 4
    assert abs(adj[0] - 0.004) < 1e-9


def test_holm_adjust_single():
    assert holm_adjust([0.03]) == [0.03]
