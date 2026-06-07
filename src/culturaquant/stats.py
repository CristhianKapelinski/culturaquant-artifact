"""Statistics: Wilson CIs, McNemar paired test, and the knowledge-type x
bit-width interaction coefficient.

No result is reported without a denominator and a 95% interval. The interaction
term is the headline quantity: a uniform drop across cultural and control yields
an interaction near zero (an honest null), so the design cannot manufacture an
effect from a uniform compression hit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

Z95 = 1.959963984540054


@dataclass(frozen=True)
class Proportion:
    """An accuracy with its Wilson 95% interval and denominator."""

    k: int
    n: int

    @property
    def point(self) -> float:
        return self.k / self.n if self.n else float("nan")

    def wilson(self) -> tuple[float, float]:
        n = self.n
        if n == 0:
            return (float("nan"), float("nan"))
        p = self.k / n
        z = Z95
        denom = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denom
        half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        return (max(0.0, center - half), min(1.0, center + half))


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on discordant pairs (b, c).

    b = FP16 correct & quant wrong; c = FP16 wrong & quant correct.
    Exact binomial test against p=0.5 on the b+c discordant pairs.
    """
    n = b + c
    if n == 0:
        return 1.0
    from math import comb

    def tail(x: int) -> float:
        return sum(comb(n, i) for i in range(0, x + 1)) / (2 ** n)

    k = min(b, c)
    p = 2.0 * tail(k)
    return min(1.0, p)


def paired_delta_bootstrap(
    fp16_correct: list[int],
    quant_correct: list[int],
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    """Paired per-item accuracy delta (fp16 - quant) with bootstrap 95% CI.

    Returns (point_delta, ci_lo, ci_hi) in proportion units.
    """
    import random

    assert len(fp16_correct) == len(quant_correct)
    n = len(fp16_correct)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    diffs = [fp16_correct[i] - quant_correct[i] for i in range(n)]
    point = sum(diffs) / n
    rng = random.Random(seed)
    boots = []
    for _ in range(n_boot):
        s = 0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        boots.append(s / n)
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot)]
    return point, lo, hi


def interaction_logit(
    rows: list[tuple[int, int, int]],
    n_boot: int,
    seed: int,
) -> dict:
    """Fit accuracy ~ is_cultural + is_low_precision + interaction via a small
    Newton-Raphson logistic regression, returning the interaction coefficient
    with a bootstrap 95% CI.

    rows: list of (is_cultural in {0,1}, is_low_precision in {0,1}, correct in {0,1}).
    The interaction coefficient is the additional log-odds drop that cultural
    items suffer under low precision, beyond the main effects. Negative = cultural
    items lose MORE accuracy under quantization than control on the same axis.

    The design has exactly four covariate patterns (cultural x precision), so the
    saturated four-parameter logistic fit is closed-form: the interaction equals
    the difference-in-differences of cell log-odds. We compute it directly and
    bootstrap at the item level (resampling each cell's successes binomially),
    which is exact and fast.
    """
    import random

    cells = {(0, 0): [0, 0], (0, 1): [0, 0], (1, 0): [0, 0], (1, 1): [0, 0]}
    for (c, lp, y) in rows:
        cells[(c, lp)][0] += y
        cells[(c, lp)][1] += 1

    def logodds(k, n):
        # Haldane-Anscombe 0.5 continuity correction for empty/full cells
        p = (k + 0.5) / (n + 1.0)
        return math.log(p / (1.0 - p))

    def interaction_from(cellcounts):
        # (cultural lp - cultural fp) - (control lp - control fp)
        cl = logodds(*cellcounts[(1, 1)]) - logodds(*cellcounts[(1, 0)])
        co = logodds(*cellcounts[(0, 1)]) - logodds(*cellcounts[(0, 0)])
        return cl - co

    inter = interaction_from(cells)

    rng = random.Random(seed)
    n_total = len(rows)
    boots = []
    for _ in range(n_boot):
        bc = {}
        for key, (k, n) in cells.items():
            if n == 0:
                bc[key] = [0, 0]
                continue
            p = k / n
            kk = sum(1 for _ in range(n) if rng.random() < p)
            bc[key] = [kk, n]
        boots.append(interaction_from(bc))
    boots.sort()
    lo = boots[int(0.025 * len(boots))] if boots else float("nan")
    hi = boots[int(0.975 * len(boots))] if boots else float("nan")
    return {"interaction": inter, "ci_lo": lo, "ci_hi": hi, "n": n_total}


Cells = dict[tuple[int, int], list[int]]


def _empty_cells() -> Cells:
    return {(0, 0): [0, 0], (0, 1): [0, 0], (1, 0): [0, 0], (1, 1): [0, 0]}


def _logodds(k: int, n: int) -> float:
    """Haldane-Anscombe 0.5-corrected log-odds, safe for empty/full cells."""
    p = (k + 0.5) / (n + 1.0)
    return math.log(p / (1.0 - p))


def interaction_from_cells(cells: Cells) -> float:
    """Difference-in-differences of cell log-odds = the saturated interaction.

    cells maps (is_cultural, is_low_precision) -> [successes, n]. Negative means
    cultural items lose more accuracy under low precision than control items.
    """
    cl = _logodds(*cells[(1, 1)]) - _logodds(*cells[(1, 0)])
    co = _logodds(*cells[(0, 1)]) - _logodds(*cells[(0, 0)])
    return cl - co


def cluster_bootstrap_interaction(
    per_cluster: list[list[tuple[int, int, int]]],
    n_boot: int,
    seed: int,
) -> tuple[float, float, float]:
    """Model-clustered bootstrap CI for the saturated interaction.

    per_cluster is a list (one entry per cluster, e.g. per model) of paired
    item rows (is_cultural, fp_correct, quant_correct). The point estimate pools
    every item-by-cluster row; the bootstrap resamples CLUSTERS with replacement
    and then items within each chosen cluster, so model-level non-independence is
    respected and the CI is wider than naive stacking.
    """
    import random

    pooled = _empty_cells()
    for paired in per_cluster:
        for (cu, fc, qc) in paired:
            pooled[(cu, 0)][0] += fc
            pooled[(cu, 0)][1] += 1
            pooled[(cu, 1)][0] += qc
            pooled[(cu, 1)][1] += 1
    point = interaction_from_cells(pooled)

    rng = random.Random(seed)
    k = len(per_cluster)
    if k == 0:
        return (float("nan"), float("nan"), float("nan"))
    boots = []
    for _ in range(n_boot):
        chosen = [per_cluster[rng.randrange(k)] for _ in range(k)]
        bc = _empty_cells()
        for paired in chosen:
            m = len(paired)
            for _ in range(m):
                cu, fc, qc = paired[rng.randrange(m)]
                bc[(cu, 0)][0] += fc
                bc[(cu, 0)][1] += 1
                bc[(cu, 1)][0] += qc
                bc[(cu, 1)][1] += 1
        boots.append(interaction_from_cells(bc))
    boots.sort()
    lo = boots[int(0.025 * len(boots))]
    hi = boots[int(0.975 * len(boots))]
    return point, lo, hi


def dersimonian_laird(
    per_cluster: list[list[tuple[int, int, int]]],
    n_boot: int,
    seed: int,
) -> tuple[float, float, float, float]:
    """Random-effects meta-analysis of per-cluster interaction coefficients.

    Each cluster's interaction and its bootstrap variance feed a DerSimonian-Laird
    pooled estimate. Returns (pooled, ci_lo, ci_hi, tau2).
    """
    import random

    rng = random.Random(seed + 7)
    ys: list[float] = []
    vs: list[float] = []
    for paired in per_cluster:
        cells = _empty_cells()
        for (cu, fc, qc) in paired:
            cells[(cu, 0)][0] += fc
            cells[(cu, 0)][1] += 1
            cells[(cu, 1)][0] += qc
            cells[(cu, 1)][1] += 1
        y = interaction_from_cells(cells)
        m = len(paired)
        bs = []
        for _ in range(n_boot):
            bc = _empty_cells()
            for _ in range(m):
                cu, fc, qc = paired[rng.randrange(m)]
                bc[(cu, 0)][0] += fc
                bc[(cu, 0)][1] += 1
                bc[(cu, 1)][0] += qc
                bc[(cu, 1)][1] += 1
            bs.append(interaction_from_cells(bc))
        mean = sum(bs) / len(bs)
        var = sum((x - mean) ** 2 for x in bs) / (len(bs) - 1)
        ys.append(y)
        vs.append(max(var, 1e-6))

    k = len(ys)
    if k == 0:
        return (float("nan"), float("nan"), float("nan"), 0.0)
    w = [1 / v for v in vs]
    ybar = sum(wi * yi for wi, yi in zip(w, ys)) / sum(w)
    q = sum(wi * (yi - ybar) ** 2 for wi, yi in zip(w, ys))
    c = sum(w) - sum(wi**2 for wi in w) / sum(w)
    tau2 = max(0.0, (q - (k - 1)) / c) if c > 0 else 0.0
    wstar = [1 / (v + tau2) for v in vs]
    pooled = sum(wi * yi for wi, yi in zip(wstar, ys)) / sum(wstar)
    se = math.sqrt(1 / sum(wstar))
    return pooled, pooled - Z95 * se, pooled + Z95 * se, tau2


def holm_adjust(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, in the input order."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        a = min(1.0, (m - rank) * pvals[idx])
        a = max(a, running)
        running = a
        adj[idx] = a
    return adj
