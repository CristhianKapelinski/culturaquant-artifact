"""QDIS gate math: cross-quantization-level disagreement signals and budgeted
threshold fitting. Pure functions, no model and no GPU.

The gate decides *when to retrieve* on a quantized edge device from two cheap
quant passes (int8 and NF4) over the existing constrained-likelihood MCQ scorer.
Each item contributes two length-normalized option log-prob vectors; from those
we derive:

  - FLIP    : hard disagreement between the int8 and NF4 argmax (dominant trigger).
  - JSD     : Jensen-Shannon divergence between the two softmax posteriors over the
              5 options, base e, in [0, ln 2] (soft disagreement).
  - MARGIN  : min over precisions of (top1 - top2) log-prob margin. This is the
              generic-uncertainty (TARG) signal; it is LOGGED but excluded from the
              primary QDIS trigger, and reused only to build the generic baseline.

The QDIS rule retrieves iff ``FLIP or JSD >= tau_jsd``. ``tau_jsd`` is fit by
5-fold cross-fitting to hit a target retrieval budget B, so no item's gate
decision is calibrated on itself and the budget is honest.

The generic (TARG) baseline retrieves iff the int8 margin < ``tau_m``, with
``tau_m`` set to the same target budget. Both gates are thus always compared at a
matched retrieval fraction.
"""

from __future__ import annotations

import math

LN2 = math.log(2.0)


def _softmax(logprobs: list[float]) -> list[float]:
    """Stable softmax over a vector of (length-normalized) log-probs."""
    m = max(logprobs)
    exps = [math.exp(x - m) for x in logprobs]
    s = sum(exps)
    return [e / s for e in exps]


def argmax(logprobs: list[float]) -> int:
    """Index of the maximum, ties broken by lowest index (deterministic)."""
    best_i, best_v = 0, -float("inf")
    for i, v in enumerate(logprobs):
        if v > best_v:
            best_v, best_i = v, i
    return best_i


def flip(l_int8: list[float], l_nf4: list[float]) -> bool:
    """True iff the int8 and NF4 argmax options disagree (hard disagreement)."""
    return argmax(l_int8) != argmax(l_nf4)


def _kl(p: list[float], q: list[float]) -> float:
    s = 0.0
    for pi, qi in zip(p, q):
        if pi > 0.0:
            s += pi * math.log(pi / qi)
    return s


def jsd(p_int8: list[float], p_nf4: list[float]) -> float:
    """Jensen-Shannon divergence (base e, in [0, ln 2]) between the option
    posteriors of the two quant levels.

    The inputs are the per-option log-prob vectors; they are softmaxed here, so
    callers pass the raw ``score_item_logprobs`` output directly.
    """
    p = _softmax(p_int8)
    q = _softmax(p_nf4)
    m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]
    val = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    # numerical guard: clamp into the analytic range
    return min(max(val, 0.0), LN2)


def _top1_top2_margin(logprobs: list[float]) -> float:
    s = sorted(logprobs, reverse=True)
    return s[0] - s[1]


def margin_min(l_int8: list[float], l_nf4: list[float]) -> float:
    """min over precisions of the top1-top2 log-prob margin (joint low-margin
    term). Small => both quant levels are jointly unsure. Logged, not in the
    primary QDIS rule."""
    return min(_top1_top2_margin(l_int8), _top1_top2_margin(l_nf4))


def qdis_components(l_int8: list[float], l_nf4: list[float]) -> dict:
    """All gate signals for one item from its two quant-level log-prob vectors."""
    return {
        "flip": int(flip(l_int8, l_nf4)),
        "jsd": jsd(l_int8, l_nf4),
        "margin_min": margin_min(l_int8, l_nf4),
        "margin_int8": _top1_top2_margin(l_int8),
        "argmax_int8": argmax(l_int8),
        "argmax_nf4": argmax(l_nf4),
    }


def _quantile_tau_jsd(flips: list[int], jsds: list[float], budget: float) -> float:
    """Threshold on JSD so that (FLIP + JSD-triggered) retrieval fraction ~= budget.

    FLIP items always retrieve. Among non-FLIP items, take the JSD value at the
    quantile that adds just enough items to reach the target fraction. Returns a
    ``tau`` such that retrieving iff ``flip or jsd >= tau`` hits the budget on
    this calibration set. If FLIP alone already meets/exceeds the budget, tau is
    set above the max JSD (retrieve on FLIP only).
    """
    n = len(flips)
    if n == 0:
        return float("inf")
    n_flip = sum(flips)
    target = budget * n
    extra = target - n_flip
    if extra <= 0:
        return float("inf")  # FLIP already saturates the budget
    non_flip_jsd = sorted(
        (j for f, j in zip(flips, jsds) if not f), reverse=True
    )
    if not non_flip_jsd:
        return float("inf")
    k = int(round(extra))
    k = max(0, min(k, len(non_flip_jsd)))
    if k == 0:
        return float("inf")
    # threshold at the k-th largest non-FLIP JSD; >= tau triggers exactly k items
    return non_flip_jsd[k - 1]


def _quantile_tau_margin(margins: list[float], budget: float) -> float:
    """Threshold on the int8 margin so that ``margin < tau_m`` fires on ~budget
    fraction of items (the generic/TARG gate). Smaller margin = more uncertain =
    retrieve."""
    n = len(margins)
    if n == 0:
        return -float("inf")
    target = int(round(budget * n))
    if target <= 0:
        return -float("inf")
    asc = sorted(margins)
    target = min(target, n)
    # retrieve the `target` smallest margins: tau is just above the target-th value
    if target >= n:
        return float("inf")
    return asc[target - 1] + 1e-12


def _folds(n: int, k: int, seed: int) -> list[list[int]]:
    """Deterministic k-fold partition of indices 0..n-1."""
    import random

    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    return [idx[i::k] for i in range(k)]


def fit_tau_jsd(
    items: list[dict], budget: float, folds: int = 5, seed: int = 20260607
) -> dict:
    """5-fold cross-fitted QDIS gate decisions at a target ``budget``.

    ``items`` is a list of dicts each carrying at least ``flip`` (0/1) and ``jsd``
    (float), in a fixed order. For each held-out fold, ``tau_jsd`` is fit on the
    other folds via :func:`_quantile_tau_jsd` and applied to the held-out items,
    so no item is calibrated on itself.

    Returns ``{"retrieve": [bool], "tau_per_fold": [float], "budget": B}`` where
    ``retrieve[i]`` is the gate decision for item i. The realized retrieval
    fraction approximates ``budget`` (exact only in expectation because tau is
    cross-fitted).
    """
    n = len(items)
    retrieve = [False] * n
    if n == 0:
        return {"retrieve": retrieve, "tau_per_fold": [], "budget": budget}
    k = min(folds, n)
    parts = _folds(n, k, seed)
    taus: list[float] = []
    for held in parts:
        held_set = set(held)
        train = [i for i in range(n) if i not in held_set]
        tr_flips = [int(items[i]["flip"]) for i in train]
        tr_jsds = [float(items[i]["jsd"]) for i in train]
        tau = _quantile_tau_jsd(tr_flips, tr_jsds, budget)
        taus.append(tau)
        for i in held:
            f = int(items[i]["flip"])
            j = float(items[i]["jsd"])
            retrieve[i] = bool(f) or (j >= tau)
    return {"retrieve": retrieve, "tau_per_fold": taus, "budget": budget}


def fit_tau_margin(
    items: list[dict], budget: float, folds: int = 5, seed: int = 20260607
) -> dict:
    """5-fold cross-fitted generic (TARG) margin gate at target ``budget``.

    ``items`` carry ``margin_int8``. For each held-out fold, ``tau_m`` is fit on
    the rest so ``margin_int8 < tau_m`` fires on ~budget of train items, then
    applied to the held-out fold. This is the baseline gate, cross-fitted on the
    SAME folds as QDIS so the matched-budget comparison is apples-to-apples.
    """
    n = len(items)
    retrieve = [False] * n
    if n == 0:
        return {"retrieve": retrieve, "tau_per_fold": [], "budget": budget}
    k = min(folds, n)
    parts = _folds(n, k, seed)
    taus: list[float] = []
    for held in parts:
        held_set = set(held)
        train = [i for i in range(n) if i not in held_set]
        tr_margins = [float(items[i]["margin_int8"]) for i in train]
        tau = _quantile_tau_margin(tr_margins, budget)
        taus.append(tau)
        for i in held:
            retrieve[i] = float(items[i]["margin_int8"]) < tau
    return {"retrieve": retrieve, "tau_per_fold": taus, "budget": budget}


def gate_decisions(
    items: list[dict], gate_kind: str, budget: float, folds: int = 5, seed: int = 20260607
) -> list[bool]:
    """Cross-fitted retrieve-or-not decisions for a gate at a target budget.

    ``gate_kind`` is ``"quant_aware"`` (QDIS: FLIP or JSD>=tau) or ``"generic"``
    (TARG margin: int8 margin < tau_m). Returns a boolean mask aligned with
    ``items``.
    """
    if gate_kind == "quant_aware":
        return fit_tau_jsd(items, budget, folds, seed)["retrieve"]
    if gate_kind == "generic":
        return fit_tau_margin(items, budget, folds, seed)["retrieve"]
    raise ValueError(f"unknown gate_kind {gate_kind!r}")


def oracle_decisions(closed_correct: list[int], rag_correct: list[int]) -> list[bool]:
    """Oracle ceiling: retrieve iff closed-book is WRONG and always-RAG is RIGHT.
    Maximum recoverable accuracy at minimum budget; costs nothing extra."""
    return [
        (cb == 0 and rag == 1) for cb, rag in zip(closed_correct, rag_correct)
    ]


def apply_gate(
    closed_correct: list[int], rag_correct: list[int], retrieve: list[bool]
) -> list[int]:
    """Per-item correctness under a gate: use the RAG answer where the gate fired,
    otherwise keep the closed-book answer."""
    return [
        (rag_correct[i] if retrieve[i] else closed_correct[i])
        for i in range(len(retrieve))
    ]
