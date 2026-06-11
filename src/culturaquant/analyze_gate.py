"""Aggregate the QDIS gate JSONLs into the EXP_A metrics and emit LaTeX macros.

Implements spec section 5: the accuracy-vs-budget frontier per gate; cultural pp
recovered; the cultural x condition interaction (reusing ``stats.py``); per-stratum
recovery; gate precision-at-budget vs the oracle; and the pre-registered
matched-budget QDIS-vs-margin cultural test with a model-clustered bootstrap
(tie reportable). The macro block follows ``analyze.emit_macros`` style but is
written under the artifact's own ``data/results/`` tree, never to the paper dir.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyze import _emit, _pct
from .run_gate import BUDGETS, GATES
from .stats import cluster_bootstrap_interaction, mcnemar_exact_p, paired_delta_bootstrap

N_BOOT = 2000
BOOT_SEED = 20260607
RECOVERY_TARGET = 0.80  # B* = smallest budget recovering >=80% of the RAG gain
STRATUM_TAGS = {"regional_facts": "RegFacts", "cuisine": "Cuisine", "geography": "Geography"}
BUDGET_TAGS = {f"{b:.2f}": f"B{int(round(b * 100)):02d}" for b in BUDGETS}


def load_gate_runs(out_dir: Path) -> dict[str, dict]:
    """Return {model: blob} for every gate__*.json in out_dir."""
    runs: dict[str, dict] = {}
    for path in sorted(out_dir.glob("gate__*.json")):
        blob = json.loads(path.read_text())
        runs[blob["manifest"]["model"]] = blob
    return runs


def _cultural_mask(records: list[dict]) -> list[bool]:
    return [r["group"] == "cultural" for r in records]


def _sub(values: list, mask: list[bool]) -> list:
    return [v for v, m in zip(values, mask) if m]


def _acc(correct: list[int]) -> float:
    return sum(correct) / len(correct) if correct else float("nan")


def _condition_correct(blob: dict, condition: str, gate: str, budget: str) -> list[int]:
    cond = blob["conditions"]
    if condition == "closed_book":
        return cond["closed_book"]
    if condition == "always_rag":
        return cond["always_rag"]
    if condition == "gate":
        return cond["gates"][gate][budget]["correct"]
    raise ValueError(condition)


def _gate_gain_recovered(closed_acc: float, rag_acc: float, gate_acc: float) -> float:
    gap = rag_acc - closed_acc
    if gap <= 0:
        return float("nan")
    return (gate_acc - closed_acc) / gap


def _find_bstar(runs: dict, models: list[str]) -> str:
    """Smallest budget at which EITHER gate recovers >=80% of the pooled cultural
    always-RAG gain (the pre-registered matched-budget B*)."""
    for b in (f"{x:.2f}" for x in BUDGETS):
        for gate in GATES:
            closed, rag, gated = [], [], []
            for m in models:
                blob = runs[m]
                mask = _cultural_mask(blob["records"])
                closed += _sub(blob["conditions"]["closed_book"], mask)
                rag += _sub(blob["conditions"]["always_rag"], mask)
                gated += _sub(blob["conditions"]["gates"][gate][b]["correct"], mask)
            rec = _gate_gain_recovered(_acc(closed), _acc(rag), _acc(gated))
            if rec == rec and rec >= RECOVERY_TARGET:
                return b
    return f"{BUDGETS[-1]:.2f}"  # fall back to the largest budget


def _gate_precision(blob: dict, gate: str, budget: str, cultural_only: bool) -> tuple[int, int]:
    """(worth-retrieving hits, retrieved count) for gate precision-at-budget.

    A retrieved item is 'worth it' iff closed-book wrong AND always-RAG right
    (the oracle definition).
    """
    cond = blob["conditions"]
    closed = cond["closed_book"]
    rag = cond["always_rag"]
    retrieve = cond["gates"][gate][budget]["retrieve"]
    mask = _cultural_mask(blob["records"])
    hits = retr = 0
    for i in range(len(retrieve)):
        if cultural_only and not mask[i]:
            continue
        if retrieve[i]:
            retr += 1
            if closed[i] == 0 and rag[i] == 1:
                hits += 1
    return hits, retr


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate the QDIS gate runs.")
    ap.add_argument("--out-dir", required=True, help="dir with gate__*.json")
    ap.add_argument("--macro-out", required=True,
                    help="results_gate_macros.tex path (under data/results/)")
    ap.add_argument("--run-date", default="2026-06-08")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    runs = load_gate_runs(out_dir)
    models = sorted(runs.keys())

    macros: list[str] = []
    macros.append("% === CulturaQuant GATE (EXP_A) RESULTS MACROS (auto-generated) ===")
    macros.append("% Regenerate: culturaquant-run-gate ... && culturaquant-analyze-gate ...")
    _emit(macros, "cqgRunDate", args.run_date)
    _emit(macros, "cqgNModels", len(models))
    _emit(macros, "cqgNBoot", N_BOOT)
    _emit(macros, "cqgRecoveryTarget", f"{RECOVERY_TARGET:.2f}")

    if not models:
        Path(args.macro_out).write_text("\n".join(macros) + "\n", encoding="utf-8")
        print("[gate-macros] no runs found; wrote header only")
        return

    # --- closed-book / always-RAG cultural references (pooled over models) ---
    pooled_closed_cult, pooled_rag_cult = [], []
    pooled_closed_ctrl, pooled_rag_ctrl = [], []
    for m in models:
        blob = runs[m]
        mask = _cultural_mask(blob["records"])
        cmask = [not x for x in mask]
        pooled_closed_cult += _sub(blob["conditions"]["closed_book"], mask)
        pooled_rag_cult += _sub(blob["conditions"]["always_rag"], mask)
        pooled_closed_ctrl += _sub(blob["conditions"]["closed_book"], cmask)
        pooled_rag_ctrl += _sub(blob["conditions"]["always_rag"], cmask)
    cb_cult = _acc(pooled_closed_cult)
    rag_cult = _acc(pooled_rag_cult)
    _emit(macros, "cqgClosedBookCulturalAcc", _pct(cb_cult))
    _emit(macros, "cqgAlwaysRagCulturalAcc", _pct(rag_cult))
    _emit(macros, "cqgCulturalGapPP", _pct(rag_cult - cb_cult))
    _emit(macros, "cqgClosedBookControlAcc", _pct(_acc(pooled_closed_ctrl)))
    _emit(macros, "cqgAlwaysRagControlAcc", _pct(_acc(pooled_rag_ctrl)))
    _emit(macros, "cqgCulturalN", len(pooled_closed_cult))
    _emit(macros, "cqgControlN", len(pooled_closed_ctrl))

    # --- oracle cultural ceiling + budget ---
    pooled_oracle_cult, pooled_oracle_budget = [], []
    for m in models:
        blob = runs[m]
        mask = _cultural_mask(blob["records"])
        oracle = blob["conditions"]["oracle_gate"]
        closed = blob["conditions"]["closed_book"]
        rag = blob["conditions"]["always_rag"]
        for i in range(len(oracle)):
            if not mask[i]:
                continue
            pooled_oracle_cult.append(rag[i] if oracle[i] else closed[i])
            pooled_oracle_budget.append(oracle[i])
    _emit(macros, "cqgOracleCulturalAcc", _pct(_acc(pooled_oracle_cult)))
    _emit(macros, "cqgOracleCulturalBudget",
          _pct(sum(pooled_oracle_budget) / max(1, len(pooled_oracle_budget))))

    # --- frontier: per gate per budget pooled cultural accuracy + realized budget ---
    macros.append("% --- frontier: cultural accuracy vs budget, per gate ---")
    for gate in GATES:
        gtag = "Qdis" if gate == "quant_aware" else "Margin"
        for b in (f"{x:.2f}" for x in BUDGETS):
            btag = BUDGET_TAGS[b]
            cult, realized = [], []
            for m in models:
                blob = runs[m]
                mask = _cultural_mask(blob["records"])
                cult += _sub(blob["conditions"]["gates"][gate][b]["correct"], mask)
                # realized budget on cultural items
                rmask = _sub(blob["conditions"]["gates"][gate][b]["retrieve"], mask)
                realized += rmask
            acc = _acc(cult)
            rec = _gate_gain_recovered(cb_cult, rag_cult, acc)
            _emit(macros, f"cqgFrontier{gtag}{btag}Acc", _pct(acc))
            _emit(macros, f"cqgFrontier{gtag}{btag}Recovered",
                  _pct(rec) if rec == rec else "nan")
            _emit(macros, f"cqgFrontier{gtag}{btag}Budget",
                  _pct(sum(realized) / max(1, len(realized))))

    # --- B* and the pre-registered matched-budget QDIS-vs-margin cultural test ---
    macros.append("% --- pre-registered matched-budget QDIS vs margin (cultural) ---")
    bstar = _find_bstar(runs, models)
    _emit(macros, "cqgBStar", _pct(float(bstar)))

    qdis_cult, margin_cult = [], []
    per_cluster_q: list[list] = []
    for m in models:
        blob = runs[m]
        mask = _cultural_mask(blob["records"])
        q = _sub(blob["conditions"]["gates"]["quant_aware"][bstar]["correct"], mask)
        g = _sub(blob["conditions"]["gates"]["generic"][bstar]["correct"], mask)
        qdis_cult += q
        margin_cult += g
        # cluster row format reused by cluster_bootstrap_interaction:
        # (is_cultural=1, "fp_correct"=margin, "quant_correct"=qdis); the
        # interaction-of-cells reduces here to the paired (qdis - margin) contrast
        # on a single group, which we instead compute via paired bootstrap below.
        per_cluster_q.append([(1, g[i], q[i]) for i in range(len(q))])

    # per-item paired cultural delta (QDIS - margin), pooled
    d, dlo, dhi = paired_delta_bootstrap(qdis_cult, margin_cult, N_BOOT, BOOT_SEED)
    _emit(macros, "cqgQdisVsMarginDelta", _pct(d))
    _emit(macros, "cqgQdisVsMarginDeltaCIlo", _pct(dlo))
    _emit(macros, "cqgQdisVsMarginDeltaCIhi", _pct(dhi))
    # model-clustered bootstrap of the same paired delta
    clo, chi = _clustered_paired_delta(per_cluster_q, N_BOOT, BOOT_SEED)
    _emit(macros, "cqgQdisVsMarginClCIlo", _pct(clo))
    _emit(macros, "cqgQdisVsMarginClCIhi", _pct(chi))
    if clo > 0:
        outcome = "qdis_wins"
    elif chi < 0:
        outcome = "margin_wins"
    else:
        outcome = "tie"
    _emit(macros, "cqgQdisVsMarginOutcome", outcome)
    b = sum(1 for i in range(len(qdis_cult)) if qdis_cult[i] and not margin_cult[i])
    c = sum(1 for i in range(len(qdis_cult)) if not qdis_cult[i] and margin_cult[i])
    _emit(macros, "cqgQdisVsMarginMcP", f"{mcnemar_exact_p(b, c):.4f}")

    # --- cultural x condition interaction (gate selectivity), at B* ---
    macros.append("% --- cultural x (closed-book vs gated) interaction at B* ---")
    for gate in GATES:
        gtag = "Qdis" if gate == "quant_aware" else "Margin"
        clusters = []
        for m in models:
            blob = runs[m]
            closed = blob["conditions"]["closed_book"]
            gated = blob["conditions"]["gates"][gate][bstar]["correct"]
            rows = []
            for i, r in enumerate(blob["records"]):
                cu = 1 if r["group"] == "cultural" else 0
                rows.append((cu, closed[i], gated[i]))
            clusters.append(rows)
        point, ilo, ihi = cluster_bootstrap_interaction(clusters, N_BOOT, BOOT_SEED)
        _emit(macros, f"cqgInteraction{gtag}", f"{point:.3f}")
        _emit(macros, f"cqgInteraction{gtag}ClCIlo", f"{ilo:.3f}")
        _emit(macros, f"cqgInteraction{gtag}ClCIhi", f"{ihi:.3f}")

    # --- per-stratum cultural recovery at B* (QDIS gate) ---
    macros.append("% --- per-stratum cultural recovery at B* (QDIS) ---")
    for st, tag in STRATUM_TAGS.items():
        closed, rag, gated = [], [], []
        for m in models:
            blob = runs[m]
            for i, r in enumerate(blob["records"]):
                if r["stratum"] != st:
                    continue
                closed.append(blob["conditions"]["closed_book"][i])
                rag.append(blob["conditions"]["always_rag"][i])
                gated.append(
                    blob["conditions"]["gates"]["quant_aware"][bstar]["correct"][i]
                )
        if not closed:
            continue
        _emit(macros, f"cqgStratum{tag}ClosedAcc", _pct(_acc(closed)))
        _emit(macros, f"cqgStratum{tag}RagAcc", _pct(_acc(rag)))
        _emit(macros, f"cqgStratum{tag}QdisAcc", _pct(_acc(gated)))

    # --- gate precision-at-budget vs oracle (cultural), at B* ---
    macros.append("% --- gate precision-at-budget vs oracle (cultural) at B* ---")
    for gate in GATES:
        gtag = "Qdis" if gate == "quant_aware" else "Margin"
        hits = retr = 0
        for m in models:
            h, n = _gate_precision(runs[m], gate, bstar, cultural_only=True)
            hits += h
            retr += n
        prec = hits / retr if retr else float("nan")
        _emit(macros, f"cqgPrecision{gtag}", _pct(prec) if prec == prec else "nan")
        _emit(macros, f"cqgPrecision{gtag}Retrieved", retr)

    # --- retrieval hit diagnostic: non-empty passage rate (cultural) ---
    nonempty = total = 0
    for m in models:
        for r in runs[m]["records"]:
            if r["group"] != "cultural":
                continue
            total += 1
            if r["retrieved"]:
                nonempty += 1
        break  # retrieval is model-independent; one model suffices
    _emit(macros, "cqgRetrievalHitRate", _pct(nonempty / total) if total else "nan")

    Path(args.macro_out).write_text("\n".join(macros) + "\n", encoding="utf-8")
    print(f"[gate-macros] wrote {len(macros)} lines to {args.macro_out}; B*={bstar}")


def _clustered_paired_delta(
    per_cluster: list[list], n_boot: int, seed: int
) -> tuple[float, float]:
    """Model-clustered bootstrap CI of the paired (qdis - margin) cultural delta.

    Each cluster is a list of (cu, margin_correct, qdis_correct) rows. Resample
    clusters with replacement, then items within, and recompute the pooled delta.
    """
    import random

    rng = random.Random(seed)
    k = len(per_cluster)
    if k == 0:
        return (float("nan"), float("nan"))
    boots = []
    for _ in range(n_boot):
        chosen = [per_cluster[rng.randrange(k)] for _ in range(k)]
        s = nn = 0
        for cl in chosen:
            m = len(cl)
            for _ in range(m):
                _, margin_c, qdis_c = cl[rng.randrange(m)]
                s += qdis_c - margin_c
                nn += 1
        boots.append(s / nn if nn else 0.0)
    boots.sort()
    return boots[int(0.025 * len(boots))], boots[int(0.975 * len(boots))]


if __name__ == "__main__":
    main()
