"""Round-2 diagnostic: print the per-test McNemar/Holm family table, the gate
sensitivity, and the per-model interaction coefficients for a prediction grid.

This is a human-readable cross-check; every macro the paper uses is emitted by
``culturaquant.analyze`` (the canonical, tested path), which reuses the SAME
statistics functions imported here. No statistics are reimplemented in this file.

Usage:
    python scripts/compute_round2_clustered.py --out-dir data/results/grid
"""

from __future__ import annotations

import argparse
from pathlib import Path

from culturaquant.analyze import load_preds, model_tag
from culturaquant.stats import (
    cluster_bootstrap_interaction,
    dersimonian_laird,
    holm_adjust,
    interaction_from_cells,
    mcnemar_exact_p,
)

N_BOOT = 2000
SEED = 20260607
BAND_CLEARERS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
]


def _paired(fp_rows: dict, q_rows: dict, restrict=None):
    rows = []
    for i, fr in fp_rows.items():
        if i not in q_rows or (restrict is not None and i not in restrict):
            continue
        cu = 1 if fr["group"] == "cultural" else 0
        rows.append((cu, fr["correct"], q_rows[i]["correct"]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data/results/grid")
    args = ap.parse_args()
    grid = load_preds(Path(args.out_dir))["grid"]

    print("=== HEADLINE CLUSTERED INTERACTION (model-clustered bootstrap) ===")
    for prec in ("int8", "nf4"):
        clusters = [
            _paired(grid[(m, "fp16")], grid[(m, prec)])
            for m in BAND_CLEARERS
            if (m, "fp16") in grid and (m, prec) in grid
        ]
        pt, lo, hi = cluster_bootstrap_interaction(clusters, N_BOOT, SEED)
        dl = dersimonian_laird(clusters, N_BOOT, SEED)
        print(f"  {prec}: interaction={pt:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
        print(f"    DL random-effects: {dl[0]:.3f} [{dl[1]:.3f}, {dl[2]:.3f}] tau2={dl[3]:.3f}")

    print("\n=== LOCALIZED-ONLY (freshly-authored items) int8 ===")
    sample = grid[(BAND_CLEARERS[0], "fp16")]
    loc = {i for i, r in sample.items()
           if r["group"] == "cultural" and r["stratum"] != "proverbs"}
    loc_clusters = [
        _paired(grid[(m, "fp16")], grid[(m, "int8")], restrict=loc)
        for m in BAND_CLEARERS
        if (m, "int8") in grid
    ]
    pt, lo, hi = cluster_bootstrap_interaction(loc_clusters, N_BOOT, SEED)
    print(f"  n localized ids = {len(loc)}")
    print(f"  int8 localized clustered interaction={pt:.3f} [{lo:.3f}, {hi:.3f}]")

    print("\n=== PER-MODEL int8 interaction ===")
    for m in BAND_CLEARERS:
        if (m, "int8") not in grid:
            continue
        cells = {(0, 0): [0, 0], (0, 1): [0, 0], (1, 0): [0, 0], (1, 1): [0, 0]}
        for (cu, fc, qc) in _paired(grid[(m, "fp16")], grid[(m, "int8")]):
            cells[(cu, 0)][0] += fc
            cells[(cu, 0)][1] += 1
            cells[(cu, 1)][0] += qc
            cells[(cu, 1)][1] += 1
        print(f"  {m.split('/')[-1]}: {interaction_from_cells(cells):.3f}")

    print("\n=== McNEMAR FAMILY + HOLM CORRECTION ===")
    labels, pvals, bc = [], [], []
    for m in BAND_CLEARERS:
        for prec in ("int8", "nf4"):
            if (m, "int8") not in grid or (m, prec) not in grid:
                continue
            fp_rows, q_rows = grid[(m, "fp16")], grid[(m, prec)]
            for grp in ("cultural", "control"):
                b = c = 0
                for i, fr in fp_rows.items():
                    if fr["group"] != grp or i not in q_rows:
                        continue
                    qc = q_rows[i]["correct"]
                    b += fr["correct"] == 1 and qc == 0
                    c += fr["correct"] == 0 and qc == 1
                labels.append(f"{model_tag(m)}/{prec}/{grp}")
                pvals.append(mcnemar_exact_p(b, c))
                bc.append((b, c))
    adj = holm_adjust(pvals)
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    for i in order:
        star = "  <-- survives Holm" if adj[i] < 0.05 else ""
        b, c = bc[i]
        print(f"  {labels[i]:46s} p={pvals[i]:.4f}  p_holm={adj[i]:.4f}{star} (b={b},c={c})")


if __name__ == "__main__":
    main()
