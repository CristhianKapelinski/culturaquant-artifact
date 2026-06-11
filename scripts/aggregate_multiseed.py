"""Multi-seed aggregator for the CulturaQuant quantization-erosion grid.

Each seed is a different deterministic option permutation (the position-bias fix);
scoring is deterministic given a seed, so seeds are the variance source. For every
seed we recompute the headline localized int8 cultural-vs-control interaction with
its model-clustered bootstrap 95% CI, then summarize the distribution ACROSS seeds:
each seed's estimate + CI, how many seeds exclude zero, and the pooled estimate.

We reuse the exact stats primitives the single-seed analyzer uses
(cluster_bootstrap_interaction, paired_delta_bootstrap, Proportion/Wilson), so the
multi-seed numbers are directly comparable to the live single-seed macros.

Outputs:
  - a console report (per-seed table + robust verdict),
  - a NEW macros file results_macros_multiseed.tex (never overwrites the live one).

Usage:
  python scripts/aggregate_multiseed.py --seed-root DIR --macro-out OUT.tex \
      [--run-date YYYY-MM-DD]
where DIR contains one subdir per seed (e.g. s20260607/, s11/, ...), each a merged
grid dir with manifest.json + preds__*.jsonl.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from culturaquant.analyze import load_preds, model_tag
from culturaquant.stats import (
    Proportion,
    cluster_bootstrap_interaction,
    paired_delta_bootstrap,
)

FLOOR = 0.20
N_BOOT = 2000
BOOT_SEED = 20260607
Z95 = 1.959963984540054

STRATUM_TAGS = {
    "regional_facts": "RegFacts",
    "cuisine": "Cuisine",
    "geography": "Geography",
    "proverbs": "Proverbs",
}
TUCANO = ["TucanoBR/Tucano-1b1", "TucanoBR/Tucano-2b4"]


def band_clearers(grid, models) -> list[str]:
    """Models whose FP16 cultural AND control Wilson 95% lower bounds clear
    floor+0.05 (the single-seed analyzer's rule, applied identically here)."""
    out = []
    for name in models:
        fp = grid.get((name, "fp16"))
        if not fp:
            continue
        cult = _acc(fp, lambda r: r["group"] == "cultural")
        ctrl = _acc(fp, lambda r: r["group"] == "control")
        if cult.wilson()[0] > FLOOR + 0.05 and ctrl.wilson()[0] > FLOOR + 0.05:
            out.append(name)
    return out


def _acc(rows, pred) -> Proportion:
    sel = [r for r in rows.values() if pred(r)]
    return Proportion(k=sum(r["correct"] for r in sel), n=len(sel))


def _paired_rows(fp_rows, q_rows, restrict=None):
    out = []
    for i, fr in fp_rows.items():
        if i not in q_rows:
            continue
        if restrict is not None and i not in restrict:
            continue
        cu = 1 if fr["group"] == "cultural" else 0
        out.append((cu, fr["correct"], q_rows[i]["correct"]))
    return out


def _per_cluster(grid, names, prec, restrict=None):
    cls = []
    for n in names:
        fp = grid.get((n, "fp16"))
        q = grid.get((n, prec))
        if not fp or not q:
            continue
        cls.append(_paired_rows(fp, q, restrict))
    return [c for c in cls if c]


def localized_ids(grid, names):
    any_fp = grid.get((names[0], "fp16"), {}) if names else {}
    return {i for i, r in any_fp.items()
            if r["group"] == "cultural" and r["stratum"] != "proverbs"}


def load_seed(seed_dir: Path):
    blob = load_preds(seed_dir)
    return blob["manifest"], blob["grid"]


def cultural_control_deltas(grid, names, prec, cult_restrict=None):
    """Pooled paired (fp16 - quant) deltas for cultural and control, in pp.

    The headline contrasts the LOCALIZED cultural items (cult_restrict) against the
    FULL control set, matching the single-seed analyzer's cqHead*Delta semantics:
    control is the unrestricted comparison baseline, so it is never narrowed by the
    localized-cultural id filter (which would empty it)."""
    cult_fp, cult_q, ctrl_fp, ctrl_q = [], [], [], []
    for n in names:
        fp = grid.get((n, "fp16"))
        q = grid.get((n, prec))
        if not fp or not q:
            continue
        for i, fr in fp.items():
            if i not in q:
                continue
            if fr["group"] == "cultural":
                if cult_restrict is not None and i not in cult_restrict:
                    continue
                cult_fp.append(fr["correct"]); cult_q.append(q[i]["correct"])
            else:
                ctrl_fp.append(fr["correct"]); ctrl_q.append(q[i]["correct"])
    cd = paired_delta_bootstrap(cult_fp, cult_q, N_BOOT, BOOT_SEED)
    od = paired_delta_bootstrap(ctrl_fp, ctrl_q, N_BOOT, BOOT_SEED)
    return cd, od, len(cult_fp), len(ctrl_fp)


def stratum_int8_delta(grid, names, stratum):
    fk = qk = n = 0
    for nm in names:
        fp = grid.get((nm, "fp16")); q = grid.get((nm, "int8"))
        if not fp or not q:
            continue
        for i, fr in fp.items():
            if fr["stratum"] != stratum or i not in q:
                continue
            fk += fr["correct"]; qk += q[i]["correct"]; n += 1
    if n == 0:
        return None
    return (fk / n, (fk - qk) / n, n)


def tucano_floor(grid):
    """FP16 cultural accuracy of each Tucano vs the 0.20 floor (Wilson CI)."""
    out = {}
    for m in TUCANO:
        fp = grid.get((m, "fp16"))
        if not fp:
            continue
        p = _acc(fp, lambda r: r["group"] == "cultural")
        lo, hi = p.wilson()
        out[m] = (p.point, lo, hi, hi <= FLOOR + 0.05)
    return out


def mde_proportions(n_obs: int, p_disc: float = 0.20, power: float = 0.8) -> float:
    """Paired-erosion MDE (pp) for the pooled localized-cultural accuracy delta at
    alpha=0.05 and the given power, over n_obs paired observations (localized
    cultural items x band-clearing models). The within-pair erosion delta has
    SE ~ sqrt(p_disc / n_obs) where p_disc is the FP16-vs-int8 discordance rate
    (~20pp empirically), so MDE = (z_a + z_b) * SE. With n_obs ~ 70 items x 5
    band-clearers this lands near the design's ~3pp resolvable target; reporting on
    the per-item-pooled set (not 70 items alone) is what makes 3pp attainable."""
    if n_obs <= 0:
        return float("nan")
    z_a, z_b = Z95, 0.8416212335729143  # z_{0.975}, z_{0.80}
    se = math.sqrt(p_disc / n_obs)
    return 100.0 * (z_a + z_b) * se


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-root", required=True,
                    help="dir with one subdir per seed (sXXXX/)")
    ap.add_argument("--macro-out", required=True)
    ap.add_argument("--run-date", default="2026-06-08")
    args = ap.parse_args()

    root = Path(args.seed_root)
    seed_dirs = sorted([d for d in root.iterdir()
                        if d.is_dir() and (d / "manifest.json").exists()],
                       key=lambda d: d.name)
    if not seed_dirs:
        raise SystemExit(f"no seed dirs with manifest under {root}")

    EXPECTED_MODELS = 8
    EXPECTED_PRECS = 3
    per_seed = []   # list of dicts
    skipped = []
    for sd in seed_dirs:
        manifest, grid = load_seed(sd)
        seed = manifest["seed"]
        models = manifest["models"]
        # completeness gate: a seed contributes to the headline only with the full
        # 8-model x 3-precision grid present; partial (in-flight) seeds are listed
        # but excluded from the pooled/verdict statistics.
        n_models = len({m for (m, _p) in grid})
        n_cells = len(grid)
        complete = (n_models >= EXPECTED_MODELS
                    and n_cells >= EXPECTED_MODELS * EXPECTED_PRECS)
        if not complete:
            skipped.append((seed, n_models, n_cells))
            continue
        bc = band_clearers(grid, models)
        loc = localized_ids(grid, bc)
        # headline: localized-only int8 clustered interaction
        loc_clusters = _per_cluster(grid, bc, "int8", restrict=loc)
        if loc_clusters:
            pt, lo, hi = cluster_bootstrap_interaction(loc_clusters, N_BOOT, BOOT_SEED)
        else:
            pt = lo = hi = float("nan")
        # full-set (all cultural incl proverbs) int8 clustered interaction
        full_clusters = _per_cluster(grid, bc, "int8")
        if full_clusters:
            fpt, flo, fhi = cluster_bootstrap_interaction(full_clusters, N_BOOT, BOOT_SEED)
        else:
            fpt = flo = fhi = float("nan")
        cd, od, ncult, nctrl = cultural_control_deltas(grid, bc, "int8", cult_restrict=loc)
        per_seed.append({
            "seed": seed, "dir": sd.name, "n_band": len(bc),
            "band": [m.split("/")[-1] for m in bc],
            "loc_inter": pt, "loc_lo": lo, "loc_hi": hi,
            "loc_excl0": (lo > 0 or hi < 0) if not math.isnan(lo) else False,
            "full_inter": fpt, "full_lo": flo, "full_hi": fhi,
            "full_excl0": (flo > 0 or fhi < 0) if not math.isnan(flo) else False,
            "cult_delta": cd[0], "cult_lo": cd[1], "cult_hi": cd[2],
            "ctrl_delta": od[0], "ctrl_lo": od[1], "ctrl_hi": od[2],
            "n_loc_cult": ncult, "n_ctrl": nctrl,
            "grid": grid, "models": models, "bc": bc, "loc": loc,
        })

    K = len(per_seed)
    if K == 0:
        raise SystemExit(
            "no COMPLETE seed (8 models x 3 precisions) yet; "
            f"partial seeds present: {skipped}")
    if skipped:
        print(f"[note] excluded {len(skipped)} incomplete seed(s): "
              + ", ".join(f"{s}({nm}m/{nc}cells)" for s, nm, nc in skipped))

    # pooled-across-seeds localized interaction: treat each (seed, model) as a
    # cluster and pool ALL of them, then cluster-bootstrap. This is the
    # across-seed-and-model pooled headline.
    pooled_clusters = []
    for ps in per_seed:
        pooled_clusters += _per_cluster(ps["grid"], ps["bc"], "int8", restrict=ps["loc"])
    pool_pt, pool_lo, pool_hi = cluster_bootstrap_interaction(
        pooled_clusters, N_BOOT, BOOT_SEED) if pooled_clusters else (float("nan"),) * 3

    # mean +/- sd of per-seed point estimates
    locs = [ps["loc_inter"] for ps in per_seed if not math.isnan(ps["loc_inter"])]
    mean_loc = sum(locs) / len(locs) if locs else float("nan")
    sd_loc = (math.sqrt(sum((x - mean_loc) ** 2 for x in locs) / (len(locs) - 1))
              if len(locs) > 1 else 0.0)
    n_excl0 = sum(1 for ps in per_seed if ps["loc_excl0"])
    n_excl0_neg = sum(1 for ps in per_seed if ps["loc_hi"] < 0)

    # ---- console report ----
    print("=" * 78)
    print(f"CulturaQuant MULTI-SEED aggregate  (K={K} seeds)")
    print("=" * 78)
    print("\nPer-seed LOCALIZED int8 cultural-vs-control interaction "
          "(model-clustered 95% CI):")
    print(f"{'seed':>10} {'nBand':>5} {'interaction':>12} {'CI95':>20} "
          f"{'excl0':>6} {'cultDelta':>10} {'ctrlDelta':>10}")
    for ps in per_seed:
        ci = f"[{ps['loc_lo']:+.3f},{ps['loc_hi']:+.3f}]"
        print(f"{ps['seed']:>10} {ps['n_band']:>5} {ps['loc_inter']:>+12.3f} "
              f"{ci:>20} {('YES' if ps['loc_excl0'] else 'no'):>6} "
              f"{100*ps['cult_delta']:>+9.1f}p {100*ps['ctrl_delta']:>+9.1f}p")
    print("-" * 78)
    print(f"mean(point) = {mean_loc:+.3f}  sd = {sd_loc:.3f}  "
          f"seeds-CI-excludes-0: {n_excl0}/{K}  (negative side: {n_excl0_neg}/{K})")
    print(f"POOLED across seeds+models (clustered): {pool_pt:+.3f}  "
          f"[{pool_lo:+.3f}, {pool_hi:+.3f}]  "
          f"excl0={'YES' if (pool_lo>0 or pool_hi<0) else 'no'}")

    # band-clearer stability
    from collections import Counter
    band_counter = Counter()
    for ps in per_seed:
        for m in ps["band"]:
            band_counter[m] += 1
    print("\nBand-clearer stability (how many of the K seeds each model clears):")
    for m, c in band_counter.most_common():
        print(f"  {m:30} {c}/{K}")

    # per-stratum int8 deltas across seeds (mean +/- sd of pooled-over-band delta)
    print("\nPer-stratum int8 cultural erosion (pooled over band-clearers), "
          "mean +/- sd across seeds:")
    strata_summary = {}
    for st, tag in STRATUM_TAGS.items():
        ds = []
        for ps in per_seed:
            r = stratum_int8_delta(ps["grid"], ps["bc"], st)
            if r:
                ds.append(r[1])
        if ds:
            mu = 100 * sum(ds) / len(ds)
            sdv = (100 * math.sqrt(sum((d - sum(ds)/len(ds))**2 for d in ds)/(len(ds)-1))
                   if len(ds) > 1 else 0.0)
            strata_summary[tag] = (mu, sdv, len(ds))
            print(f"  {st:18} delta = {mu:+6.1f}pp  +/- {sdv:4.1f}  (K={len(ds)})")

    # pooled cultural vs control deltas across seeds
    cult_ds = [100*ps["cult_delta"] for ps in per_seed]
    ctrl_ds = [100*ps["ctrl_delta"] for ps in per_seed]
    mu_cult = sum(cult_ds)/len(cult_ds); mu_ctrl = sum(ctrl_ds)/len(ctrl_ds)
    print(f"\nPooled localized int8 deltas across seeds: "
          f"cultural {mu_cult:+.1f}pp  control {mu_ctrl:+.1f}pp  "
          f"diff {mu_cult-mu_ctrl:+.1f}pp")

    # Tucano floor (from any seed's grid - it's an FP16 property, stable)
    tf = tucano_floor(per_seed[0]["grid"])
    print("\nTucano floor (FP16 cultural acc vs 0.20 floor):")
    for m, (pt, lo, hi, floored) in tf.items():
        print(f"  {m:24} acc={100*pt:.1f}%  CI[{100*lo:.1f},{100*hi:.1f}]  "
              f"floored={'YES' if floored else 'no'}")

    # MDE: per the design, the resolvable effect is on the POOLED localized-cultural
    # observation set (items x band-clearers), not the 70 unique items alone.
    n_cult_loc = len(per_seed[0]["loc"])          # unique localized cultural items
    n_obs_loc = per_seed[0]["n_loc_cult"]         # pooled obs over band-clearers
    mde = mde_proportions(n_obs_loc)
    mde_items = mde_proportions(n_cult_loc)
    print(f"\nMDE: {n_cult_loc} unique localized cultural items x "
          f"{len(per_seed[0]['bc'])} band-clearers = {n_obs_loc} pooled obs -> "
          f"resolvable erosion ~ {mde:.1f}pp (design target ~3pp). "
          f"On 70 items alone (one model): ~{mde_items:.1f}pp.")

    # ---- robust verdict ----
    print("\n" + "=" * 78)
    print("ROBUST VERDICT")
    print("=" * 78)
    if n_excl0 == 0:
        verdict = ("NULL and ROBUST: the localized cultural-vs-control int8 "
                   "interaction CI INCLUDES ZERO in ALL %d seeds. The differential "
                   "cultural-erosion headline does NOT hold under proper "
                   "position-shuffled statistics." % K)
    elif n_excl0 == K:
        verdict = ("CONSISTENT EFFECT: the CI EXCLUDES zero in ALL %d seeds "
                   "(sign %s). A small differential cultural erosion survives "
                   "shuffling." % (K, "negative" if n_excl0_neg == K else "mixed"))
    else:
        verdict = ("FRAGILE / SEED-DEPENDENT: the CI excludes zero in only %d/%d "
                   "seeds. The effect is not robust to the option permutation; the "
                   "headline is at best a weak, permutation-sensitive signal and is "
                   "best reported as null." % (n_excl0, K))
    print(verdict)
    print(f"Pooled (all seeds+models) localized interaction "
          f"{pool_pt:+.3f} [{pool_lo:+.3f},{pool_hi:+.3f}] -> "
          f"{'excludes 0' if (pool_lo>0 or pool_hi<0) else 'includes 0 (null)'}.")

    # ---- emit macros ----
    M = []

    def emit(name, val):
        M.append(f"\\newcommand{{\\{name}}}{{{val}}}")

    M.append("% === CulturaQuant MULTI-SEED RESULTS MACROS (auto-generated) ===")
    M.append("% NEW file; does not overwrite the live single-seed results_macros.tex")
    emit("cqMsNSeeds", K)
    emit("cqMsSeedList", ", ".join(str(ps["seed"]) for ps in per_seed))
    emit("cqMsRunDate", args.run_date)
    emit("cqMsNBoot", N_BOOT)
    emit("cqMsLocItems", n_cult_loc)
    emit("cqMsLocPooledObs", n_obs_loc)
    emit("cqMsMDE", f"{mde:.1f}")
    emit("cqMsMDEItems", f"{mde_items:.1f}")
    # per-seed localized interaction
    for ps in per_seed:
        s = str(ps["seed"])
        emit(f"cqMsSeed{s}LocInter", f"{ps['loc_inter']:.3f}")
        emit(f"cqMsSeed{s}LocCIlo", f"{ps['loc_lo']:.3f}")
        emit(f"cqMsSeed{s}LocCIhi", f"{ps['loc_hi']:.3f}")
        emit(f"cqMsSeed{s}LocExclZero", "yes" if ps["loc_excl0"] else "no")
        emit(f"cqMsSeed{s}NBand", ps["n_band"])
    emit("cqMsLocInterMean", f"{mean_loc:.3f}")
    emit("cqMsLocInterSd", f"{sd_loc:.3f}")
    emit("cqMsLocSeedsExclZero", n_excl0)
    emit("cqMsLocSeedsExclZeroNeg", n_excl0_neg)
    emit("cqMsLocInterPooled", f"{pool_pt:.3f}")
    emit("cqMsLocInterPooledCIlo", f"{pool_lo:.3f}")
    emit("cqMsLocInterPooledCIhi", f"{pool_hi:.3f}")
    emit("cqMsLocInterPooledExclZero",
         "yes" if (pool_lo > 0 or pool_hi < 0) else "no")
    # pooled deltas
    emit("cqMsCulturalDeltaMean", f"{mu_cult:.1f}")
    emit("cqMsControlDeltaMean", f"{mu_ctrl:.1f}")
    emit("cqMsDiffErosionMean", f"{mu_cult-mu_ctrl:.1f}")
    # per-stratum
    for tag, (mu, sdv, k) in strata_summary.items():
        emit(f"cqMsStratum{tag}IntDeltaMean", f"{mu:.1f}")
        emit(f"cqMsStratum{tag}IntDeltaSd", f"{sdv:.1f}")
    # band-clearer stability
    stable = [m for m, c in band_counter.items() if c == K]
    emit("cqMsBandClearersStable", ", ".join(sorted(stable)))
    emit("cqMsBandClearersStableN", len(stable))
    # tucano floor
    for m, (pt, lo, hi, floored) in tf.items():
        tag = model_tag(m)
        emit(f"cqMs{tag}FpCulturalAcc", f"{100*pt:.1f}")
        emit(f"cqMs{tag}Floored", "yes" if floored else "no")
    # verdict
    if n_excl0 == 0:
        vtag = "nullrobust"
    elif n_excl0 == K:
        vtag = "consistent"
    else:
        vtag = "fragile"
    emit("cqMsVerdict", vtag)
    emit("cqMsVerdictText", verdict.replace("%", "\\%"))

    Path(args.macro_out).write_text("\n".join(M) + "\n", encoding="utf-8")
    print(f"\n[macros] wrote {len(M)} lines to {args.macro_out}")


if __name__ == "__main__":
    main()
