"""Analyze the prediction grid and emit the LaTeX results-macros block.

Computes, on IDENTICAL items across precisions:
  - PILOT: per-stratum FP16 accuracy with Wilson 95% CI vs the 0.20 random floor.
  - E1: per-model cultural-vs-control erosion deltas under int8 and nf4, the
        McNemar paired test, the paired bootstrap delta CI, and the
        knowledge-type x bit-width interaction coefficient with bootstrap CI.
  - E2: how the quantization-induced (cultural - control) erosion increase scales
        with log parameter count, using each model's own FP16 gap as the
        subtracted reference.

Every number is written into a \\newcommand macro. The prose never hardcodes.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from .stats import (
    Proportion,
    cluster_bootstrap_interaction,
    dersimonian_laird,
    holm_adjust,
    interaction_logit,
    mcnemar_exact_p,
    paired_delta_bootstrap,
)

FLOOR = 0.20
N_BOOT = 2000
BOOT_SEED = 20260607

# approximate parameter counts (billions) for the size axis
PARAM_B = {
    "Qwen/Qwen2.5-0.5B-Instruct": 0.5,
    "Qwen/Qwen2.5-1.5B-Instruct": 1.5,
    "Qwen/Qwen2.5-3B-Instruct": 3.0,
    "Qwen/Qwen3-0.6B": 0.6,
    "Qwen/Qwen3-1.7B": 1.7,
    "Qwen/Qwen3-4B": 4.0,
    "TucanoBR/Tucano-1b1": 1.1,
    "TucanoBR/Tucano-2b4": 2.4,
}

NUM_WORDS = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four",
             "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}


def _spell(s: str) -> str:
    return "".join(NUM_WORDS.get(c, c) for c in s)


def model_tag(name: str) -> str:
    base = name.split("/")[-1]
    base = re.sub(r"[^A-Za-z0-9]", "", base)
    return _spell(base)


def prec_tag(p: str) -> str:
    return {"fp16": "Fp", "int8": "Int", "nf4": "Nf"}[p]


def load_preds(out_dir: Path) -> dict:
    """Return {(model, precision): {id: row}} and the manifest."""
    manifest = json.loads((out_dir / "manifest.json").read_text())
    grid: dict[tuple[str, str], dict[str, dict]] = {}
    for run in manifest["runs"]:
        key = (run["model"], run["precision"])
        rows = {}
        with (out_dir / run["preds_file"]).open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                rows[r["id"]] = r
        grid[key] = rows
    return {"manifest": manifest, "grid": grid}


def _acc(rows: dict, pred_filter) -> Proportion:
    sel = [r for r in rows.values() if pred_filter(r)]
    k = sum(r["correct"] for r in sel)
    return Proportion(k=k, n=len(sel))


def _texsafe(s: str) -> str:
    """Escape LaTeX-special characters in a free-text macro value."""
    repl = {"_": "\\_", "&": "\\&", "%": "\\%", "#": "\\#", "$": "\\$",
            "{": "\\{", "}": "\\}", "~": "\\textasciitilde{}",
            "^": "\\textasciicircum{}"}
    return "".join(repl.get(c, c) for c in s)


def _emit(macros: list[str], name: str, value) -> None:
    if isinstance(value, str):
        value = _texsafe(value)
    macros.append(f"\\newcommand{{\\{name}}}{{{value}}}")


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--macro-out", required=True)
    ap.add_argument("--run-date", default="2026-06-07")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    blob = load_preds(out_dir)
    manifest = blob["manifest"]
    grid = blob["grid"]

    macros: list[str] = []
    macros.append("% === CulturaQuant RESULTS MACROS (auto-generated; do not edit by hand) ===")
    macros.append("% Regenerate: culturaquant-run ... && culturaquant-analyze ...")

    # --- provenance ---
    ref = manifest["reference_machine"]
    macros.append("% --- provenance ---")
    _emit(macros, "cqRefGPU", f"{ref.get('gpu','?')}, {ref.get('gpu_mem_gib','?')} GiB")
    _emit(macros, "cqRefCUDA", ref.get("cuda", "?"))
    _emit(macros, "cqRefTorch", ref.get("torch", "?"))
    _emit(macros, "cqRefRAM", ref.get("ram_gib", "?"))
    _emit(macros, "cqRefCPUcores", ref.get("cpu_count", "?"))
    _emit(macros, "cqRefOS", ref.get("platform", "?"))
    _emit(macros, "cqSeed", manifest["seed"])
    _emit(macros, "cqRunDate", args.run_date)
    _emit(macros, "cqNItems", manifest["n_items"])
    _emit(macros, "cqNBoot", N_BOOT)
    strata = manifest["item_strata"]
    _emit(macros, "cqNCultural", strata["by_group"].get("cultural", 0))
    _emit(macros, "cqNControl", strata["by_group"].get("control", 0))
    for st, c in strata["by_stratum"].items():
        _emit(macros, f"cqNStratum{_spell(st.title().replace('_',''))}", c)
    for rg, c in strata["by_region"].items():
        tag = "Unlocalized" if rg == "-" else rg
        _emit(macros, f"cqNRegion{tag}", c)
    _emit(macros, "cqFloor", f"{FLOOR:.2f}")

    models = manifest["models"]

    # --- PILOT: per-stratum FP16 accuracy with Wilson CI ---
    macros.append("% --- PILOT: per-stratum FP16 accuracy (Wilson 95% CI) vs 0.20 floor ---")
    # aggregate FP16 across the reference model set (use the largest Qwen2.5 as the
    # pilot reference; also emit per-model below)
    pilot_strata = ["proverbs", "regional_facts", "cuisine", "geography", "generic"]
    for name in models:
        key = (name, "fp16")
        if key not in grid:
            continue
        rows = grid[key]
        mt = model_tag(name)
        for st in pilot_strata:
            p = _acc(rows, lambda r, st=st: r["stratum"] == st)
            if p.n == 0:
                continue
            lo, hi = p.wilson()
            stag = _spell(st.title().replace("_", ""))
            _emit(macros, f"cqPilot{mt}{stag}Acc", _pct(p.point))
            _emit(macros, f"cqPilot{mt}{stag}CIlo", _pct(lo))
            _emit(macros, f"cqPilot{mt}{stag}CIhi", _pct(hi))
            _emit(macros, f"cqPilot{mt}{stag}N", p.n)
            # floored flag: CI upper bound at or below a small margin over floor
            floored = "yes" if hi <= FLOOR + 0.05 else "no"
            _emit(macros, f"cqPilot{mt}{stag}Floored", floored)
        # cultural-vs-control FP16 gap (the static gap E2 subtracts) and the
        # Wilson 95% lower bounds the PILOT gate keys on.
        cult = _acc(rows, lambda r: r["group"] == "cultural")
        ctrl = _acc(rows, lambda r: r["group"] == "control")
        _emit(macros, f"cqFp{mt}CulturalAcc", _pct(cult.point))
        _emit(macros, f"cqFp{mt}ControlAcc", _pct(ctrl.point))
        _emit(macros, f"cqFp{mt}CulturalN", cult.n)
        _emit(macros, f"cqFp{mt}ControlN", ctrl.n)
        _emit(macros, f"cqFp{mt}CulturalLo", _pct(cult.wilson()[0]))
        _emit(macros, f"cqFp{mt}ControlLo", _pct(ctrl.wilson()[0]))
        _emit(macros, f"cqFp{mt}StaticGap", _pct(ctrl.point - cult.point))

    # --- E1: differential erosion + interaction ---
    macros.append("% --- E1: cultural-vs-control erosion, McNemar, interaction ---")
    for name in models:
        fp_key = (name, "fp16")
        if fp_key not in grid:
            continue
        mt = model_tag(name)
        for prec in ("int8", "nf4"):
            q_key = (name, prec)
            if q_key not in grid:
                continue
            pt = prec_tag(prec)
            fp_rows, q_rows = grid[fp_key], grid[q_key]
            ids = [i for i in fp_rows if i in q_rows]
            for grp in ("cultural", "control"):
                gids = [i for i in ids if fp_rows[i]["group"] == grp]
                fpc = [fp_rows[i]["correct"] for i in gids]
                qc = [q_rows[i]["correct"] for i in gids]
                b = sum(1 for i in gids if fp_rows[i]["correct"] and not q_rows[i]["correct"])
                c = sum(1 for i in gids if not fp_rows[i]["correct"] and q_rows[i]["correct"])
                pval = mcnemar_exact_p(b, c)
                d, dlo, dhi = paired_delta_bootstrap(fpc, qc, N_BOOT, BOOT_SEED)
                gtag = "Cultural" if grp == "cultural" else "Control"
                fp_acc = sum(fpc) / len(fpc)
                q_acc = sum(qc) / len(qc)
                _emit(macros, f"cqEOne{mt}{pt}{gtag}FpAcc", _pct(fp_acc))
                _emit(macros, f"cqEOne{mt}{pt}{gtag}QuantAcc", _pct(q_acc))
                _emit(macros, f"cqEOne{mt}{pt}{gtag}Delta", _pct(d))
                _emit(macros, f"cqEOne{mt}{pt}{gtag}DeltaCIlo", _pct(dlo))
                _emit(macros, f"cqEOne{mt}{pt}{gtag}DeltaCIhi", _pct(dhi))
                _emit(macros, f"cqEOne{mt}{pt}{gtag}N", len(gids))
                _emit(macros, f"cqEOne{mt}{pt}{gtag}McB", b)
                _emit(macros, f"cqEOne{mt}{pt}{gtag}McC", c)
                _emit(macros, f"cqEOne{mt}{pt}{gtag}McP", f"{pval:.4f}")
            # interaction term: cultural extra log-odds drop under this precision
            inter_rows = []
            for i in ids:
                isc = 1 if fp_rows[i]["group"] == "cultural" else 0
                inter_rows.append((isc, 0, fp_rows[i]["correct"]))
                inter_rows.append((isc, 1, q_rows[i]["correct"]))
            res = interaction_logit(inter_rows, N_BOOT, BOOT_SEED)
            _emit(macros, f"cqEOne{mt}{pt}Interaction", f"{res['interaction']:.3f}")
            _emit(macros, f"cqEOne{mt}{pt}InteractionCIlo", f"{res['ci_lo']:.3f}")
            _emit(macros, f"cqEOne{mt}{pt}InteractionCIhi", f"{res['ci_hi']:.3f}")
            sig = "yes" if (res["ci_lo"] > 0 or res["ci_hi"] < 0) else "no"
            _emit(macros, f"cqEOne{mt}{pt}InteractionSig", sig)
            # differential erosion = control delta minus cultural delta (pp)
            # (positive = cultural erodes more than control)
            cult_ids = [i for i in ids if fp_rows[i]["group"] == "cultural"]
            ctrl_ids = [i for i in ids if fp_rows[i]["group"] == "control"]
            cult_d = (sum(fp_rows[i]["correct"] - q_rows[i]["correct"] for i in cult_ids)
                      / max(1, len(cult_ids)))
            ctrl_d = (sum(fp_rows[i]["correct"] - q_rows[i]["correct"] for i in ctrl_ids)
                      / max(1, len(ctrl_ids)))
            _emit(macros, f"cqEOne{mt}{pt}DiffErosion", _pct(cult_d - ctrl_d))

    # --- E2: size scaling of the quantization-induced gap increase ---
    macros.append("% --- E2: size scaling of quantization-induced (cultural-control) increase ---")
    # build per-(family, precision) series of (log paramB, quant-induced increase)
    families = {
        "QwenTwoFive": ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct",
                        "Qwen/Qwen2.5-3B-Instruct"],
        "QwenThree": ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"],
        "Tucano": ["TucanoBR/Tucano-1b1", "TucanoBR/Tucano-2b4"],
    }
    for fam, fam_models in families.items():
        present = [m for m in fam_models if (m, "fp16") in grid]
        if len(present) < 2:
            continue
        for prec in ("int8", "nf4"):
            pt = prec_tag(prec)
            xs, ys = [], []
            for m in present:
                fp_rows = grid[(m, "fp16")]
                if (m, prec) not in grid:
                    continue
                q_rows = grid[(m, prec)]
                ids = [i for i in fp_rows if i in q_rows]
                cult_ids = [i for i in ids if fp_rows[i]["group"] == "cultural"]
                ctrl_ids = [i for i in ids if fp_rows[i]["group"] == "control"]
                # quant-induced increase = (control delta - cultural delta) under quant
                cult_d = (sum(fp_rows[i]["correct"] - q_rows[i]["correct"] for i in cult_ids)
                          / max(1, len(cult_ids)))
                ctrl_d = (sum(fp_rows[i]["correct"] - q_rows[i]["correct"] for i in ctrl_ids)
                          / max(1, len(ctrl_ids)))
                increase = cult_d - ctrl_d
                xs.append(math.log(PARAM_B[m]))
                ys.append(increase)
                mt = model_tag(m)
                _emit(macros, f"cqETwo{mt}{pt}Increase", _pct(increase))
            if len(xs) >= 2:
                slope = _ols_slope(xs, ys)
                _emit(macros, f"cqETwo{fam}{pt}Slope", _pct(slope))
                _emit(macros, f"cqETwo{fam}{pt}Npts", len(xs))

    # --- HEADLINE: pooled differential erosion over band-clearing models ---
    # A model clears the band if BOTH its FP16 cultural and control accuracy CIs
    # sit clearly above the floor (point > floor + 0.10 on both groups). Floored
    # models (e.g. the smallest Qwen2.5 and the from-scratch native-PT Tucanos)
    # are demoted to sanity anchors and excluded from the headline, per the PILOT.
    macros.append("% --- HEADLINE: pooled differential erosion (band-clearing models only) ---")
    # A model clears the band only if BOTH its FP16 cultural and control Wilson
    # 95% lower bounds sit clearly above the 0.20 random floor (lo > floor + 0.05).
    # This excludes models whose control or cultural accuracy is statistically
    # indistinguishable from random at FP16 (a floored, non-measurable task),
    # which would be scoring a non-detector by erosion. Such models are demoted to
    # sanity anchors (the from-scratch native-PT Tucanos and the smallest Qwen2.5).
    band_clearers = []
    for name in models:
        fp_rows = grid.get((name, "fp16"))
        if not fp_rows:
            continue
        cult = _acc(fp_rows, lambda r: r["group"] == "cultural")
        ctrl = _acc(fp_rows, lambda r: r["group"] == "control")
        cult_lo = cult.wilson()[0]
        ctrl_lo = ctrl.wilson()[0]
        if cult_lo > FLOOR + 0.05 and ctrl_lo > FLOOR + 0.05:
            band_clearers.append(name)
    _emit(macros, "cqHeadNBandClearers", len(band_clearers))
    _emit(macros, "cqHeadBandClearers",
          ", ".join(n.split("/")[-1] for n in band_clearers))
    for prec in ("int8", "nf4"):
        # skip a precision that no band-clearer was run at (e.g. quick grids)
        if not any((name, prec) in grid for name in band_clearers):
            continue
        pt = prec_tag(prec)
        cult_fp, cult_q, ctrl_fp, ctrl_q = [], [], [], []
        inter_rows = []
        for name in band_clearers:
            fp_rows = grid[(name, "fp16")]
            q_rows = grid.get((name, prec))
            if not q_rows:
                continue
            ids = [i for i in fp_rows if i in q_rows]
            for i in ids:
                isc = fp_rows[i]["group"] == "cultural"
                inter_rows.append((1 if isc else 0, 0, fp_rows[i]["correct"]))
                inter_rows.append((1 if isc else 0, 1, q_rows[i]["correct"]))
                if isc:
                    cult_fp.append(fp_rows[i]["correct"])
                    cult_q.append(q_rows[i]["correct"])
                else:
                    ctrl_fp.append(fp_rows[i]["correct"])
                    ctrl_q.append(q_rows[i]["correct"])
        cd, cdlo, cdhi = paired_delta_bootstrap(cult_fp, cult_q, N_BOOT, BOOT_SEED)
        od, odlo, odhi = paired_delta_bootstrap(ctrl_fp, ctrl_q, N_BOOT, BOOT_SEED)
        _emit(macros, f"cqHead{pt}CulturalDelta", _pct(cd))
        _emit(macros, f"cqHead{pt}CulturalDeltaCIlo", _pct(cdlo))
        _emit(macros, f"cqHead{pt}CulturalDeltaCIhi", _pct(cdhi))
        _emit(macros, f"cqHead{pt}CulturalN", len(cult_fp))
        _emit(macros, f"cqHead{pt}ControlDelta", _pct(od))
        _emit(macros, f"cqHead{pt}ControlDeltaCIlo", _pct(odlo))
        _emit(macros, f"cqHead{pt}ControlDeltaCIhi", _pct(odhi))
        _emit(macros, f"cqHead{pt}ControlN", len(ctrl_fp))
        _emit(macros, f"cqHead{pt}DiffErosion", _pct(cd - od))
        res = interaction_logit(inter_rows, N_BOOT, BOOT_SEED)
        _emit(macros, f"cqHead{pt}Interaction", f"{res['interaction']:.3f}")
        _emit(macros, f"cqHead{pt}InteractionCIlo", f"{res['ci_lo']:.3f}")
        _emit(macros, f"cqHead{pt}InteractionCIhi", f"{res['ci_hi']:.3f}")
        sig = "yes" if (res["ci_lo"] > 0 or res["ci_hi"] < 0) else "no"
        _emit(macros, f"cqHead{pt}InteractionSig", sig)

    # --- E3: coarse macro-region erosion (N+NE vs SE+S), band-clearers pooled ---
    macros.append("% --- E3: coarse macro-region erosion (N+NE vs SE+S) ---")
    region_groups = {"NNE": ("N", "NE"), "SES": ("SE", "S")}
    for prec in ("int8", "nf4"):
        pt = prec_tag(prec)
        for rg_tag, regs in region_groups.items():
            fp_c, q_c = [], []
            for name in band_clearers:
                fp_rows = grid[(name, "fp16")]
                q_rows = grid.get((name, prec))
                if not q_rows:
                    continue
                ids = [i for i in fp_rows
                       if i in q_rows and fp_rows[i]["region"] in regs
                       and fp_rows[i]["group"] == "cultural"]
                for i in ids:
                    fp_c.append(fp_rows[i]["correct"])
                    q_c.append(q_rows[i]["correct"])
            if not fp_c:
                continue
            d, dlo, dhi = paired_delta_bootstrap(fp_c, q_c, N_BOOT, BOOT_SEED)
            _emit(macros, f"cqEThree{pt}{rg_tag}Delta", _pct(d))
            _emit(macros, f"cqEThree{pt}{rg_tag}DeltaCIlo", _pct(dlo))
            _emit(macros, f"cqEThree{pt}{rg_tag}DeltaCIhi", _pct(dhi))
            _emit(macros, f"cqEThree{pt}{rg_tag}N", len(fp_c))

    _emit_round2(macros, grid, band_clearers)
    _emit_wallclock(macros, manifest, band_clearers)

    Path(args.macro_out).write_text("\n".join(macros) + "\n", encoding="utf-8")
    print(f"[macros] wrote {len(macros)} lines to {args.macro_out}")


GATE_LO = 0.25
CONTROL_CEILING = 0.95
STRATUM_TAGS = {
    "regional_facts": "RegFacts",
    "cuisine": "Cuisine",
    "geography": "Geography",
    "proverbs": "Proverbs",
}


def _paired_rows(fp_rows: dict, q_rows: dict, restrict=None):
    """Paired (is_cultural, fp_correct, quant_correct) rows on shared ids."""
    out = []
    for i, fr in fp_rows.items():
        if i not in q_rows:
            continue
        if restrict is not None and i not in restrict:
            continue
        cu = 1 if fr["group"] == "cultural" else 0
        out.append((cu, fr["correct"], q_rows[i]["correct"]))
    return out


def _per_cluster(grid, band_clearers, prec, restrict=None):
    clusters = []
    for name in band_clearers:
        fp_rows = grid.get((name, "fp16"))
        q_rows = grid.get((name, prec))
        if not fp_rows or not q_rows:
            continue
        clusters.append(_paired_rows(fp_rows, q_rows, restrict))
    return clusters


def _emit_round2(macros: list[str], grid, band_clearers) -> None:
    """Round-2 robustness macros: model-clustered and random-effects interaction
    CIs, the localized-only (freshly authored) interaction, per-stratum int8
    erosion, the Holm-corrected McNemar family, the gate, proverbs framing, the
    control-ceiling count, and the band-clearer wall-clock. Every value is derived
    from the committed per-item predictions, never transcribed.
    """
    macros.append("% --- ROUND 2: clustered/random-effects CIs, localized, strata, Holm ---")
    _emit(macros, "cqGateLo", f"{GATE_LO:.2f}")
    _emit(macros, "cqHolmNTests", len(band_clearers) * 4)

    # localized = freshly authored cultural items (non-proverb cultural)
    any_fp = grid.get((band_clearers[0], "fp16"), {}) if band_clearers else {}
    loc_ids = {i for i, r in any_fp.items()
               if r["group"] == "cultural" and r["stratum"] != "proverbs"}

    for prec in ("int8", "nf4"):
        pt = prec_tag(prec)
        clusters = _per_cluster(grid, band_clearers, prec)
        if not clusters:
            continue
        _, clo, chi = cluster_bootstrap_interaction(clusters, N_BOOT, BOOT_SEED)
        _emit(macros, f"cqHead{pt}InteractionClCIlo", f"{clo:.3f}")
        _emit(macros, f"cqHead{pt}InteractionClCIhi", f"{chi:.3f}")
        re_coef, re_lo, re_hi, _tau = dersimonian_laird(clusters, N_BOOT, BOOT_SEED)
        _emit(macros, f"cqHead{pt}InteractionReCoef", f"{re_coef:.3f}")
        _emit(macros, f"cqHead{pt}InteractionReCIlo", f"{re_lo:.3f}")
        _emit(macros, f"cqHead{pt}InteractionReCIhi", f"{re_hi:.3f}")
        cult_obs = sum(1 for cl in clusters for r in cl if r[0] == 1)
        ctrl_obs = sum(1 for cl in clusters for r in cl if r[0] == 0)
        _emit(macros, f"cqHead{pt}CulturalObs", cult_obs)
        _emit(macros, f"cqHead{pt}ControlObs", ctrl_obs)

    # localized-only int8 clustered interaction (the headline localized number)
    loc_clusters = _per_cluster(grid, band_clearers, "int8", restrict=loc_ids)
    if loc_clusters:
        lpt, llo, lhi = cluster_bootstrap_interaction(loc_clusters, N_BOOT, BOOT_SEED)
        _emit(macros, "cqLocIntInteraction", f"{lpt:.3f}")
        _emit(macros, "cqLocIntInteractionClCIlo", f"{llo:.3f}")
        _emit(macros, "cqLocIntInteractionClCIhi", f"{lhi:.3f}")
        cult_k = cult_n = 0
        for cl in loc_clusters:
            for (cu, fc, qc) in cl:
                if cu == 1:
                    cult_k += fc - qc
                    cult_n += 1
        _emit(macros, "cqLocIntCulturalDelta", _pct(cult_k / max(1, cult_n)))
        _emit(macros, "cqLocIntCulturalObs", cult_n)
    _emit(macros, "cqLocIntItems", len(loc_ids))

    # per-stratum int8 cultural erosion, pooled over band-clearers
    for st, tag in STRATUM_TAGS.items():
        fk = qk = n = 0
        for name in band_clearers:
            fp_rows = grid.get((name, "fp16"))
            q_rows = grid.get((name, "int8"))
            if not fp_rows or not q_rows:
                continue
            for i, fr in fp_rows.items():
                if fr["stratum"] != st or i not in q_rows:
                    continue
                fk += fr["correct"]
                qk += q_rows[i]["correct"]
                n += 1
        if n == 0:
            continue
        _emit(macros, f"cqStratum{tag}IntFp", _pct(fk / n))
        _emit(macros, f"cqStratum{tag}IntDelta", _pct((fk - qk) / n))

    # proverbs framing: share of cultural items, FP16 accuracy range over band-clearers
    n_cultural = sum(1 for r in any_fp.values() if r["group"] == "cultural")
    n_prov = sum(1 for r in any_fp.values() if r["stratum"] == "proverbs")
    if n_cultural:
        _emit(macros, "cqProverbsShare", round(100 * n_prov / n_cultural))
    prov_pts = []
    for name in band_clearers:
        fp_rows = grid.get((name, "fp16"))
        if not fp_rows:
            continue
        k = sum(r["correct"] for r in fp_rows.values() if r["stratum"] == "proverbs")
        nn = sum(1 for r in fp_rows.values() if r["stratum"] == "proverbs")
        if nn:
            prov_pts.append(100 * k / nn)
    if prov_pts:
        _emit(macros, "cqProverbsFpLo", f"{min(prov_pts):.1f}")
        _emit(macros, "cqProverbsFpHi", f"{max(prov_pts):.1f}")

    # control-ceiling count: band-clearers whose FP16 control accuracy is below ceiling
    below = 0
    for name in band_clearers:
        fp_rows = grid.get((name, "fp16"))
        if not fp_rows:
            continue
        k = sum(r["correct"] for r in fp_rows.values() if r["group"] == "control")
        nn = sum(1 for r in fp_rows.values() if r["group"] == "control")
        if nn and k / nn < CONTROL_CEILING:
            below += 1
    _emit(macros, "cqNControlBelowCeiling", below)

    # CO macro-region item count (reported as omitted from the coarse N+NE vs SE+S split)
    co_n = sum(1 for r in any_fp.values()
               if r["group"] == "cultural" and r["region"] == "CO")
    _emit(macros, "cqNRegionCOomitted", co_n)

    # Holm-corrected per-model McNemar family (5 band-clearers x 2 precisions x 2 groups)
    labels, pvals = [], []
    for name in band_clearers:
        for prec in ("int8", "nf4"):
            fp_rows = grid.get((name, "fp16"))
            q_rows = grid.get((name, prec))
            if not fp_rows or not q_rows:
                continue
            for grp in ("cultural", "control"):
                b = c = 0
                for i, fr in fp_rows.items():
                    if fr["group"] != grp or i not in q_rows:
                        continue
                    qc = q_rows[i]["correct"]
                    if fr["correct"] == 1 and qc == 0:
                        b += 1
                    elif fr["correct"] == 0 and qc == 1:
                        c += 1
                labels.append((model_tag(name), prec_tag(prec),
                               "Cultural" if grp == "cultural" else "Control"))
                pvals.append(mcnemar_exact_p(b, c))
    adj = holm_adjust(pvals)
    for (mt, pt, gtag), a in zip(labels, adj):
        _emit(macros, f"cqEOne{mt}{pt}{gtag}McPHolm", f"{a:.3f}")

    # band-clearer total wall-clock (minutes), from the manifest run timings
    # (filled in by the caller via the manifest; see _emit_wallclock)


def _emit_wallclock(macros: list[str], manifest, band_clearers) -> None:
    band_set = set(band_clearers)
    sec = sum(r["seconds"] for r in manifest["runs"] if r["model"] in band_set)
    _emit(macros, "cqBandClearerWallMin", f"{sec / 60:.1f}")


def _ols_slope(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    den = sum((xs[i] - mx) ** 2 for i in range(n))
    return num / den if den else float("nan")


if __name__ == "__main__":
    main()
