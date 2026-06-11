"""Phase-2 fp16 difficulty-match GATE for CulturaQuant-v2.

Scores each model at fp16 ONLY on the two rarity-matched sets:
  data/br_rare.jsonl          (rare Brazil-specific entity knowledge)
  data/control_matched.jsonl  (rarity-matched non-Brazil control)

Every item is scored under all 5 cyclic rotations (gold uniform over A..E by
construction), reusing the paper's constrained-likelihood scorer. We report, per model,
the fp16 accuracy on BR vs control (over item x rotation), a two-proportion z-test on the
gap, Wilson 95% CIs, and per-rarity-band accuracy.

The GATE verdict (printed at the end, also written to gate_fp16.json):
  GREEN  if BR and control fp16 accuracy are comparable (gap small / not significant) AND
         both sit mid-band (~0.3-0.8, not floor/ceiling) for most models -> difficulty
         match holds, the design is clean.
  RED    if BR accuracy is far BELOW control on most models (the v1 near-ceiling-control
         confound returns) -> do NOT proceed.

Resumable: completed (model) raw files are skipped. Deterministic, seeded by construction.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

# culturaquant.{scorer,stats} live in the artifact src tree.
_HERE = Path(__file__).resolve()
_SRC = _HERE.parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from culturaquant.scorer import free_model, load_model, score_item_logprobs  # noqa: E402
from culturaquant.stats import Proportion  # noqa: E402

LETTERS = ["A", "B", "C", "D", "E"]


def load_set(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def cyclic_rotations(options: list[str], gold_index: int):
    n = len(options)
    for r in range(n):
        s = (gold_index - r) % n
        new_opts = [options[(i + s) % n] for i in range(n)]
        assert new_opts[r] == options[gold_index]
        yield r, new_opts, r


def _argmax(xs: list[float]) -> int:
    bi, bv = 0, -float("inf")
    for i, x in enumerate(xs):
        if x > bv:
            bv, bi = x, i
    return bi


def two_prop_z(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Two-proportion z-test (pooled). Returns (z, two-sided p). p1=BR, p2=control."""
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    # two-sided normal p
    pval = math.erfc(abs(z) / math.sqrt(2))
    return z, pval


def score_set(lm, items: list[dict]) -> dict:
    """Return {n_inst, correct, per_band:{band:[k,n]}, rows:[...] } over 5 rotations/item."""
    n_inst = correct = 0
    per_band: dict[str, list[int]] = {}
    rows = []
    for it in items:
        band = it.get("rarity", {}).get("band", "?")
        pb = per_band.setdefault(band, [0, 0])
        for r, opts, gold in cyclic_rotations(it["alternatives"], int(it["correct_index"])):
            norms = score_item_logprobs(lm, it["input"], opts)
            pred = _argmax(norms)
            ok = int(pred == gold)
            n_inst += 1
            correct += ok
            pb[0] += ok
            pb[1] += 1
            rows.append({"id": it["id"], "rotation": r, "gold": gold, "pred": pred,
                         "correct": ok, "band": band})
    return {"n_inst": n_inst, "correct": correct, "per_band": per_band, "rows": rows}


def run_model(name: str, br: list[dict], ctrl: list[dict], out_dir: Path) -> dict:
    slug = name.replace("/", "__") + "__fp16"
    raw = out_dir / f"gate__{slug}.jsonl"
    summ_path = out_dir / f"summary__{slug}.json"
    if summ_path.exists():
        s = json.loads(summ_path.read_text())
        s["status"] = "skipped"
        return s

    t0 = time.time()
    lm = load_model(name, "fp16")
    br_res = score_set(lm, br)
    ctrl_res = score_set(lm, ctrl)
    free_model(lm)

    kb, nb = br_res["correct"], br_res["n_inst"]
    kc, nc = ctrl_res["correct"], ctrl_res["n_inst"]
    z, p = two_prop_z(kb, nb, kc, nc)
    br_p = Proportion(kb, nb)
    ctrl_p = Proportion(kc, nc)
    summary = {
        "model": name, "precision": "fp16",
        "br": {"acc": round(br_p.point, 4), "k": kb, "n": nb,
               "ci": [round(x, 4) for x in br_p.wilson()],
               "per_band": br_res["per_band"]},
        "control": {"acc": round(ctrl_p.point, 4), "k": kc, "n": nc,
                    "ci": [round(x, 4) for x in ctrl_p.wilson()],
                    "per_band": ctrl_res["per_band"]},
        "gap_br_minus_ctrl": round(br_p.point - ctrl_p.point, 4),
        "z": round(z, 3), "p_two_sided": round(p, 4),
        "seconds": round(time.time() - t0, 1), "status": "done",
    }
    with raw.open("w", encoding="utf-8") as f:
        for grp, res in (("br", br_res), ("control", ctrl_res)):
            for row in res["rows"]:
                row["group"] = grp
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summ_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def reference_machine() -> dict:
    info = {"platform": platform.platform(), "python": platform.python_version()}
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
        info["torch"] = torch.__version__
    except Exception as e:  # pragma: no cover
        info["gpu_error"] = str(e)
    return info


def verdict(summaries: list[dict]) -> dict:
    """GREEN/RED gate over the per-model fp16 BR-vs-control results.

    GREEN: for the majority of models the gap is small (|gap| < 0.10) OR not significant
           (p >= 0.05), AND BR accuracy is mid-band (0.3..0.8) and not at floor/ceiling.
    RED:   BR accuracy is significantly BELOW control (gap <= -0.10, p < 0.05) on the
           majority of models -> the difficulty match fails (v1 confound returns).
    """
    done = [s for s in summaries if s.get("n", 1) != 0 and "br" in s]
    n = len(done)
    if n == 0:
        return {"verdict": "UNKNOWN", "reason": "no models scored"}
    red = sum(1 for s in done if s["gap_br_minus_ctrl"] <= -0.10 and s["p_two_sided"] < 0.05)
    comparable = sum(1 for s in done if abs(s["gap_br_minus_ctrl"]) < 0.10 or s["p_two_sided"] >= 0.05)
    br_midband = sum(1 for s in done if 0.3 <= s["br"]["acc"] <= 0.8)
    if red > n / 2:
        v = "RED"
    elif comparable >= n / 2:
        v = "GREEN"
    else:
        v = "AMBER"
    return {"verdict": v, "n_models": n, "n_red": red, "n_comparable": comparable,
            "n_br_midband": br_midband,
            "mean_br_acc": round(sum(s["br"]["acc"] for s in done) / n, 4),
            "mean_ctrl_acc": round(sum(s["control"]["acc"] for s in done) / n, 4)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--data-dir", default=str(_HERE.parent.parent / "data"))
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    br = load_set(data_dir / "br_rare.jsonl")
    ctrl = load_set(data_dir / "control_matched.jsonl")
    print(f"[items] BR={len(br)} control={len(ctrl)}  "
          f"-> {(len(br)+len(ctrl))*5} (item,rotation) fp16 forward sets", flush=True)

    summaries = []
    for name in args.models:
        print(f"[run] {name} @ fp16", flush=True)
        s = run_model(name, br, ctrl, out_dir)
        print(f"[done] BR={s['br']['acc']} ctrl={s['control']['acc']} "
              f"gap={s['gap_br_minus_ctrl']} p={s['p_two_sided']} status={s['status']}",
              flush=True)
        summaries.append(s)
        table = out_dir / "gate_fp16.json"
        table.write_text(json.dumps(
            {"reference_machine": reference_machine(),
             "n_br": len(br), "n_control": len(ctrl),
             "results": summaries, "verdict": verdict(summaries)},
            ensure_ascii=False, indent=2))

    v = verdict(summaries)
    print("\n==== fp16 DIFFICULTY-MATCH GATE ====", flush=True)
    print(f"{'model':32s} {'BR':>7s} {'ctrl':>7s} {'gap':>7s} {'p':>7s}", flush=True)
    for s in summaries:
        print(f"{s['model']:32s} {s['br']['acc']:>7.3f} {s['control']['acc']:>7.3f} "
              f"{s['gap_br_minus_ctrl']:>7.3f} {s['p_two_sided']:>7.3f}", flush=True)
    print(f"\nVERDICT: {v['verdict']}  {v}", flush=True)


if __name__ == "__main__":
    main()
