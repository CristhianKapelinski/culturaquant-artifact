"""Cyclic-rotation position-bias probe.

Tests ONE hypothesis: does post-training quantization (int8, nf4) systematically
increase a model's multiple-choice position/selection bias, measured with cyclic
permutation and the RStd metric?

For each item we generate ALL 5 cyclic rotations of the 5-option list. Rotation r
places the gold option at letter position r (so across the 5 rotations the gold
answer is uniform over A..E by construction). Each (item, rotation) is scored with
the SAME constrained-likelihood scorer used by the paper; we record the model's
CHOSEN position (argmax over the 5 per-option logprobs), regardless of
correctness.

Per (model, precision) we aggregate the marginal distribution of chosen positions
p = [p_A..p_E] over all (item x rotation) instances and report:
  RStd  = std(p_A..p_E)        (0 = unbiased uniform 0.2 each; higher = concentrated)
  favored position + its rate
  accuracy averaged over rotations
  the full predicted-position histogram (counts and probabilities)

Raw per-(item,rotation) rows are written so the aggregation is fully auditable.
This script does NOT touch the paper or the existing grid outputs; it writes into
its own --out-dir.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

# Allow running as a loose script (python scripts/...) or as a module.
import sys

_HERE = Path(__file__).resolve()
_SRC = _HERE.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from culturaquant.data import load_control, load_cultural_strata, Item  # noqa: E402
from culturaquant.scorer import (  # noqa: E402
    free_model,
    load_model,
    score_item_logprobs,
)

LETTERS = ["A", "B", "C", "D", "E"]


def cyclic_rotations(options: list[str], gold_index: int):
    """Yield the 5 cyclic rotations placing gold at each position 0..4.

    Rotation r is defined so that the gold option lands at index r. We build it by
    rotating the canonical list: new[i] = old[(i - r + gold_index) mod n]. With
    n=5 this enumerates 5 distinct cyclic orderings; the relative cyclic order of
    the distractors is preserved, only the offset changes, and gold sits at r.
    """
    n = len(options)
    for r in range(n):
        # We want new_options[r] == options[gold_index].
        # Define shift s = (gold_index - r) mod n, new[i] = options[(i + s) mod n].
        s = (gold_index - r) % n
        new_options = [options[(i + s) % n] for i in range(n)]
        new_gold = r
        assert new_options[new_gold] == options[gold_index]
        yield r, new_options, new_gold


def _argmax(xs: list[float]) -> int:
    best_i, best = 0, -float("inf")
    for i, x in enumerate(xs):
        if x > best:
            best, best_i = x, i
    return best_i


def std(xs: list[float]) -> float:
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / len(xs)) ** 0.5


def gather_items(data_dir: Path) -> list[Item]:
    # The 120 author-written items only (no BRoverbs); content difficulty is
    # irrelevant for a POSITION-bias measurement.
    return load_cultural_strata(data_dir) + load_control(data_dir)


def run_one(name: str, precision: str, items: list[Item], out_dir: Path) -> dict:
    slug = name.replace("/", "__") + f"__{precision}"
    raw_path = out_dir / f"cyclic__{slug}.jsonl"
    if raw_path.exists():
        # Resume: trust a complete file (last line carries _summary).
        try:
            lines = raw_path.read_text(encoding="utf-8").splitlines()
            last = json.loads(lines[-1]) if lines else {}
            if last.get("_summary"):
                last["status"] = "skipped"
                return last
        except (json.JSONDecodeError, OSError, IndexError):
            pass

    t0 = time.time()
    lm = load_model(name, precision)
    pos_counts = [0] * 5
    n_correct = 0
    n_inst = 0
    rows = []
    for it in items:
        for r, opts, gold in cyclic_rotations(list(it.options), it.gold_index):
            norms = score_item_logprobs(lm, it.question, opts)
            pred = _argmax(norms)
            pos_counts[pred] += 1
            n_inst += 1
            if pred == gold:
                n_correct += 1
            rows.append(
                {
                    "id": it.id,
                    "group": it.group,
                    "rotation": r,
                    "gold": gold,
                    "pred": pred,
                    "correct": int(pred == gold),
                    "logprobs": [round(x, 5) for x in norms],
                }
            )
    free_model(lm)
    elapsed = time.time() - t0

    p = [c / n_inst for c in pos_counts]
    rstd = std(p)
    fav = _argmax(p)
    summary = {
        "_summary": True,
        "model": name,
        "precision": precision,
        "n_items": len(items),
        "n_instances": n_inst,
        "pos_counts": pos_counts,
        "pos_prob": [round(x, 5) for x in p],
        "rstd": round(rstd, 5),
        "favored_position": LETTERS[fav],
        "favored_rate": round(p[fav], 5),
        "accuracy": round(n_correct / n_inst, 5),
        "seconds": round(elapsed, 1),
        "status": "done",
    }

    tmp = raw_path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    tmp.replace(raw_path)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--precisions", nargs="+", default=["fp16", "int8", "nf4"])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = gather_items(data_dir)
    print(f"[items] {len(items)} items -> {len(items) * 5} (item,rotation) instances",
          flush=True)

    summaries = []
    for name in args.models:
        for prec in args.precisions:
            print(f"[run] {name} @ {prec}", flush=True)
            s = run_one(name, prec, items, out_dir)
            print(f"[done] rstd={s['rstd']} favored={s['favored_position']}"
                  f"({s['favored_rate']}) acc={s['accuracy']} "
                  f"status={s['status']}", flush=True)
            summaries.append(s)
            # Persist the RStd table after every unit so the run is resumable and
            # partial results survive a power cut.
            table_path = out_dir / "rstd_table.json"
            with table_path.open("w", encoding="utf-8") as f:
                json.dump(
                    {"reference_machine": reference_machine(),
                     "n_items": len(items),
                     "results": summaries},
                    f, ensure_ascii=False, indent=2,
                )

    print("[ok] wrote", out_dir / "rstd_table.json", flush=True)


if __name__ == "__main__":
    main()
