"""Merge sharded main-grid prediction dirs into one analyzable grid dir.

The main grid is run across several GPU hosts, each scoring a disjoint set of
models into its own out-dir (each with a partial manifest.json). This collects
all ``preds__*.jsonl`` and rebuilds ONE manifest.json whose ``runs`` list points
at every preds file and whose ``models`` is the union, so ``culturaquant.analyze``
can consume the full 8-model x 3-precision grid unchanged.

Usage:
    python scripts/merge_grid.py --in DIR1 DIR2 ... --out MERGED_DIR
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _acc(path: Path) -> tuple[int, float]:
    k = n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("_meta"):
                continue
            k += r["correct"]
            n += 1
    return n, (k / n if n else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="ins", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    base_manifest = None
    runs: dict[str, dict] = {}
    models: list[str] = []
    precisions: list[str] = []
    n_items = None
    seed = None
    strata = None

    for d in args.ins:
        dp = Path(d)
        mpath = dp / "manifest.json"
        if not mpath.exists():
            raise SystemExit(f"missing manifest in {dp}")
        man = json.loads(mpath.read_text())
        base_manifest = base_manifest or man
        n_items = n_items or man["n_items"]
        seed = seed if seed is not None else man["seed"]
        strata = strata or man.get("item_strata")
        for p in man["precisions"]:
            if p not in precisions:
                precisions.append(p)
        for m in man["models"]:
            if m not in models:
                models.append(m)
        for run in man["runs"]:
            src = dp / run["preds_file"]
            if not src.exists():
                continue
            dst = out / run["preds_file"]
            if src.resolve() != dst.resolve():
                shutil.copyfile(src, dst)
            n, acc = _acc(dst)
            run = dict(run)
            run["n"] = n
            run["acc"] = acc
            runs[run["preds_file"]] = run

    merged = {
        "seed": seed,
        "n_items": n_items,
        "n_proverbs": base_manifest.get("n_proverbs"),
        "scoring": base_manifest.get("scoring"),
        "models": models,
        "precisions": precisions,
        "reference_machine": base_manifest["reference_machine"],
        "runs": list(runs.values()),
        "item_strata": strata,
    }
    (out / "manifest.json").write_text(
        json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[merge] {len(runs)} runs, {len(models)} models -> {out}/manifest.json")


if __name__ == "__main__":
    main()
