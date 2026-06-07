"""Run the scoring grid: each model x each precision over all items.

Writes one JSONL of per-item predictions per (model, precision) into the output
directory, plus a manifest recording model ids, precisions, item counts, the
seed, and the resolved reference machine. Growing data goes to the bind-mounted
output dir, never inside the container image.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

from .data import (
    load_broverbs_proverbs,
    load_control,
    load_cultural_strata,
    Item,
)
from .scorer import free_model, load_model, score_item


def _slug(name: str, precision: str) -> str:
    return name.replace("/", "__") + f"__{precision}"


def gather_items(data_dir: Path, n_proverbs: int, seed: int) -> list[Item]:
    items = load_cultural_strata(data_dir) + load_control(data_dir)
    if n_proverbs > 0:
        items += load_broverbs_proverbs(n_proverbs, seed)
    return items


def run_one(name: str, precision: str, items: list[Item], out_dir: Path) -> dict:
    t0 = time.time()
    lm = load_model(name, precision)
    preds = []
    for it in items:
        pred = score_item(lm, it.question, list(it.options))
        preds.append(
            {
                "id": it.id,
                "group": it.group,
                "stratum": it.stratum,
                "region": it.region,
                "gold": it.gold_index,
                "pred": pred,
                "correct": int(pred == it.gold_index),
            }
        )
    free_model(lm)
    elapsed = time.time() - t0
    out = out_dir / f"preds__{_slug(name, precision)}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    acc = sum(p["correct"] for p in preds) / len(preds)
    return {"model": name, "precision": precision, "n": len(preds),
            "acc": acc, "seconds": round(elapsed, 1), "preds_file": out.name}


def reference_machine() -> dict:
    info = {"platform": platform.platform(), "python": platform.python_version()}
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_mem_gib"] = round(props.total_memory / (1024 ** 3), 1)
            info["cuda"] = torch.version.cuda
        info["torch"] = torch.__version__
    except Exception as e:  # pragma: no cover
        info["gpu_error"] = str(e)
    try:
        info["cpu_count"] = os.cpu_count()
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    kb = int(line.split()[1])
                    info["ram_gib"] = round(kb / (1024 ** 2), 1)
                    break
    except Exception:
        pass
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--precisions", nargs="+", default=["fp16", "int8", "nf4"])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-proverbs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=20260607)
    args = ap.parse_args()

    import random
    import numpy as np
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = gather_items(data_dir, args.n_proverbs, args.seed)
    summary = []
    for name in args.models:
        for prec in args.precisions:
            print(f"[run] {name} @ {prec} on {len(items)} items", flush=True)
            res = run_one(name, prec, items, out_dir)
            print(f"[done] {res}", flush=True)
            summary.append(res)

    manifest = {
        "seed": args.seed,
        "n_items": len(items),
        "n_proverbs": args.n_proverbs,
        "models": args.models,
        "precisions": args.precisions,
        "reference_machine": reference_machine(),
        "runs": summary,
        "item_strata": _strata_counts(items),
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("[manifest]", json.dumps(manifest["reference_machine"]), flush=True)


def _strata_counts(items: list[Item]) -> dict:
    from collections import Counter

    return {
        "by_group": dict(Counter(i.group for i in items)),
        "by_stratum": dict(Counter(i.stratum for i in items)),
        "by_region": dict(Counter(i.region for i in items if i.group == "cultural")),
    }


if __name__ == "__main__":
    main()
