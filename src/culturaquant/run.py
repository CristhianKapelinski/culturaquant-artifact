"""Run the scoring grid: each model x each precision over all items.

Writes one JSONL of per-item predictions per (model, precision) into the output
directory, plus a manifest recording model ids, precisions, item counts, the
seed, and the resolved reference machine. Growing data goes to the bind-mounted
output dir, never inside the container image.

Options are deterministically permuted per item (seeded by item id) so the gold
answer is not pinned at index 0. The author-written cultural and control items
ship with the correct answer fixed at index 0; scoring them in that order makes
the gold answer always "A", so any positional bias toward the first option
inflates accuracy (NF4 in particular collapses toward "A"). The SAME permutation
is applied to every precision for one item, so the paired cross-precision
comparison stays clean. This mirrors the gate path (run_gate.score_model).
"""

from __future__ import annotations

import argparse
import hashlib
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
from .scorer import (
    apply_permutation,
    free_model,
    load_model,
    option_permutation,
    score_item_logprobs,
)

# Bump when the scoring contract changes so stale outputs re-run.
SCORING_CONTRACT = "perm_options_v2"


def _slug(name: str, precision: str) -> str:
    return name.replace("/", "__") + f"__{precision}"


def gather_items(data_dir: Path, n_proverbs: int, seed: int) -> list[Item]:
    items = load_cultural_strata(data_dir) + load_control(data_dir)
    if n_proverbs > 0:
        items += load_broverbs_proverbs(n_proverbs, seed)
    return items


def _argmax(norms: list[float]) -> int:
    best_idx = 0
    best_score = -float("inf")
    for i, norm in enumerate(norms):
        if norm > best_score:
            best_score = norm
            best_idx = i
    return best_idx


def _unit_hash(name: str, precision: str, n_items: int, seed: int) -> str:
    payload = json.dumps(
        {
            "model": name,
            "precision": precision,
            "n_items": n_items,
            "seed": seed,
            "scoring": SCORING_CONTRACT,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def run_one(
    name: str, precision: str, items: list[Item], out_dir: Path, seed: int
) -> dict:
    """Score one (model, precision) over all items with permuted options.

    Idempotent: if the output JSONL exists and its embedded unit hash matches, the
    run is skipped and the cached summary returned. This makes the grid resumable
    and safe to shard across machines (each host claims a disjoint model set).
    """
    out = out_dir / f"preds__{_slug(name, precision)}.jsonl"
    uhash = _unit_hash(name, precision, len(items), seed)
    if out.exists():
        try:
            with out.open(encoding="utf-8") as f:
                first = f.readline()
            meta = json.loads(first) if first.strip() else {}
            if meta.get("_meta") and meta.get("unit_hash") == uhash:
                preds = []
                with out.open(encoding="utf-8") as f:
                    for line in f:
                        r = json.loads(line)
                        if r.get("_meta"):
                            continue
                        preds.append(r)
                acc = sum(p["correct"] for p in preds) / max(1, len(preds))
                return {"model": name, "precision": precision, "n": len(preds),
                        "acc": acc, "seconds": 0.0, "preds_file": out.name,
                        "status": "skipped"}
        except (json.JSONDecodeError, OSError):
            pass

    t0 = time.time()
    lm = load_model(name, precision)
    preds = []
    for it in items:
        perm = option_permutation(it.id, len(it.options), seed)
        opts, gold = apply_permutation(list(it.options), it.gold_index, perm)
        norms = score_item_logprobs(lm, it.question, opts)
        pred = _argmax(norms)
        preds.append(
            {
                "id": it.id,
                "group": it.group,
                "stratum": it.stratum,
                "region": it.region,
                "gold": gold,
                "gold_canonical": it.gold_index,
                "option_permutation": perm,
                "pred": pred,
                "correct": int(pred == gold),
            }
        )
    free_model(lm)
    elapsed = time.time() - t0
    tmp = out.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        meta = {"_meta": True, "unit_hash": uhash, "model": name,
                "precision": precision, "seed": seed,
                "scoring": SCORING_CONTRACT, "n_items": len(items)}
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    tmp.replace(out)
    acc = sum(p["correct"] for p in preds) / len(preds)
    return {"model": name, "precision": precision, "n": len(preds),
            "acc": acc, "seconds": round(elapsed, 1), "preds_file": out.name,
            "status": "done"}


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


def _sanity_guard(out_dir: Path, models: list[str]) -> None:
    """NF4 must not beat int8/fp16 on the cultural set by more than 8pp.

    A large NF4 advantage on the cultural items is the position-bias signature the
    shuffle is meant to remove (NF4 collapsing toward option A). After shuffling,
    NF4 cultural accuracy should not exceed the higher-precision passes by >8pp for
    any model. We only assert over the per-model preds present in ``out_dir``.
    """
    margin = 0.08

    def cult_acc(name: str, prec: str) -> float | None:
        p = out_dir / f"preds__{_slug(name, prec)}.jsonl"
        if not p.exists():
            return None
        k = n = 0
        with p.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("_meta") or r.get("group") != "cultural":
                    continue
                k += r["correct"]
                n += 1
        return k / n if n else None

    violations = []
    for name in models:
        nf4 = cult_acc(name, "nf4")
        if nf4 is None:
            continue
        for ref in ("int8", "fp16"):
            ra = cult_acc(name, ref)
            if ra is not None and nf4 - ra > margin:
                violations.append(
                    f"{name}: nf4 cultural {nf4:.3f} exceeds {ref} {ra:.3f} "
                    f"by {100 * (nf4 - ra):.1f}pp (>8pp)"
                )
    if violations:
        msg = "[sanity] NF4 cultural position-bias guard FAILED:\n  " + \
            "\n  ".join(violations)
        raise SystemExit(msg)
    print("[sanity] NF4 cultural <= int8/fp16 + 8pp: OK", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--precisions", nargs="+", default=["fp16", "int8", "nf4"])
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-proverbs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=20260607)
    ap.add_argument("--no-guard", action="store_true",
                    help="skip the NF4 cultural sanity guard (sharded runs)")
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
            res = run_one(name, prec, items, out_dir, args.seed)
            print(f"[done] {res}", flush=True)
            summary.append(res)

    manifest = {
        "seed": args.seed,
        "n_items": len(items),
        "n_proverbs": args.n_proverbs,
        "scoring": SCORING_CONTRACT,
        "models": args.models,
        "precisions": args.precisions,
        "reference_machine": reference_machine(),
        "runs": summary,
        "item_strata": _strata_counts(items),
    }
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("[manifest]", json.dumps(manifest["reference_machine"]), flush=True)

    if not args.no_guard:
        _sanity_guard(out_dir, args.models)


def _strata_counts(items: list[Item]) -> dict:
    from collections import Counter

    return {
        "by_group": dict(Counter(i.group for i in items)),
        "by_stratum": dict(Counter(i.stratum for i in items)),
        "by_region": dict(Counter(i.region for i in items if i.group == "cultural")),
    }


if __name__ == "__main__":
    main()
