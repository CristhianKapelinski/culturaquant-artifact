"""Step 2: mid-band CALIBRATION of the rarity gradient at fp16 (cyclic).

Scores one JSONL set (default gradient_cultural.jsonl) at fp16 ONLY, under all 5
cyclic rotations per item (gold uniform over A..E by construction), reusing the
paper's constrained-likelihood scorer. Reports, PER MODEL, the mean fp16 accuracy
in each rarity bucket (sl0-1 / sl2-3 / sl4-7 / sl8-15 / sl16-30) so we can read off
the bucket(s) where fp16 lands ~0.4-0.7 (the mid-band, where erosion is detectable).

Resumable: a completed per-model summary file is skipped. Deterministic.

  python calibrate_gradient.py --models Qwen/Qwen2.5-1.5B-Instruct ... \
      --data /app/v2data/gradient_cultural.jsonl --out-dir /app/v2data/results/calib
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve()
_SRC = _HERE.parent.parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from culturaquant.scorer import free_model, load_model, score_item_logprobs  # noqa: E402

LETTERS = ["A", "B", "C", "D", "E"]
BUCKETS = ["sl0-1", "sl2-3", "sl4-7", "sl8-15", "sl16-30", "sl31-50", "sl51-80"]


def load_set(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


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


def score(lm, items: list[dict]) -> dict:
    per_bucket: dict[str, list[int]] = {b: [0, 0] for b in BUCKETS}
    n = k = 0
    rows = []
    for it in items:
        band = it.get("rarity", {}).get("band", "?")
        pb = per_bucket.setdefault(band, [0, 0])
        for r, opts, gold in cyclic_rotations(it["alternatives"], int(it["correct_index"])):
            norms = score_item_logprobs(lm, it["input"], opts)
            ok = int(_argmax(norms) == gold)
            n += 1
            k += ok
            pb[0] += ok
            pb[1] += 1
            rows.append({"id": it["id"], "rotation": r, "band": band, "correct": ok})
    return {"n": n, "k": k, "per_bucket": per_bucket, "rows": rows}


def reference_machine() -> dict:
    info = {"platform": platform.platform(), "python": platform.python_version()}
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
        info["torch"] = torch.__version__
    except Exception as e:
        info["gpu_error"] = str(e)
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    items = load_set(Path(args.data))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[items] {len(items)} -> {len(items)*5} (item,rotation) fp16 instances", flush=True)

    summaries = []
    for name in args.models:
        slug = name.replace("/", "__") + "__fp16"
        summ_path = out_dir / f"calib__{slug}.json"
        if summ_path.exists():
            s = json.loads(summ_path.read_text())
            s["status"] = "skipped"
            summaries.append(s)
            print(f"[skip] {name}", flush=True)
        else:
            print(f"[run] {name} @ fp16", flush=True)
            t0 = time.time()
            lm = load_model(name, "fp16")
            res = score(lm, items)
            free_model(lm)
            per_bucket_acc = {
                b: (round(v[0] / v[1], 4) if v[1] else None, v[1])
                for b, v in res["per_bucket"].items()
            }
            s = {
                "model": name, "precision": "fp16",
                "overall_acc": round(res["k"] / res["n"], 4), "n": res["n"],
                "per_bucket_acc": per_bucket_acc,
                "seconds": round(time.time() - t0, 1), "status": "done",
            }
            summ_path.write_text(json.dumps(s, ensure_ascii=False, indent=2))
            (out_dir / f"calib__{slug}.rows.jsonl").write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in res["rows"]) + "\n")
            summaries.append(s)
            print(f"[done] overall={s['overall_acc']} buckets="
                  f"{ {b: per_bucket_acc[b][0] for b in BUCKETS} }", flush=True)

        (out_dir / "calibration_table.json").write_text(json.dumps(
            {"reference_machine": reference_machine(), "n_items": len(items),
             "results": summaries}, ensure_ascii=False, indent=2))

    # Mean bucket accuracy across models -> identify the mid-band.
    print("\n==== fp16 CALIBRATION (per-bucket accuracy) ====", flush=True)
    hdr = f"{'model':30s}" + "".join(f"{b:>9s}" for b in BUCKETS) + f"{'overall':>9s}"
    print(hdr, flush=True)
    bucket_means = {b: [] for b in BUCKETS}
    for s in summaries:
        line = f"{s['model']:30s}"
        for b in BUCKETS:
            acc = s["per_bucket_acc"].get(b, [None, 0])[0]
            line += f"{(f'{acc:.3f}' if acc is not None else '  -  '):>9s}"
            if acc is not None:
                bucket_means[b].append(acc)
        line += f"{s['overall_acc']:>9.3f}"
        print(line, flush=True)
    mline = f"{'MEAN':30s}"
    for b in BUCKETS:
        vals = bucket_means[b]
        mline += f"{(f'{sum(vals)/len(vals):.3f}' if vals else '  -  '):>9s}"
    print(mline, flush=True)


if __name__ == "__main__":
    main()
