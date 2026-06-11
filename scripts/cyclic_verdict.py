"""Read the cyclic position-bias outputs and render the verdict.

Consumes ``rstd_table.json`` (per model x precision RStd, favored position, etc.)
and prints:
  - the RStd table (fp16 vs int8 vs nf4, favored position + rate, accuracy),
  - per-model deltas RStd(int8)-RStd(fp16) and RStd(nf4)-RStd(fp16),
  - a sign test and Wilcoxon signed-rank test on RStd(nf4) vs RStd(fp16) and
    RStd(int8) vs RStd(fp16) across models,
  - predicted-position histograms (fp16 vs nf4) for the requested models,
  - an explicit, honest verdict line.

No external stats deps: exact sign test via binomial, Wilcoxon via the normal
approximation with a small-n exact fallback note.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

LETTERS = ["A", "B", "C", "D", "E"]


def binom_two_sided_p(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial p-value for k successes in n trials (p=0.5)."""
    if n == 0:
        return 1.0
    def pmf(i):
        return math.comb(n, i) * p**i * (1 - p) ** (n - i)
    obs = pmf(k)
    return min(1.0, sum(pmf(i) for i in range(n + 1) if pmf(i) <= obs + 1e-12))


def wilcoxon_signed_rank(diffs: list[float]):
    """Wilcoxon signed-rank on nonzero diffs. Returns (W, z, p_normal, n)."""
    nz = [d for d in diffs if abs(d) > 1e-12]
    n = len(nz)
    if n == 0:
        return 0.0, 0.0, 1.0, 0
    order = sorted(range(n), key=lambda i: abs(nz[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(abs(nz[order[j + 1]]) - abs(nz[order[i]])) < 1e-12:
            j += 1
        avg = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_plus = sum(ranks[i] for i in range(n) if nz[i] > 0)
    w_minus = sum(ranks[i] for i in range(n) if nz[i] < 0)
    W = min(w_plus, w_minus)
    mean = n * (n + 1) / 4.0
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
    z = (W - mean) / sd if sd > 0 else 0.0
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return W, z, min(1.0, p), n


def load(out_dir: Path):
    tbl = json.loads((out_dir / "rstd_table.json").read_text(encoding="utf-8"))
    by = {}
    for r in tbl["results"]:
        by.setdefault(r["model"], {})[r["precision"]] = r
    return tbl, by


def short(name: str) -> str:
    return name.split("/")[-1]


def histogram(out_dir: Path, model: str, precision: str):
    slug = model.replace("/", "__") + f"__{precision}"
    p = out_dir / f"cyclic__{slug}.jsonl"
    if not p.exists():
        return None
    last = p.read_text(encoding="utf-8").splitlines()[-1]
    s = json.loads(last)
    return s.get("pos_prob"), s.get("pos_counts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hist-models", nargs="*", default=[])
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    tbl, by = load(out_dir)

    print("== reference machine ==")
    print(json.dumps(tbl.get("reference_machine", {})))
    print(f"n_items={tbl.get('n_items')}  (x5 rotations per item)\n")

    precs = ["fp16", "int8", "nf4"]
    print("== RStd table (favored position / rate ; accuracy) ==")
    hdr = f"{'model':<26}" + "".join(f"{p:>22}" for p in precs)
    print(hdr)
    for model in by:
        row = f"{short(model):<26}"
        for p in precs:
            r = by[model].get(p)
            if r:
                row += f"{r['rstd']:.4f} {r['favored_position']}{int(round(r['favored_rate']*100)):>2}% a{int(round(r['accuracy']*100)):>2}%".rjust(22)
            else:
                row += f"{'-':>22}"
        print(row)

    print("\n== per-model RStd deltas vs fp16 ==")
    d_int8, d_nf4 = [], []
    for model in by:
        f = by[model].get("fp16"); i8 = by[model].get("int8"); n4 = by[model].get("nf4")
        di = (i8["rstd"] - f["rstd"]) if (f and i8) else None
        dn = (n4["rstd"] - f["rstd"]) if (f and n4) else None
        if di is not None:
            d_int8.append(di)
        if dn is not None:
            d_nf4.append(dn)
        print(f"{short(model):<26} d(int8-fp16)={di if di is None else round(di,4):>8}"
              f"   d(nf4-fp16)={dn if dn is None else round(dn,4):>8}")

    for label, diffs in [("nf4 vs fp16", d_nf4), ("int8 vs fp16", d_int8)]:
        n = len(diffs)
        pos = sum(1 for d in diffs if d > 1e-12)
        neg = sum(1 for d in diffs if d < -1e-12)
        p_sign = binom_two_sided_p(pos, pos + neg)
        W, z, p_w, nw = wilcoxon_signed_rank(diffs)
        mean = sum(diffs) / n if n else 0.0
        print(f"\n== {label}: n={n}, increased(+)={pos}, decreased(-)={neg}, "
              f"mean delta={mean:+.4f} ==")
        print(f"   sign test two-sided p={p_sign:.4f}  "
              f"(supported only if + dominates AND p small)")
        print(f"   Wilcoxon W={W:.1f} z={z:.2f} p~={p_w:.4f} (normal approx, n={nw})")

    for model in args.hist_models:
        full = next((m for m in by if short(m) == model or m == model), None)
        if not full:
            print(f"\n[hist] model {model} not found"); continue
        print(f"\n== predicted-position histogram: {short(full)} ==")
        for p in ("fp16", "nf4"):
            h = histogram(out_dir, full, p)
            if not h:
                print(f"  {p}: (missing)"); continue
            probs, counts = h
            bars = "  ".join(f"{LETTERS[i]}:{probs[i]*100:4.1f}%" for i in range(5))
            print(f"  {p:<5} {bars}")
            graph = ""
            for i in range(5):
                graph += f"    {LETTERS[i]} " + "#" * int(round(probs[i] * 100)) + f" {probs[i]*100:.1f}%\n"
            print(graph, end="")


if __name__ == "__main__":
    main()
