"""Phase-1 quality report for the FULL CulturaQuant-v2 build.

Reads data/br_rare.jsonl + data/control_matched.jsonl (+ build_stats.json) and reports:
  - counts, per-region (BR), per-relation, per-country (control)
  - gold-position balance (A..E) for both sets
  - item-by-item rarity match + KS test on sitelinks AND log-pageviews
  - verified/dropped accounting (from build_stats.json)
  - flagged items (dup options, gold-in-distractors, option-in-question)
No network. Mirrors build/quality_report.py but for the scaled, paired files.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
LETTERS = ["A", "B", "C", "D", "E"]


def load(name: str) -> list[dict]:
    p = DATA / name
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def ks_2samp(a: list[float], b: list[float]) -> str:
    if not a or not b:
        return "n/a"
    av, bv = sorted(a), sorted(b)
    allv = sorted(set(av + bv))

    def cdf(s, x):
        lo, hi = 0, len(s)
        while lo < hi:
            mid = (lo + hi) // 2
            if s[mid] <= x:
                lo = mid + 1
            else:
                hi = mid
        return lo / len(s)

    d = max(abs(cdf(av, x) - cdf(bv, x)) for x in allv)
    n, m = len(a), len(b)
    crit = 1.358 * math.sqrt((n + m) / (n * m))
    note = "n.s. (distributions overlap)" if d <= crit else "SIGNIFICANT (mismatch)"
    return f"{note}; D={d:.3f}, crit_0.05={crit:.3f}"


def _median(xs):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def flags(rec):
    f = []
    alts = rec["alternatives"]
    if len(set(alts)) != len(alts):
        f.append("DUP_OPTION")
    gold = alts[rec["correct_index"]]
    if alts.count(gold) > 1:
        f.append("GOLD_DUP_IN_DISTRACTORS")
    for a in alts:
        if a != gold and a and a in rec["input"]:
            f.append(f"OPTION_IN_QUESTION:{a}")
    if rec["rarity"]["sitelinks"] > 5:
        f.append("NOT_LONGTAIL")
    return f


def report():
    br = load("br_rare.jsonl")
    ctrl = load("control_matched.jsonl")
    stats = json.loads((DATA / "build_stats.json").read_text()) if (DATA / "build_stats.json").exists() else {}

    print("=" * 78)
    print("CulturaQuant-v2 FULL BUILD QUALITY REPORT")
    print("=" * 78)
    print(f"BR-rare items: {len(br)}   control items: {len(ctrl)}\n")

    print("VERIFIED / DROPPED (gold re-verification):")
    print(f"  BR candidates fetched         : {stats.get('br_candidates','?')}")
    print(f"  BR verified                   : {stats.get('br_verified','?')}")
    print(f"  BR dropped (ASK failed)       : {stats.get('br_dropped','?')}")
    print(f"  BR final (MCQ built)          : {stats.get('br_final','?')}")
    print(f"  BR paired w/ control          : {stats.get('br_paired','?')}")
    print(f"  control candidates fetched    : {stats.get('control_candidates','?')}")
    print(f"  control verified (candidates) : {stats.get('control_verified_candidates','?')}")
    print(f"  control final (matched)       : {stats.get('control_final','?')}")

    print("\nREGION COVERAGE (BR):")
    rc = Counter(r["region"] for r in br)
    for k in ["N", "NE", "CO", "SE", "S"]:
        print(f"  {k:3s}: {rc.get(k,0)}")
    print(f"  per-relation: {dict(Counter(r['relation'] for r in br))}")

    print("\nCONTROL SOURCE COUNTRIES:")
    cc = Counter(r["country"] for r in ctrl)
    print("  " + "  ".join(f"{k}={v}" for k, v in sorted(cc.items())))
    print(f"  per-relation: {dict(Counter(r['relation'] for r in ctrl))}")

    print("\nGOLD POSITION BALANCE:")
    for name, recs in (("BR", br), ("control", ctrl)):
        c = Counter(LETTERS[r["correct_index"]] for r in recs)
        print(f"  {name:8s}: " + "  ".join(f"{L}={c.get(L,0)}" for L in LETTERS))

    # Rarity match + KS
    bs = [r["rarity"]["sitelinks"] for r in br]
    cs = [r["rarity"]["sitelinks"] for r in ctrl]
    bp = [r["rarity"]["log_pageviews"] for r in br]
    cp = [r["rarity"]["log_pageviews"] for r in ctrl]
    print("\nRARITY MATCH (BR vs control):")
    print(f"  BR      median sitelinks={_median(bs):.1f}  median log_pv={_median(bp):.3f}")
    print(f"  control median sitelinks={_median(cs):.1f}  median log_pv={_median(cp):.3f}")
    print(f"  KS sitelinks  : {ks_2samp([float(x) for x in bs], [float(x) for x in cs])}")
    print(f"  KS log_pviews : {ks_2samp(bp, cp)}")
    band_br = Counter(r["rarity"]["band"] for r in br)
    band_ct = Counter(r["rarity"]["band"] for r in ctrl)
    print(f"  band BR      : {dict(band_br)}")
    print(f"  band control : {dict(band_ct)}")

    af = [(r["id"], flags(r)) for r in br + ctrl if flags(r)]
    print(f"\nFLAGGED ITEMS: {len(af)}")
    for i, fl in af[:40]:
        print(f"  {i}: {', '.join(fl)}")


if __name__ == "__main__":
    report()
