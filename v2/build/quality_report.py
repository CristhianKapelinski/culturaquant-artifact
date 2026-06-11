"""Quality report for the CulturaQuant-v2 smoke batch.

Lists every item (question, options, gold, region, rarity), compares cultural vs control
rarity side by side, runs a KS test on the two sitelink/pageview distributions, checks
gold-position balance, and flags potentially weak/ambiguous items (duplicate option
labels, gold label appearing among distractors, item label colliding with an option).
No network access; reads only the JSONL the generator wrote.
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


def ks_2samp(a: list[float], b: list[float]) -> tuple[float, str]:
    """Two-sample KS statistic and a coarse significance note (no scipy)."""
    if not a or not b:
        return float("nan"), "n/a"
    av, bv = sorted(a), sorted(b)
    allv = sorted(set(av + bv))

    def cdf(s: list[float], x: float) -> float:
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
    crit = 1.358 * math.sqrt((n + m) / (n * m))  # alpha=0.05
    note = "n.s. (distributions overlap)" if d <= crit else "SIGNIFICANT (mismatch)"
    return d, f"{note}; D={d:.3f}, crit_0.05={crit:.3f}"


def fmt_options(rec: dict) -> str:
    lines = []
    for i, alt in enumerate(rec["alternatives"]):
        mark = " <- GOLD" if i == rec["correct_index"] else ""
        lines.append(f"      {LETTERS[i]}) {alt}{mark}")
    return "\n".join(lines)


def flags(rec: dict) -> list[str]:
    f = []
    alts = rec["alternatives"]
    if len(set(alts)) != len(alts):
        f.append("DUP_OPTION")
    gold = alts[rec["correct_index"]]
    if alts.count(gold) > 1:
        f.append("GOLD_DUP_IN_DISTRACTORS")
    # item label colliding with an option (self-reveal)
    inp = rec["input"]
    for a in alts:
        if a != gold and a and a in inp:
            f.append(f"OPTION_IN_QUESTION:{a}")
    if rec["rarity"]["sitelinks"] > 3:
        f.append("NOT_LONGTAIL")
    return f


def report() -> None:
    cult = load("smoke_cultural.jsonl")
    ctrl = load("smoke_control.jsonl")

    print("=" * 78)
    print("CulturaQuant-v2 SMOKE QUALITY REPORT")
    print("=" * 78)
    print(f"cultural items: {len(cult)}   control items: {len(ctrl)}\n")

    for name, recs in (("CULTURAL", cult), ("CONTROL", ctrl)):
        print("#" * 78)
        print(f"# {name}")
        print("#" * 78)
        for r in recs:
            region = r.get("region", r.get("country", "-"))
            rr = r["rarity"]
            print(f"\n[{r['id']}] rel={r['relation']} region/country={region}")
            print(f"   Q: {r['input']}")
            print(fmt_options(r))
            print(f"   gold_qid={r.get('alt_qids', ['?'])[r['correct_index']]}  "
                  f"sitelinks={rr['sitelinks']}  log_pv={rr['log_pageviews']}  band={rr['band']}")
            print(f"   src={r['source_url']}")
            fl = flags(r)
            if fl:
                print(f"   !! FLAGS: {', '.join(fl)}")

    # Position balance
    print("\n" + "=" * 78)
    print("GOLD POSITION BALANCE")
    print("=" * 78)
    for name, recs in (("cultural", cult), ("control", ctrl)):
        c = Counter(LETTERS[r["correct_index"]] for r in recs)
        print(f"  {name:8s}: " + "  ".join(f"{L}={c.get(L,0)}" for L in LETTERS))

    # Rarity side-by-side + KS
    print("\n" + "=" * 78)
    print("RARITY: CULTURAL vs CONTROL (item-by-item match)")
    print("=" * 78)
    print(f"  {'cult_id':12s} {'cult_sl':7s} {'cult_pv':8s}  |  "
          f"{'ctrl_id':12s} {'ctrl_sl':7s} {'ctrl_pv':8s}  match?")
    by_match = {r.get("matched_cultural_id"): r for r in ctrl}
    for r in cult:
        m = by_match.get(r["id"])
        if m:
            same = r["rarity"]["band"] == m["rarity"]["band"]
            print(f"  {r['id']:12s} {r['rarity']['sitelinks']:<7d} "
                  f"{r['rarity']['log_pageviews']:<8.3f}  |  "
                  f"{m['id']:12s} {m['rarity']['sitelinks']:<7d} "
                  f"{m['rarity']['log_pageviews']:<8.3f}  "
                  f"{'YES' if same else 'BAND-DIFF'}")
        else:
            print(f"  {r['id']:12s} {r['rarity']['sitelinks']:<7d} "
                  f"{r['rarity']['log_pageviews']:<8.3f}  |  (no control match)")

    cs = [r["rarity"]["sitelinks"] for r in cult]
    ks_sl = [r["rarity"]["sitelinks"] for r in ctrl]
    cp = [r["rarity"]["log_pageviews"] for r in cult]
    kp = [r["rarity"]["log_pageviews"] for r in ctrl]
    d_sl, note_sl = ks_2samp([float(x) for x in cs], [float(x) for x in ks_sl])
    d_pv, note_pv = ks_2samp(cp, kp)
    print("\n  KS sitelinks  : " + note_sl)
    print("  KS log_pviews : " + note_pv)
    print(f"\n  cultural  median sitelinks={_median(cs):.1f}  median log_pv={_median(cp):.3f}")
    print(f"  control   median sitelinks={_median(ks_sl):.1f}  median log_pv={_median(kp):.3f}")

    # Region coverage
    print("\n" + "=" * 78)
    print("REGION COVERAGE (cultural)")
    print("=" * 78)
    rc = Counter(r["region"] for r in cult)
    print("  " + "  ".join(f"{k}={v}" for k, v in sorted(rc.items())))

    allflags = [(r["id"], flags(r)) for r in cult + ctrl if flags(r)]
    print("\n" + "=" * 78)
    print(f"FLAGGED ITEMS: {len(allflags)}")
    print("=" * 78)
    for i, fl in allflags:
        print(f"  {i}: {', '.join(fl)}")


def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return float("nan")
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


if __name__ == "__main__":
    report()
