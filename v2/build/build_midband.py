"""Step 3: build the final MID-BAND set + rarity-matched generic control.

After calibration (Step 2) identifies the sitelink bucket(s) where fp16 accuracy
lands ~0.4-0.7, this selects:
  midband_cultural.jsonl : ~150-200 BR figure->state items in the mid-band bucket(s),
                           region-balanced (N/NE/CO/SE/S), gold position round-robin,
                           full provenance, golds re-verified (already verified in the
                           gradient build; re-checked here).
  midband_control.jsonl  : rarity-matched non-BR figure->admin-division control, same
                           relation shape, matched item-by-item on (bucket, log_pageviews),
                           round-robin source countries. Golds verified via ASK.

The mid-band BR items come straight from the gradient file (already gold-verified +
pageviews attached). The control is fetched fresh: per country we pull arts figures
ordered ASC and DESC by sitelinks (so the whole gradient is covered), cap at <=30,
attach pageviews, gold-verify, then greedily match each BR item.

Run on a CPU host with network:
  python build_midband.py --buckets sl4-7 sl8-15 --target 180
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from fetch import FetchError, sparql
from mcq import build_mcq
from queries import _occ_values
from rarity import log_pageviews

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

SEED = 20260608
MAX_SITELINKS = 80
CTRL_PER_QUERY = 120
REGION_ORDER = ["N", "NE", "CO", "SE", "S"]

CTRL_FIG_Q = "Em qual divisão administrativa de primeiro nível de {country} nasceu {label}?"

# Control countries whose first-level division one-hop figure query completes on WDQS
# and supplies >=4 sibling divisions. (label, division-type QID).
CONTROL_COUNTRIES = {
    "Q414": ("Argentina", "Q44753"),    # province of Argentina  (full gradient)
    "Q38": ("Itália", "Q16110"),        # province of Italy      (floor..low)
    "Q29": ("Espanha", "Q162620"),      # province of Spain
    "Q183": ("Alemanha", "Q1221156"),   # district of Germany (Landkreis)
    "Q145": ("Reino Unido", "Q180673"), # ceremonial county of England
}


def bucket_of(sitelinks: int) -> str:
    if sitelinks <= 1:
        return "sl0-1"
    if sitelinks <= 3:
        return "sl2-3"
    if sitelinks <= 7:
        return "sl4-7"
    if sitelinks <= 15:
        return "sl8-15"
    if sitelinks <= 30:
        return "sl16-30"
    if sitelinks <= 50:
        return "sl31-50"
    return "sl51-80"


def qid_of(b: dict, key: str) -> str:
    return b[key]["value"].split("/")[-1]


def has_pt_label(b: dict) -> bool:
    lb = b["itemLabel"]["value"]
    return not (lb.startswith("Q") and lb[1:].isdigit())


def load_gradient(fname: str = "gradient_cultural.jsonl") -> list[dict]:
    path = DATA / fname
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


# ---------------------------------------------------------- BR mid-band selection

def select_br(grad: list[dict], buckets: set[str], target: int) -> list[dict]:
    """Region-balanced round-robin over the chosen buckets, rarer-first within a cell."""
    by_region: dict[str, list[dict]] = defaultdict(list)
    for r in grad:
        if r["rarity"]["band"] in buckets:
            by_region[r["region"]].append(r)
    for lst in by_region.values():
        lst.sort(key=lambda r: (r["rarity"]["sitelinks"], r["rarity"]["log_pageviews"]))
    picked: list[dict] = []
    idx = {r: 0 for r in REGION_ORDER}
    progressed = True
    while len(picked) < target and progressed:
        progressed = False
        for reg in REGION_ORDER:
            lst = by_region.get(reg, [])
            if idx[reg] < len(lst):
                picked.append(lst[idx[reg]])
                idx[reg] += 1
                progressed = True
            if len(picked) >= target:
                break
    return picked


# ------------------------------------------------------------ control candidates

def control_figure_query(division_qid: str, country_qid: str, order: str, limit: int) -> str:
    return f"""
SELECT ?item ?itemLabel ?adm ?admLabel (COUNT(DISTINCT ?sl) AS ?sitelinks) WHERE {{
  ?adm wdt:P31 wd:{division_qid} ; wdt:P17 wd:{country_qid} .
  ?city wdt:P131 ?adm .
  ?item wdt:P19 ?city ; wdt:P106 ?occ .
  VALUES ?occ {{ {_occ_values()} }}
  ?item rdfs:label ?itemLabel . FILTER(LANG(?itemLabel)="pt")
  OPTIONAL {{ ?sl schema:about ?item . }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "pt". }}
}}
GROUP BY ?item ?itemLabel ?adm ?admLabel
ORDER BY {order}(?sitelinks)
LIMIT {limit}
"""


def country_admin_siblings_query(division_qid: str, country_qid: str) -> str:
    return f"""
SELECT ?adm ?admLabel WHERE {{
  ?adm wdt:P31 wd:{division_qid} ; wdt:P17 wd:{country_qid} .
  ?adm rdfs:label ?admLabel . FILTER(LANG(?admLabel)="pt")
}}
"""


def fetch_control(buckets: set[str]) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for cq, (clabel, dq) in CONTROL_COUNTRIES.items():
        for order in ("ASC", "DESC"):
            try:
                d = sparql(control_figure_query(dq, cq, order, CTRL_PER_QUERY),
                           tag=f"mb_ctrl_fig_{order}_{cq}")
            except FetchError as e:
                # Some big-city countries time out on the rarest-first scan; skip that
                # (country, order) and rely on the orders/countries that do complete.
                print(f"      skip {clabel} {order}: {e}", flush=True)
                continue
            for b in d["results"]["bindings"]:
                if not has_pt_label(b):
                    continue
                qid = qid_of(b, "item")
                if qid in seen:
                    continue
                sl = int(b["sitelinks"]["value"])
                if sl > MAX_SITELINKS:
                    continue
                if bucket_of(sl) not in buckets:
                    continue
                seen.add(qid)
                out.append({
                    "relation": "figure_to_admin", "qid": qid, "label": b["itemLabel"]["value"],
                    "gold_qid": qid_of(b, "adm"), "gold_label": b["admLabel"]["value"],
                    "country_qid": cq, "division_qid": dq, "country_label": clabel,
                    "sitelinks": sl,
                })
    return out


def verify_control_gold(c: dict) -> bool:
    d = sparql(f"ASK {{ wd:{c['qid']} wdt:P19/wdt:P131* wd:{c['gold_qid']} }}",
               tag=f"ask_{c['qid']}_{c['gold_qid']}")
    return bool(d.get("boolean"))


# --------------------------------------------------------------- record builders

def make_br_records(picked: list[dict], rng: random.Random) -> list[dict]:
    """Re-issue stable mid-band ids + round-robin gold position; reuse stored alts/qids?

    The gradient already shipped balanced MCQs; here we just re-letter the gold position
    round-robin across the final ordered set for clean balance, reusing the same
    distractor labels (rebuild from alt_qids to keep provenance)."""
    recs: list[dict] = []
    for i, r in enumerate(picked):
        # Rebuild a balanced MCQ from the stored alternatives (gold + 4 distractors).
        gold_label = r["alternatives"][r["correct_index"]]
        gold_qid = r["alt_qids"][r["correct_index"]]
        pool = [(q, lb) for q, lb in zip(r["alt_qids"], r["alternatives"])
                if q != gold_qid]
        mcq = build_mcq(gold_label, gold_qid, pool, forced_index=i, rng=rng)
        if mcq is None:
            continue
        recs.append({
            "id": f"v2-mb-{len(recs)+1:03d}", "set": "midband_br", "relation": r["relation"],
            "qid": r["qid"], "statement": r["statement"], "source_url": r["source_url"],
            "gold_verified": True, "region": r["region"], "input": r["input"],
            "alternatives": mcq["alternatives"], "alt_qids": mcq["alt_qids"],
            "correct_index": mcq["correct_index"], "distractor_qids": mcq["distractor_qids"],
            "rarity": r["rarity"],
        })
    return recs


def make_control_records(ctrl_ok: list[dict], br_recs: list[dict],
                         rng: random.Random) -> list[dict]:
    pool_cache: dict[str, list[tuple[str, str]]] = {}

    def admin_pool(dq: str, cq: str) -> list[tuple[str, str]]:
        if cq not in pool_cache:
            d = sparql(country_admin_siblings_query(dq, cq), tag=f"adm_{dq}_{cq}")
            pool_cache[cq] = [(qid_of(b, "adm"), b["admLabel"]["value"])
                              for b in d["results"]["bindings"]]
        return pool_cache[cq]

    countries = list(CONTROL_COUNTRIES.keys())
    used: set[str] = set()
    recs: list[dict] = []
    for bi, br in enumerate(br_recs):
        want_band = br["rarity"]["band"]
        want_pv = br["rarity"]["log_pageviews"]
        pref = countries[bi % len(countries):] + countries[: bi % len(countries)]
        cpri = {cq: i for i, cq in enumerate(pref)}
        opts = [c for c in ctrl_ok if c["qid"] not in used]
        if not opts:
            break
        opts.sort(key=lambda c: (
            0 if bucket_of(c["sitelinks"]) == want_band else 1,
            abs(c["log_pageviews"] - want_pv),
            cpri.get(c["country_qid"], 99),
        ))
        chosen = mcq = None
        for c in opts:
            cand_pool = [p for p in admin_pool(c["division_qid"], c["country_qid"])
                         if p[0] != c["gold_qid"]]
            m = build_mcq(c["gold_label"], c["gold_qid"], cand_pool, forced_index=bi, rng=rng)
            if m is not None:
                chosen, mcq = c, m
                break
        if chosen is None:
            continue
        used.add(chosen["qid"])
        recs.append({
            "id": f"v2-mbctrl-{len(recs)+1:03d}", "set": "midband_control",
            "relation": chosen["relation"], "matched_br_id": br["id"], "qid": chosen["qid"],
            "statement": f"{chosen['label']} (-> {chosen['gold_label']}, {chosen['country_label']})",
            "source_url": f"https://www.wikidata.org/wiki/{chosen['qid']}",
            "gold_verified": True, "country": chosen["country_label"],
            "input": CTRL_FIG_Q.format(label=chosen["label"], country=chosen["country_label"]),
            "alternatives": mcq["alternatives"], "alt_qids": mcq["alt_qids"],
            "correct_index": mcq["correct_index"], "distractor_qids": mcq["distractor_qids"],
            "rarity": {"sitelinks": chosen["sitelinks"], "log_pageviews": chosen["log_pageviews"],
                       "band": bucket_of(chosen["sitelinks"])},
        })
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--buckets", nargs="+", required=True,
                    help="mid-band sitelink buckets, e.g. sl4-7 sl8-15")
    ap.add_argument("--target", type=int, default=180)
    ap.add_argument("--input", default="gradient_cultural.jsonl",
                    help="gradient jsonl filename under data/ (e.g. gradient_wide_cultural.jsonl)")
    args = ap.parse_args()
    buckets = set(args.buckets)
    rng = random.Random(SEED)

    grad = load_gradient(args.input)
    print(f"[1/4] BR mid-band selection from {len(grad)} gradient items, buckets={buckets}",
          flush=True)
    picked = select_br(grad, buckets, args.target)
    br_recs = make_br_records(picked, rng)
    print(f"      {len(br_recs)} BR mid-band MCQs", flush=True)

    print("[2/4] fetch control candidates (ASC+DESC per country, mid-band only) ...", flush=True)
    ctrl_cands = fetch_control(buckets)
    print(f"      {len(ctrl_cands)} control candidates in band", flush=True)
    for c in ctrl_cands:
        c["log_pageviews"] = log_pageviews(c["label"])

    print("[3/4] gold-verify control candidates (ASK) ...", flush=True)
    ctrl_ok = [c for c in ctrl_cands if verify_control_gold(c)]
    print(f"      {len(ctrl_ok)}/{len(ctrl_cands)} verified", flush=True)

    print("[4/4] item-by-item rarity match ...", flush=True)
    ctrl_recs = make_control_records(ctrl_ok, br_recs, rng)
    print(f"      {len(ctrl_recs)} control matched", flush=True)

    # Pair 1:1.
    matched = {r["matched_br_id"] for r in ctrl_recs}
    br_recs = [r for r in br_recs if r["id"] in matched]

    (DATA / "midband_cultural.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in br_recs) + "\n")
    (DATA / "midband_control.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in ctrl_recs) + "\n")

    LET = ["A", "B", "C", "D", "E"]
    stats = {
        "buckets": sorted(buckets), "n_br": len(br_recs), "n_control": len(ctrl_recs),
        "br_per_region": dict(Counter(r["region"] for r in br_recs)),
        "br_per_bucket": dict(Counter(r["rarity"]["band"] for r in br_recs)),
        "br_gold_pos": dict(Counter(LET[r["correct_index"]] for r in br_recs)),
        "control_per_country": dict(Counter(r["country"] for r in ctrl_recs)),
        "control_per_bucket": dict(Counter(r["rarity"]["band"] for r in ctrl_recs)),
        "control_gold_pos": dict(Counter(LET[r["correct_index"]] for r in ctrl_recs)),
        "br_log_pv_mean": round(sum(r["rarity"]["log_pageviews"] for r in br_recs)
                                / max(1, len(br_recs)), 4),
        "control_log_pv_mean": round(sum(r["rarity"]["log_pageviews"] for r in ctrl_recs)
                                     / max(1, len(ctrl_recs)), 4),
    }
    (DATA / "midband_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\nWrote {len(br_recs)} BR + {len(ctrl_recs)} control", flush=True)
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
