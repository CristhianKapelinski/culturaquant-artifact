"""CulturaQuant-v2 WIDE rarity-gradient build (mid-band power expansion).

gradient.py capped at 30 sitelinks and queried figures only, yielding just ~48
items in the sl16-30 band (fp16 ~0.42). That is a clean mid-band but far too few
for statistical power. This wide build:
  - raises the ceiling to 80 sitelinks (the sl31-80 band is the *easier* mid-band,
    fp16 ~0.45-0.65, and is densely populated by the most-linked arts figures);
  - adds festival_to_state (a genuinely cultural relation, not just birthplace),
    DESC by sitelinks, to diversify beyond a single relation;
  - buckets 0-1 / 2-3 / 4-7 / 8-15 / 16-30 / 31-50 / 51-80 so calibration can pick
    the exact fp16 0.4-0.65 window.

Region-balanced (bucket x region round-robin), gold re-verified (ASK). Writes
v2/data/gradient_wide_cultural.jsonl. Run locally / on a host with network (ssh-curl).
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from fetch import sparql
from mcq import build_mcq
from queries import BR_STATES_QUERY, STATE_REGION, _occ_values
from rarity import log_pageviews

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

SEED = 20260609
MAX_SITELINKS = 80
FIGURE_ASC_PER_STATE = 40
FIGURE_DESC_PER_STATE = 200   # the mid/high band lives here (most-linked artists)
FESTIVAL_LIMIT = 600
TARGET = 700

REGION_ORDER = ["N", "NE", "CO", "SE", "S"]
FIG_Q = "Em qual estado brasileiro nasceu {label}?"
FEST_Q = "Em qual estado brasileiro acontece o {label}?"


def bucket_of(sl: int) -> str:
    if sl <= 1: return "sl0-1"
    if sl <= 3: return "sl2-3"
    if sl <= 7: return "sl4-7"
    if sl <= 15: return "sl8-15"
    if sl <= 30: return "sl16-30"
    if sl <= 50: return "sl31-50"
    return "sl51-80"


BUCKETS = ["sl0-1", "sl2-3", "sl4-7", "sl8-15", "sl16-30", "sl31-50", "sl51-80"]


def qid_of(b, k): return b[k]["value"].split("/")[-1]
def has_pt_label(b):
    lb = b["itemLabel"]["value"]; return not (lb.startswith("Q") and lb[1:].isdigit())


def br_state_pool():
    d = sparql(BR_STATES_QUERY, tag="br_states")
    return [(qid_of(b, "state"), b["stateLabel"]["value"]) for b in d["results"]["bindings"]]


def figure_query(state_qid, order, limit):
    return f"""
SELECT ?item ?itemLabel (COUNT(DISTINCT ?sl) AS ?sitelinks) WHERE {{
  VALUES ?occ {{ {_occ_values()} }}
  ?item wdt:P106 ?occ ; wdt:P19 ?city .
  ?city wdt:P131* wd:{state_qid} .
  ?item rdfs:label ?itemLabel . FILTER(LANG(?itemLabel)="pt")
  OPTIONAL {{ ?sl schema:about ?item . }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "pt". }}
}}
GROUP BY ?item ?itemLabel
ORDER BY {order}(?sitelinks)
LIMIT {limit}
"""


def festival_query(limit):
    """BR festivals -> state, most-linked first (no cap; drop >MAX at selection)."""
    return f"""
SELECT ?item ?itemLabel ?state ?stateLabel (COUNT(DISTINCT ?sl) AS ?sitelinks) WHERE {{
  ?item wdt:P31/wdt:P279* wd:Q132241 ; wdt:P131* ?state .
  ?state wdt:P31 wd:Q485258 .
  ?item rdfs:label ?itemLabel . FILTER(LANG(?itemLabel)="pt")
  OPTIONAL {{ ?sl schema:about ?item . }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "pt". }}
}}
GROUP BY ?item ?itemLabel ?state ?stateLabel
ORDER BY DESC(COUNT(DISTINCT ?sl))
LIMIT {limit}
"""


def fetch_candidates():
    out, seen = [], set()
    state_label = {q: lb for q, lb in br_state_pool()}
    # figures
    for st, region in STATE_REGION.items():
        for order, lim, tag in (("ASC", FIGURE_ASC_PER_STATE, f"wgrad_fig_asc_{st}"),
                                 ("DESC", FIGURE_DESC_PER_STATE, f"wgrad_fig_desc_{st}")):
            d = sparql(figure_query(st, order, lim), tag=tag)
            for b in d["results"]["bindings"]:
                if not has_pt_label(b): continue
                qid = qid_of(b, "item")
                if qid in seen: continue
                sl = int(b["sitelinks"]["value"])
                if sl > MAX_SITELINKS: continue
                seen.add(qid)
                out.append({"relation": "figure_to_state", "qid": qid,
                            "label": b["itemLabel"]["value"], "gold_qid": st,
                            "gold_label": state_label.get(st, "?"), "region": region,
                            "sitelinks": sl})
    # festivals (global)
    d = sparql(festival_query(FESTIVAL_LIMIT), tag="wgrad_festival")
    for b in d["results"]["bindings"]:
        if not has_pt_label(b): continue
        qid = qid_of(b, "item")
        if qid in seen: continue
        st = qid_of(b, "state"); region = STATE_REGION.get(st)
        if region is None: continue
        sl = int(b["sitelinks"]["value"])
        if sl > MAX_SITELINKS: continue
        seen.add(qid)
        out.append({"relation": "festival_to_state", "qid": qid,
                    "label": b["itemLabel"]["value"], "gold_qid": st,
                    "gold_label": b["stateLabel"]["value"], "region": region, "sitelinks": sl})
    return out


def verify_gold(c):
    qid, gold = c["qid"], c["gold_qid"]
    body = (f"wd:{qid} wdt:P19/wdt:P131* wd:{gold}" if c["relation"].startswith("figure")
            else f"wd:{qid} wdt:P131* wd:{gold}")
    d = sparql(f"ASK {{ {body} }}", tag=f"ask_{qid}_{gold}")
    return bool(d.get("boolean"))


def select_gradient(cands, target):
    cells = defaultdict(list)
    for c in cands:
        cells[(bucket_of(c["sitelinks"]), c["region"])].append(c)
    for lst in cells.values():
        lst.sort(key=lambda c: (c["sitelinks"], c["log_pageviews"]))
    picked, idx = [], defaultdict(int)
    order = [(b, r) for b in BUCKETS for r in REGION_ORDER]
    progressed = True
    while len(picked) < target and progressed:
        progressed = False
        for cell in order:
            lst = cells.get(cell, [])
            if idx[cell] < len(lst):
                picked.append(lst[idx[cell]]); idx[cell] += 1; progressed = True
            if len(picked) >= target: break
    return picked


def make_records(cands, rng):
    pool = br_state_pool(); recs = []
    for i, c in enumerate(cands):
        mcq = build_mcq(c["gold_label"], c["gold_qid"], pool, forced_index=i, rng=rng)
        if mcq is None: continue
        relw = "P131*" if c["relation"] == "festival_to_state" else "P19/P131*"
        q = FEST_Q if c["relation"] == "festival_to_state" else FIG_Q
        recs.append({"id": f"v2-wgrad-{len(recs)+1:03d}", "set": "gradient_wide",
                     "relation": c["relation"], "qid": c["qid"],
                     "statement": f"{c['label']} ({relw} -> {c['gold_label']})",
                     "source_url": f"https://www.wikidata.org/wiki/{c['qid']}",
                     "gold_verified": True, "region": c["region"],
                     "input": q.format(label=c["label"]),
                     "alternatives": mcq["alternatives"], "alt_qids": mcq["alt_qids"],
                     "correct_index": mcq["correct_index"],
                     "distractor_qids": mcq["distractor_qids"],
                     "rarity": {"sitelinks": c["sitelinks"], "log_pageviews": c["log_pageviews"],
                                "band": bucket_of(c["sitelinks"])}})
    return recs


def main():
    rng = random.Random(SEED); stats = {}
    print("[1/4] fetch wide BR gradient (figures cap 80 + festivals) ...", flush=True)
    cands = fetch_candidates()
    print(f"      {len(cands)} candidates  buckets={dict(Counter(bucket_of(c['sitelinks']) for c in cands))}", flush=True)
    print("[2/4] pageviews ...", flush=True)
    for c in cands: c["log_pageviews"] = log_pageviews(c["label"])
    print("[3/4] (bucket x region) round-robin selection ...", flush=True)
    selected = select_gradient(cands, int(TARGET * 1.4))
    print(f"      {len(selected)} pre-verify", flush=True)
    print("[4/4] gold ASK verify ...", flush=True)
    verified = []
    for c in selected:
        if verify_gold(c): verified.append(c)
        if len(verified) >= TARGET: break
    print(f"      verified={len(verified)}", flush=True)
    recs = make_records(verified, rng)
    stats["n"] = len(recs)
    stats["per_bucket"] = dict(Counter(r["rarity"]["band"] for r in recs))
    stats["per_region"] = dict(Counter(r["region"] for r in recs))
    stats["per_relation"] = dict(Counter(r["relation"] for r in recs))
    out = DATA / "gradient_wide_cultural.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n")
    (DATA / "gradient_wide_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\nWrote {len(recs)} -> {out}\n  per_bucket={stats['per_bucket']}\n  per_relation={stats['per_relation']}", flush=True)


if __name__ == "__main__":
    main()
