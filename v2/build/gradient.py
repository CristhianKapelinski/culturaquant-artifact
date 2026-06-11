"""CulturaQuant-v2 RARITY-GRADIENT build (Step 1 of the mid-band re-tune).

The sitelinks<=5 build floored fp16 accuracy at chance (~0.2): the items the model
holds at all are too rare to erode. To FIND the mid-band we first need a calibration
set that SPANS popularity, so fp16 accuracy can be measured as a function of rarity.

This module re-queries Wikidata (via the cached ssh-curl fetcher) for Brazilian
cultural entities across a sitelink GRADIENT 0..~30:
  figure_to_state : BR arts figures (musician/singer/composer/writer/painter/poet/
                    sculptor) -> birthplace state, queried PER state for region balance.
  festival_to_state: BR festivals -> the state they are held in.
Each candidate carries its rarity (sitelinks, log10 pt-wiki pageviews) and is bucketed
0-1 / 2-3 / 4-7 / 8-15 / 16-30. Gold is re-verified with an ASK query. ~300-400
candidates, region-balanced across N/NE/CO/SE/S, are written as MCQs to
v2/data/gradient_cultural.jsonl so the fp16 gate can score per-bucket accuracy and
identify where fp16 lands ~0.4-0.7.

Run on a CPU host with network:  python gradient.py
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

SEED = 20260608
MIN_SITELINKS = 0
MAX_SITELINKS = 30        # gradient ceiling (we drop >30 at selection time)
FIGURE_ASC_PER_STATE = 60   # rare floor: rarest figures per state (asc)
FIGURE_DESC_PER_STATE = 120  # mid/high band: most-linked figures per state (desc)
TARGET = 360              # gradient candidate target (design: 300-400)

REGION_ORDER = ["N", "NE", "CO", "SE", "S"]

FIG_Q = "Em qual estado brasileiro nasceu {label}?"
FEST_Q = "Em qual estado brasileiro acontece o {label}?"


def bucket_of(sitelinks: int) -> str:
    if sitelinks <= 1:
        return "sl0-1"
    if sitelinks <= 3:
        return "sl2-3"
    if sitelinks <= 7:
        return "sl4-7"
    if sitelinks <= 15:
        return "sl8-15"
    return "sl16-30"


BUCKETS = ["sl0-1", "sl2-3", "sl4-7", "sl8-15", "sl16-30"]


def qid_of(b: dict, key: str) -> str:
    return b[key]["value"].split("/")[-1]


def has_pt_label(b: dict) -> bool:
    lb = b["itemLabel"]["value"]
    return not (lb.startswith("Q") and lb[1:].isdigit())


def br_state_pool() -> list[tuple[str, str]]:
    d = sparql(BR_STATES_QUERY, tag="br_states")
    return [(qid_of(b, "state"), b["stateLabel"]["value"]) for b in d["results"]["bindings"]]


# -- gradient SPARQL: a sitelink BAND (min..max), reporting the actual count ----

def figure_query(state_qid: str, order: str, limit: int) -> str:
    """BR arts figures born in a state, ordered by sitelink count.

    order='ASC'  -> rarest first (the long-tail floor; sitelinks 0..1 dominate).
    order='DESC' -> most-linked first (captures the 2..30 mid/high band that an
                    ascending+capped query never reaches). No sitelink cap: capping
                    the transitive P131* scan is what times out on WDQS; instead we
                    pull the most-linked and DROP >30 in Python at selection time.
    """
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


def fetch_candidates() -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    state_label = {q: lb for q, lb in br_state_pool()}

    # Figures -> birthplace state, per state. ASC pulls the rare floor (sitelinks 0-1),
    # DESC pulls the mid/high band (sitelinks ~2..79). Together they span the gradient.
    for st, region in STATE_REGION.items():
        for order, lim, tag in (
            ("ASC", FIGURE_ASC_PER_STATE, f"grad_fig_asc_{st}"),
            ("DESC", FIGURE_DESC_PER_STATE, f"grad_fig_desc_{st}"),
        ):
            d = sparql(figure_query(st, order, lim), tag=tag)
            for b in d["results"]["bindings"]:
                if not has_pt_label(b):
                    continue
                qid = qid_of(b, "item")
                if qid in seen:
                    continue
                sl = int(b["sitelinks"]["value"])
                if sl > MAX_SITELINKS:
                    continue  # cap the gradient at 30 (famous = ceiling, excluded)
                seen.add(qid)
                out.append({
                    "relation": "figure_to_state", "qid": qid, "label": b["itemLabel"]["value"],
                    "gold_qid": st, "gold_label": state_label.get(st, "?"), "region": region,
                    "sitelinks": sl,
                })
    return out


def verify_gold(c: dict) -> bool:
    qid, gold = c["qid"], c["gold_qid"]
    if c["relation"].startswith("figure"):
        body = f"wd:{qid} wdt:P19/wdt:P131* wd:{gold}"
    else:
        body = f"wd:{qid} wdt:P131* wd:{gold}"
    d = sparql(f"ASK {{ {body} }}", tag=f"ask_{qid}_{gold}")
    return bool(d.get("boolean"))


def select_gradient(cands: list[dict], target: int) -> list[dict]:
    """Fill (bucket x region) cells round-robin so every rarity bucket is populated
    in every macro-region; within a cell prefer the rarer / lower-pageview item."""
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for c in cands:
        cells[(bucket_of(c["sitelinks"]), c["region"])].append(c)
    for lst in cells.values():
        lst.sort(key=lambda c: (c["sitelinks"], c["log_pageviews"]))
    picked: list[dict] = []
    idx: dict[tuple[str, str], int] = defaultdict(int)
    # Round-robin over (bucket, region) until target or exhausted.
    order = [(b, r) for b in BUCKETS for r in REGION_ORDER]
    progressed = True
    while len(picked) < target and progressed:
        progressed = False
        for cell in order:
            lst = cells.get(cell, [])
            if idx[cell] < len(lst):
                picked.append(lst[idx[cell]])
                idx[cell] += 1
                progressed = True
            if len(picked) >= target:
                break
    return picked


def make_records(cands: list[dict], rng: random.Random) -> list[dict]:
    pool = br_state_pool()
    recs: list[dict] = []
    for i, c in enumerate(cands):
        mcq = build_mcq(c["gold_label"], c["gold_qid"], pool, forced_index=i, rng=rng)
        if mcq is None:
            continue
        relw = "P131*" if c["relation"] == "festival_to_state" else "P19/P131*"
        q = FEST_Q if c["relation"] == "festival_to_state" else FIG_Q
        recs.append({
            "id": f"v2-grad-{len(recs)+1:03d}", "set": "gradient", "relation": c["relation"],
            "qid": c["qid"],
            "statement": f"{c['label']} ({relw} -> {c['gold_label']})",
            "source_url": f"https://www.wikidata.org/wiki/{c['qid']}",
            "gold_verified": True, "region": c["region"],
            "input": q.format(label=c["label"]),
            "alternatives": mcq["alternatives"], "alt_qids": mcq["alt_qids"],
            "correct_index": mcq["correct_index"], "distractor_qids": mcq["distractor_qids"],
            "rarity": {"sitelinks": c["sitelinks"], "log_pageviews": c["log_pageviews"],
                       "band": bucket_of(c["sitelinks"])},
        })
    return recs


def main() -> None:
    rng = random.Random(SEED)
    stats: dict = {}

    print("[1/4] fetch BR gradient candidates (sitelinks 0..30) ...", flush=True)
    cands = fetch_candidates()
    print(f"      {len(cands)} candidates", flush=True)
    print("      raw bucket dist:", dict(Counter(bucket_of(c["sitelinks"]) for c in cands)),
          flush=True)

    print("[2/4] pageviews (pt.wiki 2024 monthly log10) ...", flush=True)
    for c in cands:
        c["log_pageviews"] = log_pageviews(c["label"])

    print("[3/4] (bucket x region) round-robin selection ...", flush=True)
    selected = select_gradient(cands, int(TARGET * 1.5))  # over-select for ASK drops

    print("[4/4] gold re-verification (ASK per item) ...", flush=True)
    verified = []
    for c in selected:
        if verify_gold(c):
            verified.append(c)
        if len(verified) >= TARGET:
            break
    print(f"      verified={len(verified)}", flush=True)

    recs = make_records(verified, rng)

    stats["n"] = len(recs)
    stats["per_bucket"] = dict(Counter(r["rarity"]["band"] for r in recs))
    stats["per_region"] = dict(Counter(r["region"] for r in recs))
    stats["per_relation"] = dict(Counter(r["relation"] for r in recs))
    stats["bucket_x_region"] = {
        b: dict(Counter(r["region"] for r in recs if r["rarity"]["band"] == b))
        for b in BUCKETS
    }

    out = DATA / "gradient_cultural.jsonl"
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in recs) + "\n")
    (DATA / "gradient_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    print(f"\nWrote {len(recs)} -> {out}", flush=True)
    print("per_bucket:", stats["per_bucket"], flush=True)
    print("per_region:", stats["per_region"], flush=True)
    print("bucket_x_region:", json.dumps(stats["bucket_x_region"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
