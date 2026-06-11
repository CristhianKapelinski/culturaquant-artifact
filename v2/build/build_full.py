"""CulturaQuant-v2 FULL build: scaled, region-stratified, gold-verified.

Produces the two rarity-matched item sets used by the fp16 difficulty-match gate:
  data/br_rare.jsonl        -- rare Brazil-specific entity knowledge
  data/control_matched.jsonl -- rarity-matched non-Brazil control (round-robin countries)

Pipeline (every network call cached via build.fetch, so fully resumable/idempotent):
  1. BR-rare candidates
       figure_to_state : long-tail BR arts figures, queried PER state (loop all 27 QIDs,
                         ORDER BY sitelinks) so coverage is region-balanced by construction.
       festival_to_state: long-tail BR festivals (one global query), region-tagged via P131*.
       sitelinks <= MAX_SITELINKS (5, the relaxed cultural long-tail filter).
  2. pageviews (pt.wikipedia, 2024 monthly avg, log10) for every candidate.
  3. Region-stratified selection: round-robin over N/NE/CO/SE/S, rarest-first within each
     region, target TARGET_BR total with >= MIN_PER_REGION per macro-region.
  4. GOLD RE-VERIFICATION: an ASK query per selected item confirms the P131* (festival) or
     P19/P131* (figure) chain actually reaches the claimed state. Items that fail are DROPPED.
  5. Control candidates: same two shapes for a round-robin set of non-BR countries
     (France/Italy/Spain/Argentina/Germany/UK/Portugal/Mexico). pageviews + gold ASK as above.
  6. Item-by-item rarity match: each surviving BR item is paired to an unused control item of
     the same shape, nearest on (sitelink band, |log_pageviews|), cycling source countries so
     the control is not all-one-country. KS check (sitelinks AND log-pageviews) reported.
  7. MCQ: 4 sibling distractors (other BR states / other divisions of the same country),
     gold position round-robin A..E across the final ordered set.

Run:  python build_full.py            (resumes from cache; writes data/*.jsonl + build_stats.json)
"""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from fetch import sparql
from mcq import build_mcq
from queries import (
    BR_STATES_QUERY,
    STATE_REGION,
    country_admin_siblings_query,
    cultural_festival_query,
    cultural_figure_query,
    _occ_values,
)
from rarity import log_pageviews, rarity_band

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

SEED = 20260608
MAX_SITELINKS = 5
TARGET_BR = 400          # final BR-rare target (design: 300-500)
MIN_PER_REGION = 60      # design: >= 60 per macro-region
FIGURE_PER_STATE = 60    # candidates pulled per state before filtering
FESTIVAL_LIMIT = 400     # festival candidates pulled (all states)
CONTROL_PER_QUERY = 120  # control candidates pulled per (relation, country)

REGION_ORDER = ["N", "NE", "CO", "SE", "S"]

# Round-robin control source countries: (label, first-level-division-type QID).
# Each has a rich (>=4) first-level division set for sibling distractors and pt labels.
CONTROL_COUNTRIES = {
    "Q142": ("França", "Q6465"),       # department of France
    "Q38": ("Itália", "Q16110"),       # province of Italy
    "Q29": ("Espanha", "Q162620"),     # province of Spain
    "Q414": ("Argentina", "Q44753"),   # province of Argentina
    "Q183": ("Alemanha", "Q1221156"),  # district of Germany (Landkreis)
    "Q145": ("Reino Unido", "Q180673"), # ceremonial county of England
    "Q45": ("Portugal", "Q3776527"),   # district of Portugal
    "Q96": ("México", "Q856612"),      # state of Mexico
}

FIG_Q = "Em qual estado brasileiro nasceu {label}?"
FEST_Q = "Em qual estado brasileiro acontece o {label}?"
CTRL_FIG_Q = "Em qual divisão administrativa de primeiro nível de {country} nasceu {label}?"
CTRL_FEST_Q = (
    "Em qual divisão administrativa de primeiro nível de {country} acontece o {label}?"
)


def qid_of(b: dict, key: str) -> str:
    return b[key]["value"].split("/")[-1]


def has_pt_label(b: dict) -> bool:
    lb = b["itemLabel"]["value"]
    return not (lb.startswith("Q") and lb[1:].isdigit())


def br_state_pool() -> list[tuple[str, str]]:
    d = sparql(BR_STATES_QUERY, tag="br_states")
    return [(qid_of(b, "state"), b["stateLabel"]["value"]) for b in d["results"]["bindings"]]


# ----- control query builders (district-anchored, like the smoke control) -----

def control_festival_query(division_qid: str, country_qid: str, limit: int) -> str:
    return f"""
SELECT ?item ?itemLabel ?adm ?admLabel (COUNT(DISTINCT ?sl) AS ?sitelinks) WHERE {{
  ?item wdt:P31/wdt:P279* wd:Q132241 ; wdt:P131 ?adm .
  ?adm wdt:P31 wd:{division_qid} ; wdt:P17 wd:{country_qid} .
  ?item rdfs:label ?itemLabel . FILTER(LANG(?itemLabel)="pt")
  FILTER NOT EXISTS {{ ?item wikibase:sitelinks ?sx . FILTER(?sx > {MAX_SITELINKS}) }}
  OPTIONAL {{ ?sl schema:about ?item . }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "pt". }}
}}
GROUP BY ?item ?itemLabel ?adm ?admLabel
HAVING (COUNT(DISTINCT ?sl) <= {MAX_SITELINKS})
ORDER BY ?sitelinks
LIMIT {limit}
"""


def control_figure_query(division_qid: str, country_qid: str, limit: int) -> str:
    return f"""
SELECT ?item ?itemLabel ?adm ?admLabel (COUNT(DISTINCT ?sl) AS ?sitelinks) WHERE {{
  ?adm wdt:P31 wd:{division_qid} ; wdt:P17 wd:{country_qid} .
  ?city wdt:P131 ?adm .
  ?item wdt:P19 ?city ; wdt:P106 ?occ .
  VALUES ?occ {{ {_occ_values()} }}
  ?item rdfs:label ?itemLabel . FILTER(LANG(?itemLabel)="pt")
  FILTER NOT EXISTS {{ ?item wikibase:sitelinks ?sx . FILTER(?sx > {MAX_SITELINKS}) }}
  OPTIONAL {{ ?sl schema:about ?item . }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "pt". }}
}}
GROUP BY ?item ?itemLabel ?adm ?admLabel
ORDER BY ?sitelinks
LIMIT {limit}
"""


# ----------------------------------------------------------- BR candidate fetch

def fetch_br_candidates() -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    state_label = {q: lb for q, lb in br_state_pool()}

    # Festivals -> state (global), region-tagged.
    d = sparql(cultural_festival_query(MAX_SITELINKS, FESTIVAL_LIMIT), tag="cult_festival_full")
    for b in d["results"]["bindings"]:
        if not has_pt_label(b):
            continue
        qid = qid_of(b, "item")
        if qid in seen:
            continue
        st = qid_of(b, "state")
        region = STATE_REGION.get(st)
        if region is None:
            continue
        seen.add(qid)
        out.append({
            "relation": "festival_to_state", "qid": qid, "label": b["itemLabel"]["value"],
            "gold_qid": st, "gold_label": b["stateLabel"]["value"], "region": region,
            "sitelinks": int(b["sitelinks"]["value"]),
        })

    # Figures -> birthplace state, queried per state (all 27) for region balance.
    for st, region in STATE_REGION.items():
        d = sparql(cultural_figure_query(st, MAX_SITELINKS, FIGURE_PER_STATE), tag=f"cult_fig_{st}")
        for b in d["results"]["bindings"]:
            if not has_pt_label(b):
                continue
            qid = qid_of(b, "item")
            if qid in seen:
                continue
            seen.add(qid)
            out.append({
                "relation": "figure_to_state", "qid": qid, "label": b["itemLabel"]["value"],
                "gold_qid": st, "gold_label": state_label.get(st, "?"), "region": region,
                "sitelinks": int(b["sitelinks"]["value"]),
            })
    return out


def fetch_control_candidates() -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for cq, (clabel, dq) in CONTROL_COUNTRIES.items():
        for rel, builder in (
            ("festival_to_admin", control_festival_query),
            ("figure_to_admin", control_figure_query),
        ):
            d = sparql(builder(dq, cq, CONTROL_PER_QUERY), tag=f"ctrl_{rel}_{cq}")
            for b in d["results"]["bindings"]:
                if not has_pt_label(b):
                    continue
                qid = qid_of(b, "item")
                if qid in seen:
                    continue
                seen.add(qid)
                out.append({
                    "relation": rel, "qid": qid, "label": b["itemLabel"]["value"],
                    "gold_qid": qid_of(b, "adm"), "gold_label": b["admLabel"]["value"],
                    "country_qid": cq, "division_qid": dq, "country_label": clabel,
                    "sitelinks": int(b["sitelinks"]["value"]),
                })
    return out


# ------------------------------------------------------------- gold verification

def verify_gold(c: dict) -> bool:
    """ASK that the entity's admin chain reaches the claimed gold division.

    figure*: P19/P131* ; festival*: P131*. Cached per (qid, gold)."""
    qid, gold = c["qid"], c["gold_qid"]
    if c["relation"].startswith("figure"):
        body = f"wd:{qid} wdt:P19/wdt:P131* wd:{gold}"
    else:
        body = f"wd:{qid} wdt:P131* wd:{gold}"
    d = sparql(f"ASK {{ {body} }}", tag=f"ask_{qid}_{gold}")
    return bool(d.get("boolean"))


# ------------------------------------------------------------ pageviews + select

def attach_pageviews(items: list[dict]) -> None:
    for it in items:
        it["log_pageviews"] = log_pageviews(it["label"])


def select_region_stratified(cands: list[dict], target: int) -> list[dict]:
    by_region: dict[str, list[dict]] = defaultdict(list)
    for c in cands:
        by_region[c["region"]].append(c)
    for lst in by_region.values():
        lst.sort(key=lambda c: (c["sitelinks"], c["log_pageviews"]))
    picked: list[dict] = []
    idx = {r: 0 for r in REGION_ORDER}
    while len(picked) < target and any(
        idx[r] < len(by_region.get(r, [])) for r in REGION_ORDER
    ):
        for r in REGION_ORDER:
            lst = by_region.get(r, [])
            if idx[r] < len(lst):
                picked.append(lst[idx[r]])
                idx[r] += 1
            if len(picked) >= target:
                break
    return picked


# ----------------------------------------------------------------- record build

def make_br_records(cands: list[dict], rng: random.Random) -> list[dict]:
    pool = br_state_pool()
    recs: list[dict] = []
    for i, c in enumerate(cands):
        mcq = build_mcq(c["gold_label"], c["gold_qid"], pool, forced_index=i, rng=rng)
        if mcq is None:
            continue
        relw = "P131*" if c["relation"] == "festival_to_state" else "P19/P131*"
        q = FEST_Q if c["relation"] == "festival_to_state" else FIG_Q
        recs.append({
            "id": f"v2-br-{len(recs)+1:03d}", "set": "br_rare", "relation": c["relation"],
            "qid": c["qid"],
            "statement": f"{c['label']} ({relw} -> {c['gold_label']})",
            "source_url": f"https://www.wikidata.org/wiki/{c['qid']}",
            "gold_verified": True, "region": c["region"],
            "input": q.format(label=c["label"]),
            "alternatives": mcq["alternatives"], "alt_qids": mcq["alt_qids"],
            "correct_index": mcq["correct_index"], "distractor_qids": mcq["distractor_qids"],
            "rarity": {"sitelinks": c["sitelinks"], "log_pageviews": c["log_pageviews"],
                       "band": rarity_band(c["sitelinks"])},
        })
    return recs


def make_control_records(
    cands: list[dict], br_recs: list[dict], rng: random.Random
) -> list[dict]:
    """Item-by-item rarity match, round-robin source country.

    For each BR item, among unused control candidates of the same shape, pick the one with
    the matching sitelink band and nearest log-pageviews; break country-monoculture by
    rotating preferred country order per BR item.
    """
    pool_cache: dict[str, list[tuple[str, str]]] = {}

    def admin_pool(dq: str, cq: str) -> list[tuple[str, str]]:
        if cq not in pool_cache:
            d = sparql(country_admin_siblings_query(dq, cq), tag=f"adm_{dq}_{cq}")
            pool_cache[cq] = [(qid_of(b, "adm"), b["admLabel"]["value"])
                              for b in d["results"]["bindings"]]
        return pool_cache[cq]

    rel_map = {"festival_to_state": "festival_to_admin", "figure_to_state": "figure_to_admin"}
    countries = list(CONTROL_COUNTRIES.keys())
    used: set[str] = set()
    recs: list[dict] = []
    for bi, br in enumerate(br_recs):
        want_rel = rel_map[br["relation"]]
        want_band = br["rarity"]["band"]
        want_pv = br["rarity"]["log_pageviews"]
        # Rotate country preference so consecutive matches favor different countries.
        pref = countries[bi % len(countries):] + countries[: bi % len(countries)]
        cpri = {cq: i for i, cq in enumerate(pref)}
        opts = [c for c in cands if c["relation"] == want_rel and c["qid"] not in used]
        if not opts:
            opts = [c for c in cands if c["qid"] not in used]
        if not opts:
            break
        opts.sort(key=lambda c: (
            0 if rarity_band(c["sitelinks"]) == want_band else 1,
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
            "id": f"v2-ctrl-{len(recs)+1:03d}", "set": "control",
            "relation": chosen["relation"], "matched_br_id": br["id"], "qid": chosen["qid"],
            "statement": f"{chosen['label']} (-> {chosen['gold_label']}, {chosen['country_label']})",
            "source_url": f"https://www.wikidata.org/wiki/{chosen['qid']}",
            "gold_verified": True, "country": chosen["country_label"],
            "input": (CTRL_FEST_Q if chosen["relation"] == "festival_to_admin" else CTRL_FIG_Q)
                .format(label=chosen["label"], country=chosen["country_label"]),
            "alternatives": mcq["alternatives"], "alt_qids": mcq["alt_qids"],
            "correct_index": mcq["correct_index"], "distractor_qids": mcq["distractor_qids"],
            "rarity": {"sitelinks": chosen["sitelinks"], "log_pageviews": chosen["log_pageviews"],
                       "band": rarity_band(chosen["sitelinks"])},
        })
    return recs


def main() -> None:
    rng = random.Random(SEED)
    stats: dict = {}

    print("[1/7] fetch BR candidates (festivals + per-state figures) ...", flush=True)
    br_cands = fetch_br_candidates()
    print(f"      {len(br_cands)} BR candidates", flush=True)
    stats["br_candidates"] = len(br_cands)

    print("[2/7] pageviews (BR) ...", flush=True)
    attach_pageviews(br_cands)

    print("[3/7] region-stratified selection ...", flush=True)
    # Over-select so dropped-on-verification items can be back-filled toward the target.
    selected = select_region_stratified(br_cands, int(TARGET_BR * 1.6))
    print(f"      {len(selected)} pre-verification", flush=True)

    print("[4/7] gold re-verification (ASK per item) ...", flush=True)
    verified, dropped = [], []
    for c in selected:
        (verified if verify_gold(c) else dropped).append(c)
        if len(verified) >= TARGET_BR:
            break
    print(f"      verified={len(verified)}  dropped={len(dropped)}", flush=True)
    stats["br_verified"] = len(verified)
    stats["br_dropped"] = len(dropped)
    stats["br_dropped_qids"] = [c["qid"] for c in dropped]

    br_recs = make_br_records(verified, rng)
    print(f"      {len(br_recs)} BR MCQs built", flush=True)
    stats["br_final"] = len(br_recs)

    print("[5/7] fetch control candidates (round-robin countries) ...", flush=True)
    ctrl_cands = fetch_control_candidates()
    print(f"      {len(ctrl_cands)} control candidates", flush=True)
    attach_pageviews(ctrl_cands)

    print("[6/7] gold-verify control candidates ...", flush=True)
    ctrl_ok = [c for c in ctrl_cands if verify_gold(c)]
    print(f"      {len(ctrl_ok)}/{len(ctrl_cands)} control verified", flush=True)
    stats["control_candidates"] = len(ctrl_cands)
    stats["control_verified_candidates"] = len(ctrl_ok)

    print("[7/7] item-by-item rarity match (round-robin country) ...", flush=True)
    ctrl_recs = make_control_records(ctrl_ok, br_recs, rng)
    print(f"      {len(ctrl_recs)} control MCQs matched", flush=True)
    stats["control_final"] = len(ctrl_recs)

    # Trim BR to the matched count so the two files are paired 1:1.
    matched_ids = {r["matched_br_id"] for r in ctrl_recs}
    br_recs = [r for r in br_recs if r["id"] in matched_ids]
    stats["br_paired"] = len(br_recs)

    (DATA / "br_rare.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in br_recs) + "\n")
    (DATA / "control_matched.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in ctrl_recs) + "\n")

    # Region + relation + country + position summaries.
    stats["br_per_region"] = dict(Counter(r["region"] for r in br_recs))
    stats["br_per_relation"] = dict(Counter(r["relation"] for r in br_recs))
    stats["control_per_country"] = dict(Counter(r["country"] for r in ctrl_recs))
    stats["control_per_relation"] = dict(Counter(r["relation"] for r in ctrl_recs))
    LET = ["A", "B", "C", "D", "E"]
    stats["br_gold_pos"] = dict(Counter(LET[r["correct_index"]] for r in br_recs))
    stats["control_gold_pos"] = dict(Counter(LET[r["correct_index"]] for r in ctrl_recs))
    (DATA / "build_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))

    print(f"\nWrote {len(br_recs)} BR -> data/br_rare.jsonl", flush=True)
    print(f"Wrote {len(ctrl_recs)} control -> data/control_matched.jsonl", flush=True)
    print("Stats -> data/build_stats.json", flush=True)


if __name__ == "__main__":
    main()
