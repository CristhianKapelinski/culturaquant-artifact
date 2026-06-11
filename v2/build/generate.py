"""CulturaQuant-v2 smoke build: genuinely CULTURAL + rarity-matched control MCQs.

Cultural knowledge tested (NOT geography):
  - festival_to_state : Brazilian festival/celebration -> the state it is held in.
  - figure_to_state   : Brazilian cultural figure (musician/singer/composer/writer/
                        painter/poet/sculptor) -> their birthplace state.

Pipeline (resumable via on-disk cache):
  1. Cultural: long-tail BR festivals (all states) + BR arts figures (queried per state
     for region balance). Gold = state; region via STATE_REGION. Sitelinks from SPARQL;
     log-pageviews from Wikimedia REST.
  2. MCQ with 4 sibling distractors (other BR states), gold position round-robin A..E.
  3. Control: long-tail non-Brazilian festivals / arts figures -> first-level admin
     division, matched item-by-item to a cultural item on (relation-shape, sitelink band).
  4. Write smoke_cultural.jsonl + smoke_control.jsonl with full provenance.

Idempotent: SPARQL/pageview responses are cached; re-running reuses them. Seeded RNG.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from fetch import sparql
from mcq import build_mcq
from queries import (
    BR_STATES_QUERY,
    CONTROL_COUNTRIES,
    CONTROL_RELATIONS,
    CULTURAL_RELATIONS,
    STATE_REGION,
    control_festival_query,
    control_figure_query,
    country_admin_siblings_query,
    cultural_festival_query,
    cultural_figure_query,
)
from rarity import log_pageviews, rarity_band

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

SEED = 20260608
N_CULTURAL = 20            # smoke target
MAX_SITELINKS = 5          # relaxed long-tail filter for cultural entities (design)
FESTIVAL_LIMIT = 120       # festivals pulled (all states) before filtering
FIGURE_PER_STATE = 12      # figures pulled per state (keeps WDQS within budget)
CONTROL_LIMIT = 60         # control candidates pulled per (relation, country)

# Cultural-relation -> control-relation (same underlying shape).
REL_MAP = {
    "festival_to_state": "festival_to_admin",
    "figure_to_state": "figure_to_admin",
}

# Representative arts figures need per-region coverage; query one or two states per region
# so the smoke can balance N/NE/CO/SE/S. (Festivals already span regions on their own.)
FIGURE_STATES = [
    ("Q40040", "N"),    # Amazonas
    ("Q39517", "N"),    # Pará
    ("Q40430", "NE"),   # Bahia
    ("Q40123", "NE"),   # Ceará
    ("Q41587", "CO"),   # Goiás
    ("Q42824", "CO"),   # Mato Grosso
    ("Q39109", "SE"),   # Minas Gerais
    ("Q43233", "SE"),   # Espírito Santo
    ("Q40030", "S"),    # Rio Grande do Sul
    ("Q41115", "S"),    # Santa Catarina
]


def qid_of(binding: dict, key: str) -> str:
    return binding[key]["value"].split("/")[-1]


def has_pt_label(binding: dict) -> bool:
    label = binding["itemLabel"]["value"]
    return not (label.startswith("Q") and label[1:].isdigit())


def br_state_pool() -> list[tuple[str, str]]:
    d = sparql(BR_STATES_QUERY, tag="br_states")
    return [(qid_of(b, "state"), b["stateLabel"]["value"]) for b in d["results"]["bindings"]]


# ---------------------------------------------------------------- cultural fetch

def fetch_cultural_candidates() -> list[dict]:
    """Long-tail BR festivals (all states) + arts figures (per state)."""
    out: list[dict] = []
    seen_qids: set[str] = set()
    state_label = {q: lb for q, lb in br_state_pool()}

    # 1) Festivals -> state (one query over all states).
    fest_noun = CULTURAL_RELATIONS["festival_to_state"]["noun"]
    fest_q = CULTURAL_RELATIONS["festival_to_state"]["question"]
    d = sparql(cultural_festival_query(MAX_SITELINKS, FESTIVAL_LIMIT), tag="cult_festival")
    for b in d["results"]["bindings"]:
        if not has_pt_label(b):
            continue
        qid = qid_of(b, "item")
        if qid in seen_qids:
            continue
        state_qid = qid_of(b, "state")
        region = STATE_REGION.get(state_qid)
        if region is None:
            continue
        seen_qids.add(qid)
        out.append({
            "relation_key": "festival_to_state",
            "qid": qid,
            "label": b["itemLabel"]["value"],
            "noun": fest_noun,
            "question_tmpl": fest_q,
            "gold_qid": state_qid,
            "gold_label": b["stateLabel"]["value"],
            "region": region,
            "sitelinks": int(b["sitelinks"]["value"]),
        })

    # 2) Figures -> birthplace state (per state, for region balance).
    fig_noun = CULTURAL_RELATIONS["figure_to_state"]["noun"]
    fig_q = CULTURAL_RELATIONS["figure_to_state"]["question"]
    for state_qid, region in FIGURE_STATES:
        q = cultural_figure_query(state_qid, MAX_SITELINKS, FIGURE_PER_STATE)
        d = sparql(q, tag=f"cult_figure_{state_qid}")
        for b in d["results"]["bindings"]:
            if not has_pt_label(b):
                continue
            qid = qid_of(b, "item")
            if qid in seen_qids:
                continue
            seen_qids.add(qid)
            out.append({
                "relation_key": "figure_to_state",
                "qid": qid,
                "label": b["itemLabel"]["value"],
                "noun": fig_noun,
                "question_tmpl": fig_q,
                "gold_qid": state_qid,
                "gold_label": state_label.get(state_qid, "?"),
                "region": region,
                "sitelinks": int(b["sitelinks"]["value"]),
            })
    return out


# ---------------------------------------------------------------- control fetch

def fetch_control_candidates() -> list[dict]:
    """Long-tail non-BR festivals / arts figures -> first-level admin division."""
    out: list[dict] = []
    seen_qids: set[str] = set()
    builders = {
        "festival_to_admin": control_festival_query,
        "figure_to_admin": control_figure_query,
    }
    for rel_key, rel in CONTROL_RELATIONS.items():
        noun = rel["noun"]
        builder = builders[rel_key]
        for country_qid, (country_label, division_qid) in CONTROL_COUNTRIES.items():
            q = builder(division_qid, country_qid, MAX_SITELINKS, CONTROL_LIMIT)
            d = sparql(q, tag=f"ctrl_{rel_key}_{country_qid}")
            for b in d["results"]["bindings"]:
                if not has_pt_label(b):
                    continue
                qid = qid_of(b, "item")
                if qid in seen_qids:
                    continue
                seen_qids.add(qid)
                out.append({
                    "relation_key": rel_key,
                    "qid": qid,
                    "label": b["itemLabel"]["value"],
                    "noun": noun,
                    "gold_qid": qid_of(b, "adm"),
                    "gold_label": b["admLabel"]["value"],
                    "country_qid": country_qid,
                    "division_qid": division_qid,
                    "country_label": country_label,
                    "sitelinks": int(b["sitelinks"]["value"]),
                })
    return out


def attach_pageviews(items: list[dict]) -> None:
    for it in items:
        it["log_pageviews"] = log_pageviews(it["label"])


def source_url(qid: str) -> str:
    return f"https://www.wikidata.org/wiki/{qid}"


# ---------------------------------------------------------------- record builders

def select_cultural(cands: list[dict], rng: random.Random) -> list[dict]:
    """Pick ~N_CULTURAL rarest cultural candidates with region + relation spread.

    Round-robin over regions so all of N/NE/CO/SE/S appear, preferring the rarest items
    (lowest sitelinks, then lowest pageviews) within each region, and keeping a mix of the
    two relation shapes.
    """
    by_region: dict[str, list[dict]] = {}
    for c in cands:
        by_region.setdefault(c["region"], []).append(c)
    for lst in by_region.values():
        lst.sort(key=lambda c: (c["sitelinks"], c["log_pageviews"]))
    order = ["N", "NE", "CO", "SE", "S"]
    picked: list[dict] = []
    idx = {r: 0 for r in order}
    while len(picked) < N_CULTURAL * 2 and any(
        idx[r] < len(by_region.get(r, [])) for r in order
    ):
        for r in order:
            lst = by_region.get(r, [])
            if idx[r] < len(lst):
                picked.append(lst[idx[r]])
                idx[r] += 1
            if len(picked) >= N_CULTURAL * 2:
                break
    return picked


def make_cultural_records(cands: list[dict], rng: random.Random) -> list[dict]:
    pool = br_state_pool()
    recs: list[dict] = []
    for i, c in enumerate(cands):
        mcq = build_mcq(c["gold_label"], c["gold_qid"], pool, forced_index=i, rng=rng)
        if mcq is None:
            continue
        rel_word = "P131*" if c["relation_key"] == "festival_to_state" else "P19/P131*"
        recs.append({
            "id": f"v2-cult-{len(recs)+1:03d}",
            "set": "cultural",
            "relation": c["relation_key"],
            "qid": c["qid"],
            "statement": f"{c['label']} ({rel_word} -> {c['gold_label']})",
            "source_url": source_url(c["qid"]),
            "region": c["region"],
            "input": c["question_tmpl"].format(label=c["label"]),
            "alternatives": mcq["alternatives"],
            "alt_qids": mcq["alt_qids"],
            "correct_index": mcq["correct_index"],
            "distractor_qids": mcq["distractor_qids"],
            "rarity": {
                "sitelinks": c["sitelinks"],
                "log_pageviews": c["log_pageviews"],
                "band": rarity_band(c["sitelinks"]),
            },
        })
        if len(recs) >= N_CULTURAL:
            break
    return recs


def make_control_records(
    cands: list[dict], cultural: list[dict], rng: random.Random
) -> list[dict]:
    """Match each cultural record to a control candidate on (relation-shape, band).

    Greedy nearest-band matching with pageview tie-break; the control's country must have
    >=4 sibling divisions for distractors.
    """
    pool_cache: dict[str, list[tuple[str, str]]] = {}

    def admin_pool(division_qid: str, country_qid: str) -> list[tuple[str, str]]:
        if country_qid not in pool_cache:
            d = sparql(
                country_admin_siblings_query(division_qid, country_qid),
                tag=f"adm_{division_qid}_{country_qid}",
            )
            pool_cache[country_qid] = [
                (qid_of(b, "adm"), b["admLabel"]["value"])
                for b in d["results"]["bindings"]
            ]
        return pool_cache[country_qid]

    questions = {
        "festival_to_admin":
            "Em qual divisão administrativa de primeiro nível de {country} acontece o {label}?",
        "figure_to_admin":
            "Em qual divisão administrativa de primeiro nível de {country} nasceu {label}?",
    }

    used: set[str] = set()
    recs: list[dict] = []
    for ci, cult in enumerate(cultural):
        want_rel = REL_MAP.get(cult["relation"])
        want_band = cult["rarity"]["band"]
        want_pv = cult["rarity"]["log_pageviews"]
        opts = [c for c in cands if c["relation_key"] == want_rel and c["qid"] not in used]
        if not opts:
            opts = [c for c in cands if c["qid"] not in used]
        if not opts:
            break
        opts.sort(
            key=lambda c: (
                0 if rarity_band(c["sitelinks"]) == want_band else 1,
                abs(c["log_pageviews"] - want_pv),
            )
        )
        chosen = None
        mcq = None
        for c in opts:
            pool = [
                p for p in admin_pool(c["division_qid"], c["country_qid"])
                if p[0] != c["gold_qid"]
            ]
            m = build_mcq(c["gold_label"], c["gold_qid"], pool, forced_index=ci, rng=rng)
            if m is not None:
                chosen, mcq = c, m
                break
        if chosen is None:
            continue
        used.add(chosen["qid"])
        recs.append({
            "id": f"v2-ctrl-{len(recs)+1:03d}",
            "set": "control",
            "relation": chosen["relation_key"],
            "matched_cultural_id": cult["id"],
            "qid": chosen["qid"],
            "statement": f"{chosen['label']} (-> {chosen['gold_label']}, {chosen['country_label']})",
            "source_url": source_url(chosen["qid"]),
            "country": chosen["country_label"],
            "input": questions[chosen["relation_key"]].format(
                label=chosen["label"], country=chosen["country_label"]
            ),
            "alternatives": mcq["alternatives"],
            "alt_qids": mcq["alt_qids"],
            "correct_index": mcq["correct_index"],
            "distractor_qids": mcq["distractor_qids"],
            "rarity": {
                "sitelinks": chosen["sitelinks"],
                "log_pageviews": chosen["log_pageviews"],
                "band": rarity_band(chosen["sitelinks"]),
            },
        })
    return recs


def main() -> None:
    rng = random.Random(SEED)
    print("[1/5] fetching cultural candidates (festivals + figures) ...")
    cult_cands = fetch_cultural_candidates()
    print(f"      {len(cult_cands)} usable cultural candidates")
    print("[2/5] attaching pageviews (cultural) ...")
    attach_pageviews(cult_cands)
    selected = select_cultural(cult_cands, rng)
    cult_recs = make_cultural_records(selected, rng)
    print(f"      {len(cult_recs)} cultural MCQs built")

    print("[3/5] fetching control candidates ...")
    ctrl_cands = fetch_control_candidates()
    print(f"      {len(ctrl_cands)} usable control candidates")
    print("[4/5] attaching pageviews (control) ...")
    attach_pageviews(ctrl_cands)

    print("[5/5] matching control to cultural by rarity band ...")
    ctrl_recs = make_control_records(ctrl_cands, cult_recs, rng)
    print(f"      {len(ctrl_recs)} control MCQs built")

    (DATA / "smoke_cultural.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in cult_recs) + "\n"
    )
    (DATA / "smoke_control.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in ctrl_recs) + "\n"
    )
    print(f"\nWrote {len(cult_recs)} cultural -> data/smoke_cultural.jsonl")
    print(f"Wrote {len(ctrl_recs)} control  -> data/smoke_control.jsonl")


if __name__ == "__main__":
    main()
