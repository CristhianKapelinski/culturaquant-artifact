"""SPARQL query templates for CulturaQuant-v2 candidate generation.

CULTURAL relations (genuinely Brazilian culture, not geography):

  1. festival_to_state  -- a Brazilian festival/celebration (carnival, religious feast,
     music/film festival; P31/P279* Q132241) -> the federative unit (state) it is held in
     (P131*). This is cultural knowledge ("in which state is festival X held?").
  2. figure_to_state    -- a Brazilian cultural figure (musician, singer, composer,
     writer; P106 in a curated arts set) -> their birthplace state (P19 -> P131* state).
     This is cultural knowledge about who-comes-from-where in Brazilian arts.

Gold = the state; macro-region (N/NE/CO/SE/S) derived from the state via STATE_REGION.

CONTROL relations (non-Brazilian, same SHAPE and rarity, no Brazilian-cultural content):
  1. festival_to_admin -- a non-Brazil festival -> first-level admin division of its
     country.
  2. figure_to_admin   -- a non-Brazil arts figure -> birthplace first-level admin
     division of their country.
Gold = that division; sibling distractors are other first-level divisions of the country.

Design note: dishes/foods and music genres were probed and rejected as primary relations
because in Wikidata they are almost never tagged with a sub-national region (dish->state
via P131 yields ~2 items; genres carry no state). Festivals and cultural figures are the
two cultural relations that ARE region-tagged and exist at the long-tail band; see the
feasibility report. The sitelink long-tail filter is RELAXED to <= 5 (vs the geography
build's <= 3) because cultural entities are inherently a touch less obscure than random
creeks, yet still far from famous facts.
"""

from __future__ import annotations

# Canonical Brazilian federative-unit QID -> macro-region (N/NE/CO/SE/S).
# QIDs verified against WDQS (P31 wd:Q485258); region assignment is IBGE-standard
# geography (stable, authoritative; not model-derivable trivia).
STATE_REGION = {
    # Norte
    "Q40040": "N",   # Amazonas
    "Q39517": "N",   # Pará
    "Q40780": "N",   # Acre
    "Q42508": "N",   # Roraima
    "Q40130": "N",   # Amapá
    "Q43235": "N",   # Rondônia
    "Q43695": "N",   # Tocantins
    # Nordeste
    "Q40885": "NE",  # Alagoas
    "Q40430": "NE",  # Bahia
    "Q40123": "NE",  # Ceará
    "Q42362": "NE",  # Maranhão
    "Q38088": "NE",  # Paraíba
    "Q40942": "NE",  # Pernambuco
    "Q42722": "NE",  # Piauí
    "Q43255": "NE",  # Rio Grande do Norte
    "Q43783": "NE",  # Sergipe
    # Centro-Oeste
    "Q41587": "CO",  # Goiás
    "Q42824": "CO",  # Mato Grosso
    "Q43319": "CO",  # Mato Grosso do Sul
    "Q119158": "CO",  # Distrito Federal
    # Sudeste
    "Q43233": "SE",  # Espírito Santo
    "Q39109": "SE",  # Minas Gerais
    "Q41428": "SE",  # Rio de Janeiro
    "Q175": "SE",    # São Paulo
    # Sul
    "Q15499": "S",   # Paraná
    "Q40030": "S",   # Rio Grande do Sul
    "Q41115": "S",   # Santa Catarina
}


# Curated arts occupations for "cultural figure" relations (Brazilian + control).
# musician, singer, composer, writer, painter, poet, sculptor.
ARTS_OCCUPATIONS = [
    "Q639669",   # musician
    "Q177220",   # singer
    "Q36834",    # composer
    "Q36180",    # writer
    "Q1028181",  # painter
    "Q49757",    # poet
    "Q1281618",  # sculptor
]


# ---- Cultural candidate queries (Brazil) ----
#
# Each relation maps to a candidate-query builder and the question template / noun used to
# render the MCQ stem. Gold is always a Brazilian state.

CULTURAL_RELATIONS = {
    "festival_to_state": {
        "noun": "festival",
        "question": "Em qual estado brasileiro acontece o {label}?",
    },
    "figure_to_state": {
        "noun": "personalidade",
        "question": "Em qual estado brasileiro nasceu {label}?",
    },
}


def _occ_values() -> str:
    return " ".join(f"wd:{q}" for q in ARTS_OCCUPATIONS)


def cultural_festival_query(max_sitelinks: int, limit: int) -> str:
    """Long-tail Brazilian festivals (Q132241) -> the federative unit they are held in."""
    return f"""
SELECT ?item ?itemLabel ?state ?stateLabel (COUNT(DISTINCT ?sl) AS ?sitelinks) WHERE {{
  ?item wdt:P31/wdt:P279* wd:Q132241 ; wdt:P131* ?state .
  ?state wdt:P31 wd:Q485258 .
  ?item rdfs:label ?itemLabel . FILTER(LANG(?itemLabel)="pt")
  FILTER NOT EXISTS {{ ?item wikibase:sitelinks ?sx . FILTER(?sx > {max_sitelinks}) }}
  OPTIONAL {{ ?sl schema:about ?item . }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "pt". }}
}}
GROUP BY ?item ?itemLabel ?state ?stateLabel
HAVING (COUNT(DISTINCT ?sl) <= {max_sitelinks})
ORDER BY ?sitelinks
LIMIT {limit}
"""


def cultural_figure_query(state_qid: str, max_sitelinks: int, limit: int) -> str:
    """Long-tail Brazilian arts figures born in (a city of) a given state.

    Queried per-state so the long-tail sitelink filter stays inside WDQS's time budget and
    so per-region coverage can be controlled by the caller (one query per state).
    """
    return f"""
SELECT ?item ?itemLabel (COUNT(DISTINCT ?sl) AS ?sitelinks) WHERE {{
  VALUES ?occ {{ {_occ_values()} }}
  ?item wdt:P106 ?occ ; wdt:P19 ?city .
  ?city wdt:P131* wd:{state_qid} .
  ?item rdfs:label ?itemLabel . FILTER(LANG(?itemLabel)="pt")
  FILTER NOT EXISTS {{ ?item wikibase:sitelinks ?sx . FILTER(?sx > {max_sitelinks}) }}
  OPTIONAL {{ ?sl schema:about ?item . }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "pt". }}
}}
GROUP BY ?item ?itemLabel
ORDER BY ?sitelinks
LIMIT {limit}
"""


# Sibling pool: all Brazilian federative units (for distractors), with pt labels.
BR_STATES_QUERY = """
SELECT ?state ?stateLabel WHERE {
  ?state wdt:P31 wd:Q485258 .
  ?state rdfs:label ?stateLabel . FILTER(LANG(?stateLabel)="pt")
}
"""


# ---- Control candidate queries (non-Brazil) ----
#
# Mirror the cultural shapes (festival -> division; figure -> birthplace division) for a
# curated set of non-Brazil countries whose first-level division type is rich enough (>=5)
# to supply sibling distractors and whose division labels exist in pt.
CONTROL_COUNTRIES = {
    "Q142": ("França", "Q6465"),      # department of France
    "Q38": ("Itália", "Q16110"),      # province of Italy
    "Q29": ("Espanha", "Q162620"),    # province of Spain
    "Q414": ("Argentina", "Q44753"),  # province of Argentina
    "Q183": ("Alemanha", "Q1221156"), # district of Germany (Landkreis)
    "Q145": ("Reino Unido", "Q180673"),  # ceremonial county of England
}

CONTROL_RELATIONS = {
    "festival_to_admin": {"noun": "festival"},
    "figure_to_admin": {"noun": "personalidade"},
}


def control_festival_query(
    division_qid: str, country_qid: str, max_sitelinks: int, limit: int
) -> str:
    """Long-tail non-BR festivals -> their first-level admin division."""
    return f"""
SELECT ?item ?itemLabel ?adm ?admLabel (COUNT(DISTINCT ?sl) AS ?sitelinks) WHERE {{
  ?item wdt:P31/wdt:P279* wd:Q132241 ; wdt:P131 ?adm .
  ?adm wdt:P31 wd:{division_qid} ; wdt:P17 wd:{country_qid} .
  ?item rdfs:label ?itemLabel . FILTER(LANG(?itemLabel)="pt")
  FILTER NOT EXISTS {{ ?item wikibase:sitelinks ?sx . FILTER(?sx > {max_sitelinks}) }}
  OPTIONAL {{ ?sl schema:about ?item . }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "pt". }}
}}
GROUP BY ?item ?itemLabel ?adm ?admLabel
HAVING (COUNT(DISTINCT ?sl) <= {max_sitelinks})
ORDER BY ?sitelinks
LIMIT {limit}
"""


def control_figure_query(
    division_qid: str, country_qid: str, max_sitelinks: int, limit: int
) -> str:
    """Long-tail non-BR arts figures -> birthplace first-level admin division.

    Anchors the division FIRST (?adm of the country's first-level type), then finds cities
    directly in it (?city wdt:P131 ?adm) and figures born there. Anchoring the division and
    using a one-hop P131 keeps the query inside the WDQS 60s budget (the transitive P131*
    over every artist in a country times out). Dropping figures whose birthplace city is
    nested one extra level is acceptable: the control pool only needs item-by-item rarity
    matches to the cultural set, not exhaustive coverage.
    """
    return f"""
SELECT ?item ?itemLabel ?adm ?admLabel (COUNT(DISTINCT ?sl) AS ?sitelinks) WHERE {{
  ?adm wdt:P31 wd:{division_qid} ; wdt:P17 wd:{country_qid} .
  ?city wdt:P131 ?adm .
  ?item wdt:P19 ?city ; wdt:P106 ?occ .
  VALUES ?occ {{ {_occ_values()} }}
  ?item rdfs:label ?itemLabel . FILTER(LANG(?itemLabel)="pt")
  FILTER NOT EXISTS {{ ?item wikibase:sitelinks ?sx . FILTER(?sx > {max_sitelinks}) }}
  OPTIONAL {{ ?sl schema:about ?item . }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "pt". }}
}}
GROUP BY ?item ?itemLabel ?adm ?admLabel
ORDER BY ?sitelinks
LIMIT {limit}
"""


def country_admin_siblings_query(division_qid: str, country_qid: str) -> str:
    """First-level divisions of a country, with pt labels (distractor pool)."""
    return f"""
SELECT ?adm ?admLabel WHERE {{
  ?adm wdt:P31 wd:{division_qid} ; wdt:P17 wd:{country_qid} .
  ?adm rdfs:label ?admLabel . FILTER(LANG(?admLabel)="pt")
}}
"""
