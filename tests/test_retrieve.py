"""BM25 retrieval over a tiny in-repo fixture corpus (no network, no GPU).

Verifies the planted relevant passage is retrieved for a matching query and that
the gold-leakage guard rejects a query containing an option string.
"""

import pytest

from culturaquant.corpus import Passage, strip_accents, tokenize
from culturaquant.retrieve import BM25Retriever, truncate_tokens

FIXTURE = [
    Passage(id="bahia-0000", title="Bahia",
            text="Salvador e a capital do estado da Bahia, no nordeste do Brasil."),
    Passage(id="parana-0000", title="Parana",
            text="Curitiba e a capital do estado do Parana, na regiao sul."),
    Passage(id="culinaria-0000", title="Culinaria",
            text="O acaraje e um quitute de origem baiana feito de feijao fradinho."),
]


def test_tokenize_accent_insensitive():
    assert tokenize("Ceará") == ["ceara"]
    assert strip_accents("São Paulo") == "sao paulo"


def test_bm25_retrieves_planted_passage():
    r = BM25Retriever(FIXTURE)
    hits = r.retrieve("Qual e a capital da Bahia?", top_k=1)
    assert len(hits) == 1
    assert hits[0].id == "bahia-0000"
    assert hits[0].score > 0


def test_bm25_top_k_returns_ordered():
    r = BM25Retriever(FIXTURE)
    hits = r.retrieve("capital do estado", top_k=3)
    assert len(hits) == 3
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_query_must_not_contain_options():
    r = BM25Retriever(FIXTURE)
    # the gold option string leaked into the query -> guard fires
    with pytest.raises(AssertionError):
        r.retrieve("Qual a capital? Salvador", top_k=1,
                   options=["Salvador", "Ilheus"])


def test_options_passed_but_absent_is_fine():
    r = BM25Retriever(FIXTURE)
    hits = r.retrieve("Qual e a capital da Bahia?", top_k=1,
                      options=["Salvador", "Ilheus"])
    assert hits[0].id == "bahia-0000"


def test_empty_corpus_rejected():
    with pytest.raises(ValueError):
        BM25Retriever([])


def test_truncate_tokens():
    assert truncate_tokens("a b c d e", 3) == "a b c"
