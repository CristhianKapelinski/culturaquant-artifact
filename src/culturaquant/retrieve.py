"""Retrieval over the cultural corpus: BM25 (primary, CPU) and an optional dense
``bge-m3`` path (wave-2 robustness, import-guarded).

The query is ALWAYS the MCQ question stem only, never the options, so the gold
answer cannot leak into the retriever. This is asserted at call time. Corpus and
query share the accent-insensitive NFKD tokenizer from :mod:`corpus`, and the
returned passage text is truncated to a token budget (256 by default) to match
the edge token budget.
"""

from __future__ import annotations

from dataclasses import dataclass

from .corpus import Passage, strip_accents, tokenize

TRUNC_TOKENS = 256


@dataclass(frozen=True)
class Retrieved:
    """One retrieved passage with its retriever score."""

    id: str
    score: float
    text: str


def truncate_tokens(text: str, max_tokens: int = TRUNC_TOKENS) -> str:
    """Truncate to the first ``max_tokens`` whitespace tokens (display/injection)."""
    words = text.split()
    return " ".join(words[:max_tokens])


def _assert_no_options(query: str, options: list[str] | None) -> None:
    """Guard against gold leakage: no option may appear as a contiguous token run
    inside the query. Token-level (not raw substring) so a short option token is
    not falsely flagged inside an unrelated word."""
    if not options:
        return
    q_tokens = tokenize(query)
    q_join = " " + " ".join(q_tokens) + " "
    for opt in options:
        opt_tokens = tokenize(opt or "")
        if not opt_tokens:
            continue
        if " " + " ".join(opt_tokens) + " " in q_join:
            raise AssertionError(
                "retrieval query must be the question stem only; an option string "
                f"leaked into the query: {strip_accents(opt)!r}"
            )


class BM25Retriever:
    """BM25 over the cultural corpus via ``rank-bm25`` (Okapi BM25, CPU).

    Build once per corpus, then call :meth:`retrieve` per query. The index keeps
    the accent-insensitive tokenization used at build time so query tokenization
    matches the corpus exactly.
    """

    def __init__(self, passages: list[Passage], trunc_tokens: int = TRUNC_TOKENS):
        from rank_bm25 import BM25Okapi

        self.passages = passages
        self.trunc_tokens = trunc_tokens
        tokenized = [tokenize(p.text) for p in passages]
        # guard: BM25Okapi divides by corpus length; an empty corpus is a config error
        if not tokenized:
            raise ValueError("cannot build BM25 over an empty corpus")
        self.bm25 = BM25Okapi(tokenized)

    def retrieve(
        self, query: str, top_k: int = 1, options: list[str] | None = None
    ) -> list[Retrieved]:
        """Top-``top_k`` passages for the question-stem ``query``.

        Passing ``options`` activates the gold-leakage guard (it asserts no option
        substring appears in the query); the options are never used for scoring.
        """
        _assert_no_options(query, options)
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: list[Retrieved] = []
        for i in order[:top_k]:
            p = self.passages[i]
            out.append(
                Retrieved(
                    id=p.id,
                    score=float(scores[i]),
                    text=truncate_tokens(p.text, self.trunc_tokens),
                )
            )
        return out


class DenseRetriever:
    """Optional dense retriever (``BAAI/bge-m3`` + FAISS flat, cosine).

    Wave-2 robustness check only. The heavy ``sentence-transformers`` / ``faiss``
    imports are deferred to construction so the module stays import-safe without
    those extras installed; the interface mirrors :class:`BM25Retriever`.
    """

    def __init__(
        self,
        passages: list[Passage],
        model_name: str = "BAAI/bge-m3",
        trunc_tokens: int = TRUNC_TOKENS,
        device: str = "cpu",
    ):
        import faiss  # noqa: F401  (import-guarded heavy dep)
        from sentence_transformers import SentenceTransformer

        self.passages = passages
        self.trunc_tokens = trunc_tokens
        self.model = SentenceTransformer(model_name, device=device)
        embs = self.model.encode(
            [p.text for p in passages], normalize_embeddings=True,
            convert_to_numpy=True, show_progress_bar=False,
        )
        import faiss as _faiss

        self.index = _faiss.IndexFlatIP(embs.shape[1])
        self.index.add(embs)

    def retrieve(
        self, query: str, top_k: int = 1, options: list[str] | None = None
    ) -> list[Retrieved]:
        _assert_no_options(query, options)
        q = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False,
        )
        scores, idxs = self.index.search(q, top_k)
        out: list[Retrieved] = []
        for score, i in zip(scores[0], idxs[0]):
            if i < 0:
                continue
            p = self.passages[i]
            out.append(
                Retrieved(
                    id=p.id, score=float(score),
                    text=truncate_tokens(p.text, self.trunc_tokens),
                )
            )
        return out


def build_retriever(
    passages: list[Passage], kind: str = "bm25", **kwargs
) -> BM25Retriever | DenseRetriever:
    """Factory: ``kind`` in {"bm25", "dense"}. BM25 is the headline retriever."""
    if kind == "bm25":
        return BM25Retriever(passages, **kwargs)
    if kind == "dense":
        return DenseRetriever(passages, **kwargs)
    raise ValueError(f"unknown retriever kind {kind!r}")
