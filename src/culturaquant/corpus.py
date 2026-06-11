"""Build the PT-BR Wikipedia cultural retrieval corpus (loader-only, CC BY-SA 4.0).

Source: the ``wikimedia/wikipedia`` dataset on the HuggingFace Hub, config
``20231101.pt``. Fetched at runtime, never redistributed, mirroring the BRoverbs
loader pattern. We filter to the cultural subset named in the spec (Brazilian
cuisine, geography, culture, and the 27 state articles), chunk each article into
~200-token passages, and persist a corpus JSONL plus a content-hash manifest so a
retrieval run is pinned to exact bytes.

The heavy ``datasets`` load lives behind :func:`build_corpus`; importing this
module performs no network I/O.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

WIKI_DATASET = "wikimedia/wikipedia"
WIKI_CONFIG = "20231101.pt"
WIKI_LICENSE = "CC BY-SA 4.0"

# Seed terms that mark an article as in the cultural subset. Matched (accent- and
# case-insensitively) against the article title and its first paragraph. Covers the
# PT Wikipedia categories Culinaria do Brasil, Geografia do Brasil, Cultura do
# Brasil, plus the 27 federative-unit (state) articles.
DEFAULT_CATEGORY_SEEDS: tuple[str, ...] = (
    # cuisine
    "culinaria", "comida", "prato tipico", "gastronomia", "doce", "quitute",
    # geography / culture of Brazil
    "geografia do brasil", "cultura do brasil", "cultura brasileira",
    "folclore", "festa popular", "carnaval", "danca", "musica brasileira",
    "regiao do brasil", "regiao norte", "regiao nordeste", "regiao centro-oeste",
    "regiao sudeste", "regiao sul",
)

# The 27 federative units (states + Distrito Federal). Title match is enough.
BRAZILIAN_STATES: tuple[str, ...] = (
    "Acre", "Alagoas", "Amapa", "Amazonas", "Bahia", "Ceara",
    "Distrito Federal", "Espirito Santo", "Goias", "Maranhao", "Mato Grosso",
    "Mato Grosso do Sul", "Minas Gerais", "Para", "Paraiba", "Parana",
    "Pernambuco", "Piaui", "Rio de Janeiro", "Rio Grande do Norte",
    "Rio Grande do Sul", "Rondonia", "Roraima", "Santa Catarina", "Sao Paulo",
    "Sergipe", "Tocantins",
)

DEFAULT_CHUNK_TOKENS = 200
_WS = re.compile(r"\s+")


def strip_accents(text: str) -> str:
    """NFKD accent-insensitive, lowercased normalization (corpus AND query)."""
    nfkd = unicodedata.normalize("NFKD", text)
    no_marks = "".join(c for c in nfkd if not unicodedata.combining(c))
    return no_marks.lower()


def tokenize(text: str) -> list[str]:
    """Accent-insensitive whitespace tokenizer, shared by corpus and query so BM25
    sees identical normalization on both sides."""
    norm = strip_accents(text)
    norm = re.sub(r"[^a-z0-9\s]", " ", norm)
    return [t for t in _WS.split(norm) if t]


@dataclass(frozen=True)
class Passage:
    """One ~chunk_tokens passage from a source article."""

    id: str
    title: str
    text: str

    def as_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "text": self.text}


def _article_matches(
    title: str, first_paragraph: str, category_seeds: tuple[str, ...]
) -> bool:
    ntitle = strip_accents(title)
    if any(strip_accents(s) == ntitle for s in BRAZILIAN_STATES):
        return True
    hay = strip_accents(title + " " + first_paragraph)
    return any(strip_accents(seed) in hay for seed in category_seeds)


def chunk_article(title: str, text: str, chunk_tokens: int) -> list[str]:
    """Split an article body into ~chunk_tokens whitespace-token passages.

    Chunking is on raw whitespace tokens (not the accent-stripped ones) so the
    stored passage stays human-readable; BM25 tokenization is applied later.
    """
    words = text.split()
    if not words:
        return []
    return [
        " ".join(words[i : i + chunk_tokens])
        for i in range(0, len(words), chunk_tokens)
    ]


def build_corpus(
    out_dir: Path,
    category_seeds: tuple[str, ...] = DEFAULT_CATEGORY_SEEDS,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    max_articles: int | None = None,
    streaming: bool = True,
) -> dict:
    """Fetch, filter, chunk and persist the cultural Wikipedia corpus.

    Writes ``corpus.jsonl`` (one passage per line) and ``corpus_manifest.json``
    (source pin, content hash, parameters, counts) into ``out_dir``. Idempotent:
    if both files exist and the manifest parameters match, the build is skipped.
    Returns the manifest dict.

    ``max_articles`` caps the scan for smoke/dev runs. ``streaming`` avoids
    materializing the full PT dump in RAM.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = out_dir / "corpus.jsonl"
    manifest_path = out_dir / "corpus_manifest.json"

    params = {
        "dataset": WIKI_DATASET,
        "config": WIKI_CONFIG,
        "chunk_tokens": chunk_tokens,
        "n_category_seeds": len(category_seeds),
        "max_articles": max_articles,
    }
    if corpus_path.exists() and manifest_path.exists():
        prev = json.loads(manifest_path.read_text())
        if prev.get("params") == params:
            return prev

    from datasets import load_dataset

    ds = load_dataset(
        WIKI_DATASET, WIKI_CONFIG, split="train", streaming=streaming
    )

    passages: list[Passage] = []
    n_articles = 0
    n_scanned = 0
    for row in ds:
        n_scanned += 1
        if max_articles is not None and n_scanned > max_articles:
            break
        title = row.get("title", "") or ""
        text = row.get("text", "") or ""
        first_para = text[:500]
        if not _article_matches(title, first_para, category_seeds):
            continue
        n_articles += 1
        for ci, chunk in enumerate(chunk_article(title, text, chunk_tokens)):
            pid = f"{strip_accents(title).replace(' ', '_')}-{ci:04d}"
            passages.append(Passage(id=pid, title=title, text=chunk))

    hasher = hashlib.sha256()
    with corpus_path.open("w", encoding="utf-8") as f:
        for p in passages:
            line = json.dumps(p.as_dict(), ensure_ascii=False)
            f.write(line + "\n")
            hasher.update(line.encode("utf-8"))

    manifest = {
        "params": params,
        "license": WIKI_LICENSE,
        "source": f"{WIKI_DATASET}:{WIKI_CONFIG}",
        "n_articles": n_articles,
        "n_passages": len(passages),
        "corpus_sha256": hasher.hexdigest(),
        "corpus_file": corpus_path.name,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def load_corpus(corpus_path: Path) -> list[Passage]:
    """Load a persisted corpus JSONL into Passage objects (no network)."""
    corpus_path = Path(corpus_path)
    out: list[Passage] = []
    with corpus_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            out.append(Passage(id=r["id"], title=r["title"], text=r["text"]))
    return out


def corpus_hash(corpus_path: Path) -> str:
    """SHA-256 over the corpus file content (the pin recorded in run manifests)."""
    h = hashlib.sha256()
    h.update(Path(corpus_path).read_bytes())
    return h.hexdigest()
