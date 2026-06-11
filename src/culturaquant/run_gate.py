"""Orchestrate the QDIS selective-retrieval gate experiment (EXP_A).

Per model we run exactly four forward-pass sweeps over the gate item set (70
localized cultural + 50 control, proverbs excluded):

  1. fp16 closed-book   (offline only: oracle + fallback calibration)
  2. int8 closed-book   (deployment precision; the answer we ship)
  3. nf4  closed-book    (cheap second quant pass; the QDIS gate signal)
  4. int8 always-RAG     (top-k passage injected on every item)

Gating is POST-HOC: budgets {0.10..0.50} and both gates (QDIS, generic margin)
are derived from the cached per-item logprobs, so the five budgets cost no extra
forward passes. We persist raw per-item logprobs at all three precisions, the
QDIS components, every gate decision bitmask, retrieved passage ids + BM25
scores, per-condition correct flags, and a manifest pinning the corpus hash,
retriever config, tau values, seed and model id.

Resumable/idempotent: a model's output file is skipped if it already exists and
its manifest hash (model id + corpus hash + seed + retriever cfg) matches.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from .corpus import build_corpus, corpus_hash, load_corpus
from .data import Item, load_control, load_cultural_strata
from .gate import (
    apply_gate,
    gate_decisions,
    oracle_decisions,
    qdis_components,
)
from .retrieve import build_retriever
from .run import reference_machine

BUDGETS = (0.10, 0.20, 0.30, 0.40, 0.50)
GATES = ("quant_aware", "generic")
SEED = 20260607


def gate_items(
    data_dir: Path,
    max_cultural: int | None = None,
    max_control: int | None = None,
) -> list[Item]:
    """The gate item set: 70 localized cultural + 50 control. No proverbs (the
    gate question is ill-posed for them; see spec section 5).

    ``max_cultural``/``max_control`` cap each group for the smoke test. The
    cultural cap stratifies across cuisine/geography/regional_facts so a small
    smoke subset still touches every stratum.
    """
    cultural = load_cultural_strata(data_dir)
    control = load_control(data_dir)
    if max_cultural is not None:
        cultural = _stratified_head(cultural, max_cultural)
    if max_control is not None:
        control = control[:max_control]
    return cultural + control


def _stratified_head(items: list[Item], k: int) -> list[Item]:
    """First ``k`` items spread round-robin across strata (deterministic)."""
    by_stratum: dict[str, list[Item]] = {}
    for it in items:
        by_stratum.setdefault(it.stratum, []).append(it)
    out: list[Item] = []
    queues = list(by_stratum.values())
    qi = 0
    while len(out) < k and any(queues):
        q = queues[qi % len(queues)]
        if q:
            out.append(q.pop(0))
        qi += 1
        if not any(queues):
            break
    return out[:k]


def _slug(name: str) -> str:
    return name.replace("/", "__")


def _unit_hash(model: str, chash: str, seed: int, retr_cfg: dict) -> str:
    payload = json.dumps(
        {
            "model": model,
            "corpus_sha256": chash,
            "seed": seed,
            "retriever": retr_cfg,
            # bump when the scoring contract changes so stale outputs re-run.
            "scoring": "perm_options_v2",
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _argmax_correct(norms: list[float], gold: int) -> int:
    from .gate import argmax

    return int(argmax(norms) == gold)


def score_model(
    name: str,
    items: list[Item],
    retriever,
    retrieved: dict[str, list],
    precisions: tuple[str, ...] = ("fp16", "int8", "nf4"),
    top_k: int = 1,
    seed: int = SEED,
) -> list[dict]:
    """Run the forward-pass sweeps for one model and assemble per-item records.

    ``retrieved`` maps item id -> list[Retrieved] (precomputed on CPU, identical
    across models). Returns one record per item carrying logprobs at every
    precision, the always-RAG int8 logprobs, QDIS components and correct flags for
    the closed-book + always-RAG conditions.

    Options are deterministically permuted per item (seeded by item id) so the
    gold answer is not pinned at index 0; the SAME permutation is applied to every
    precision and to the RAG pass so the paired QDIS comparison stays clean.
    """
    from .scorer import (
        apply_permutation,
        free_model,
        load_model,
        option_permutation,
        score_item_logprobs,
    )

    # Per-item option permutation (shared across all precisions + RAG).
    perms: list[list[int]] = []
    perm_options: list[list[str]] = []
    perm_gold: list[int] = []
    for it in items:
        perm = option_permutation(it.id, len(it.options), seed)
        opts, gold = apply_permutation(list(it.options), it.gold_index, perm)
        perms.append(perm)
        perm_options.append(opts)
        perm_gold.append(gold)

    # closed-book logprobs per precision
    lp: dict[str, list[list[float]]] = {}
    for prec in precisions:
        m = load_model(name, prec)
        lp[prec] = [
            score_item_logprobs(m, it.question, perm_options[i])
            for i, it in enumerate(items)
        ]
        free_model(m)

    # always-RAG int8 logprobs (top-1 passage injected)
    m = load_model(name, "int8")
    rag_int8 = []
    for i, it in enumerate(items):
        passages = retrieved.get(it.id, [])
        ctx = passages[0].text if passages else None
        rag_int8.append(score_item_logprobs(m, it.question, perm_options[i], ctx))
    free_model(m)

    records = []
    for i, it in enumerate(items):
        gold = perm_gold[i]
        l_fp16 = lp.get("fp16", [None] * len(items))[i]
        l_int8 = lp["int8"][i]
        l_nf4 = lp["nf4"][i]
        comps = qdis_components(l_int8, l_nf4)
        passages = retrieved.get(it.id, [])
        rec = {
            "id": it.id,
            "group": it.group,
            "stratum": it.stratum,
            "region": it.region,
            "gold": gold,
            "gold_canonical": it.gold_index,
            "option_permutation": perms[i],
            "logprobs_fp16": l_fp16,
            "logprobs_int8": l_int8,
            "logprobs_nf4": l_nf4,
            "logprobs_rag_int8": rag_int8[i],
            "qdis": comps,
            "retrieved": [
                {"id": p.id, "score": p.score} for p in passages[:top_k]
            ],
            "correct_closed_book": _argmax_correct(l_int8, gold),
            "correct_always_rag": _argmax_correct(rag_int8[i], gold),
            "correct_fp16": (
                _argmax_correct(l_fp16, gold) if l_fp16 is not None else None
            ),
        }
        records.append(rec)
    return records


def derive_conditions(records: list[dict]) -> dict:
    """Derive every condition (post-hoc) from cached per-item records.

    Returns a dict with: per-condition per-item correct flags, gate bitmasks per
    (gate, budget), oracle decisions, and the tau values used. Gate fitting is
    cross-fitted over ALL gate items (cultural + control), per the spec.
    """
    closed = [r["correct_closed_book"] for r in records]
    rag = [r["correct_always_rag"] for r in records]
    gate_inputs = [
        {
            "flip": r["qdis"]["flip"],
            "jsd": r["qdis"]["jsd"],
            "margin_int8": r["qdis"]["margin_int8"],
        }
        for r in records
    ]

    out: dict = {
        "closed_book": closed,
        "always_rag": rag,
        "oracle_gate": [int(x) for x in oracle_decisions(closed, rag)],
        "gates": {},
    }
    for gate in GATES:
        out["gates"][gate] = {}
        for b in BUDGETS:
            retrieve = gate_decisions(gate_inputs, gate, b, seed=SEED)
            applied = apply_gate(closed, rag, retrieve)
            out["gates"][gate][f"{b:.2f}"] = {
                "retrieve": [int(x) for x in retrieve],
                "correct": applied,
                "budget_realized": (
                    sum(retrieve) / len(retrieve) if retrieve else 0.0
                ),
            }
    return out


def run_one_model(
    name: str,
    items: list[Item],
    retriever,
    retrieved: dict,
    out_dir: Path,
    corpus_meta: dict,
    retr_cfg: dict,
    precisions: tuple[str, ...],
    top_k: int,
) -> dict:
    """Score one model, derive all conditions, write its JSONL + manifest.

    Idempotent: if the output file exists and its embedded unit hash matches, the
    model is skipped and the cached summary returned.
    """
    out_path = out_dir / f"gate__{_slug(name)}.json"
    uhash = _unit_hash(name, corpus_meta["corpus_sha256"], SEED, retr_cfg)
    if out_path.exists():
        prev = json.loads(out_path.read_text())
        if prev.get("manifest", {}).get("unit_hash") == uhash:
            return {"model": name, "status": "skipped", "file": out_path.name}

    t0 = time.time()
    records = score_model(name, items, retriever, retrieved, precisions, top_k, SEED)
    conditions = derive_conditions(records)
    elapsed = time.time() - t0

    blob = {
        "manifest": {
            "model": name,
            "unit_hash": uhash,
            "seed": SEED,
            "corpus_sha256": corpus_meta["corpus_sha256"],
            "corpus_source": corpus_meta["source"],
            "corpus_n_passages": corpus_meta["n_passages"],
            "retriever": retr_cfg,
            "budgets": list(BUDGETS),
            "gates": list(GATES),
            "precisions": list(precisions),
            "n_items": len(items),
            "seconds": round(elapsed, 1),
            "reference_machine": reference_machine(),
        },
        "records": records,
        "conditions": conditions,
    }
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(blob, ensure_ascii=False), encoding="utf-8")
    tmp.replace(out_path)
    return {"model": name, "status": "done", "file": out_path.name,
            "seconds": round(elapsed, 1)}


def prepare_retrieval(
    items: list[Item], retriever, top_k: int
) -> dict[str, list]:
    """Retrieve top-k passages for every item (CPU, model-independent).

    Query is the question stem only; options are passed solely to activate the
    gold-leakage guard.
    """
    out: dict[str, list] = {}
    for it in items:
        out[it.id] = retriever.retrieve(
            it.question, top_k=top_k, options=list(it.options)
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the QDIS gate experiment (EXP_A).")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--corpus-dir", required=True,
                    help="dir holding/persisting corpus.jsonl + manifest")
    ap.add_argument("--precisions", nargs="+", default=["fp16", "int8", "nf4"])
    ap.add_argument("--retriever", default="bm25", choices=["bm25", "dense"])
    ap.add_argument("--top-k", type=int, default=1)
    ap.add_argument("--chunk-tokens", type=int, default=200)
    ap.add_argument("--max-articles", type=int, default=None,
                    help="cap the corpus scan (smoke/dev); None = full PT dump")
    ap.add_argument("--max-cultural", type=int, default=None,
                    help="cap cultural items (smoke), stratified across strata")
    ap.add_argument("--max-control", type=int, default=None,
                    help="cap control items (smoke)")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    import random

    import numpy as np
    import torch

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir = Path(args.corpus_dir)

    items = gate_items(data_dir, args.max_cultural, args.max_control)
    print(f"[gate] {len(items)} items (cultural+control, proverbs excluded)", flush=True)

    corpus_meta = build_corpus(
        corpus_dir, chunk_tokens=args.chunk_tokens, max_articles=args.max_articles
    )
    chash = corpus_hash(corpus_dir / corpus_meta["corpus_file"])
    corpus_meta["corpus_sha256"] = chash
    print(f"[gate] corpus {corpus_meta['n_passages']} passages, sha {chash[:12]}",
          flush=True)

    passages = load_corpus(corpus_dir / corpus_meta["corpus_file"])
    retriever = build_retriever(passages, kind=args.retriever)
    retr_cfg = {"kind": args.retriever, "top_k": args.top_k,
                "chunk_tokens": args.chunk_tokens, "trunc_tokens": 256}
    retrieved = prepare_retrieval(items, retriever, args.top_k)

    models = [m for m in args.models if m != "__none__"]
    if not models:
        print("[gate] corpus + retrieval prepared; no models to score", flush=True)

    summary = []
    for name in models:
        print(f"[gate] scoring {name}", flush=True)
        res = run_one_model(
            name, items, retriever, retrieved, out_dir, corpus_meta,
            retr_cfg, tuple(args.precisions), args.top_k,
        )
        print(f"[gate] {res}", flush=True)
        summary.append(res)

    (out_dir / "gate_run_manifest.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "models": args.models,
                "corpus": corpus_meta,
                "retriever": retr_cfg,
                "budgets": list(BUDGETS),
                "gates": list(GATES),
                "runs": summary,
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print("[gate] done", flush=True)


if __name__ == "__main__":
    main()
