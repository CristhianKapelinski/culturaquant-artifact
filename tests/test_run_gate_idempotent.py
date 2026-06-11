"""run_gate idempotency on a CPU-mockable path (no network, no GPU).

The LM loader/scorer is replaced by a deterministic stub, so the orchestration,
post-hoc condition derivation, JSONL persistence and skip-on-rerun logic are
exercised end to end without torch or a model. The first run writes the model
file; the second run skips it (the unit hash matches).
"""

import json

import pytest

from culturaquant import run_gate
from culturaquant.corpus import Passage
from culturaquant.data import Item
from culturaquant.retrieve import BM25Retriever


class _StubModel:
    def __init__(self, name, precision):
        self.name = name
        self.precision = precision


def _stub_logprobs(lm, question, options, context=None):
    """Deterministic per-option log-probs from a hash of the inputs, perturbed by
    precision and by whether a context was injected, so flips/JSD are non-trivial.
    """
    base = sum(ord(c) for c in question)
    prec_shift = {"fp16": 0, "int8": 1, "nf4": 2}.get(lm.precision, 0)
    ctx_shift = 3 if context else 0
    return [
        ((base + i * 7 + prec_shift * 5 + ctx_shift * 3) % 11) / 11.0
        for i in range(len(options))
    ]


@pytest.fixture
def patched(monkeypatch):
    # score_model does `from .scorer import ...` at call time, so patching the
    # scorer module attributes intercepts the real GPU path with a CPU stub.
    import culturaquant.scorer as scorer
    monkeypatch.setattr(scorer, "load_model", lambda n, p: _StubModel(n, p))
    monkeypatch.setattr(scorer, "free_model", lambda lm: None)
    monkeypatch.setattr(scorer, "score_item_logprobs", _stub_logprobs)
    # reference_machine touches torch.cuda; stub it for a clean CPU run
    monkeypatch.setattr(run_gate, "reference_machine", lambda: {"stub": True})


def _items():
    cult = [
        Item(id=f"c{i}", group="cultural", stratum="cuisine", region="NE",
             question=f"pergunta cultural numero {i} sobre comida", options=tuple("abcde"),
             gold_index=i % 5)
        for i in range(6)
    ]
    ctrl = [
        Item(id=f"k{i}", group="control", stratum="generic", region="-",
             question=f"pergunta controle numero {i} de geografia", options=tuple("abcde"),
             gold_index=i % 5)
        for i in range(4)
    ]
    return cult + ctrl


def _corpus():
    return [
        Passage(id=f"p{i}", title=f"t{i}",
                text=f"comida cultural numero {i} prato tipico brasileiro regiao")
        for i in range(8)
    ]


def test_run_gate_writes_then_skips(patched, tmp_path):
    items = _items()
    passages = _corpus()
    retriever = BM25Retriever(passages)
    retrieved = run_gate.prepare_retrieval(items, retriever, top_k=1)
    corpus_meta = {"corpus_sha256": "deadbeef", "source": "stub", "n_passages": len(passages),
                   "corpus_file": "corpus.jsonl"}
    retr_cfg = {"kind": "bm25", "top_k": 1}
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    res1 = run_gate.run_one_model(
        "stub/model", items, retriever, retrieved, out_dir, corpus_meta, retr_cfg,
        ("fp16", "int8", "nf4"), 1,
    )
    assert res1["status"] == "done"
    out_file = out_dir / "gate__stub__model.json"
    assert out_file.exists()

    blob = json.loads(out_file.read_text())
    assert len(blob["records"]) == len(items)
    # every record has logprobs at all 3 precisions + RAG, and no NaN in QDIS
    for r in blob["records"]:
        assert len(r["logprobs_int8"]) == 5
        assert len(r["logprobs_nf4"]) == 5
        assert len(r["logprobs_rag_int8"]) == 5
        assert r["qdis"]["jsd"] == r["qdis"]["jsd"]  # not NaN
    # all five budgets present for both gates
    for gate in run_gate.GATES:
        for b in run_gate.BUDGETS:
            assert f"{b:.2f}" in blob["conditions"]["gates"][gate]

    # second run with matching unit hash skips
    res2 = run_gate.run_one_model(
        "stub/model", items, retriever, retrieved, out_dir, corpus_meta, retr_cfg,
        ("fp16", "int8", "nf4"), 1,
    )
    assert res2["status"] == "skipped"


def test_run_gate_rescores_on_hash_change(patched, tmp_path):
    items = _items()
    retriever = BM25Retriever(_corpus())
    retrieved = run_gate.prepare_retrieval(items, retriever, top_k=1)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    base = dict(source="stub", n_passages=8, corpus_file="corpus.jsonl")
    retr_cfg = {"kind": "bm25", "top_k": 1}

    run_gate.run_one_model("stub/model", items, retriever, retrieved, out_dir,
                           {**base, "corpus_sha256": "aaa"}, retr_cfg,
                           ("int8", "nf4"), 1)
    # a different corpus hash invalidates the cached unit -> rescored
    res = run_gate.run_one_model("stub/model", items, retriever, retrieved, out_dir,
                                 {**base, "corpus_sha256": "bbb"}, retr_cfg,
                                 ("int8", "nf4"), 1)
    assert res["status"] == "done"
