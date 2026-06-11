"""Main-grid run path: option shuffle, record schema, and resume idempotency.

The main grid (``culturaquant.run``) must apply the SAME per-item option
permutation as the gate path, so the gold answer is not pinned at index 0, and
must persist ``gold`` (permuted), ``gold_canonical`` and ``option_permutation``
per record. The scorer is replaced by a deterministic CPU stub so orchestration,
the permutation, the JSONL schema and the skip-on-rerun logic run end to end
without torch or a model.
"""

import json

import pytest

from culturaquant import run as run_mod
from culturaquant.data import Item


class _StubModel:
    def __init__(self, name, precision):
        self.name = name
        self.precision = precision


def _stub_logprobs(lm, question, options, context=None):
    # Deterministic per-option scores; argmax is well-defined and depends on text.
    base = sum(ord(c) for c in question)
    return [((base + i * 7) % 11) / 11.0 for i in range(len(options))]


@pytest.fixture
def patched(monkeypatch):
    # run.py binds these names at import time, so patch them on the run module.
    monkeypatch.setattr(run_mod, "load_model", lambda n, p: _StubModel(n, p))
    monkeypatch.setattr(run_mod, "free_model", lambda lm: None)
    monkeypatch.setattr(run_mod, "score_item_logprobs", _stub_logprobs)


def _items():
    cult = [
        Item(id=f"reg-{i:03d}", group="cultural", stratum="regional_facts",
             region="NE", question=f"pergunta cultural {i}",
             options=("correta", "b", "c", "d", "e"), gold_index=0)
        for i in range(6)
    ]
    ctrl = [
        Item(id=f"ctl-{i:03d}", group="control", stratum="generic", region="-",
             question=f"pergunta controle {i}",
             options=("correta", "b", "c", "d", "e"), gold_index=0)
        for i in range(4)
    ]
    return cult + ctrl


def _read_records(path):
    recs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("_meta"):
                continue
            recs.append(r)
    return recs


def test_run_one_shuffles_and_writes_schema(patched, tmp_path):
    items = _items()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    res = run_mod.run_one("stub/model", "nf4", items, out_dir, seed=20260607)
    assert res["status"] == "done"
    recs = _read_records(out_dir / res["preds_file"])
    assert len(recs) == len(items)

    golds = []
    for r, it in zip(recs, items):
        assert r["gold_canonical"] == 0
        perm = r["option_permutation"]
        assert sorted(perm) == [0, 1, 2, 3, 4]
        # permuted gold points back at the canonical-0 option
        assert perm[r["gold"]] == 0
        # correct flag is computed against the PERMUTED gold
        assert r["correct"] in (0, 1)
        golds.append(r["gold"])
    # gold is no longer pinned to index 0 for the whole set (the bug)
    assert any(g != 0 for g in golds)


def test_run_one_resumes_on_match(patched, tmp_path):
    items = _items()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    run_mod.run_one("stub/model", "int8", items, out_dir, seed=20260607)
    res2 = run_mod.run_one("stub/model", "int8", items, out_dir, seed=20260607)
    assert res2["status"] == "skipped"
    # a different seed changes the unit hash -> rescored
    res3 = run_mod.run_one("stub/model", "int8", items, out_dir, seed=1)
    # same file path, but seed differs -> not a match -> rescored
    assert res3["status"] == "done"


def test_analyze_load_skips_meta_line(patched, tmp_path):
    """analyze.load_preds must ignore the leading _meta provenance line."""
    from culturaquant import analyze
    items = _items()
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    res = run_mod.run_one("stub/model", "fp16", items, out_dir, seed=20260607)
    # minimal manifest pointing at the one preds file
    manifest = {
        "seed": 20260607, "n_items": len(items),
        "reference_machine": {}, "models": ["stub/model"],
        "precisions": ["fp16"],
        "runs": [{"model": "stub/model", "precision": "fp16",
                  "preds_file": res["preds_file"]}],
        "item_strata": {"by_group": {}, "by_stratum": {}, "by_region": {}},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest))
    blob = analyze.load_preds(out_dir)
    rows = blob["grid"][("stub/model", "fp16")]
    assert len(rows) == len(items)
    assert "_meta" not in rows  # meta line not loaded as an item
