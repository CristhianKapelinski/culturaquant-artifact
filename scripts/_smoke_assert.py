"""Smoke-test assertions for the QDIS gate (spec section 8).

Loads the single smoke gate__*.json written by ``culturaquant-run-gate`` and
checks the must-prove list: the build_prompt regression invariant, no NaN in the
QDIS components, a non-empty BM25 passage for every cultural query, re-loadable
gate decisions / correct flags, and the always_rag-minus-closed_book cultural gap
precheck. Exits non-zero on any failure so the shell script blocks the grid.

The idempotency re-run check is done by the shell script (it re-invokes the run
and asserts every unit is skipped); this helper verifies the data contract.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from culturaquant.scorer import build_prompt

GAP_FLOOR_PP = 5.0  # spec: always_rag - closed_book cultural gap should exceed ~5pp
NF4_GAIN_TOL_PP = 8.0  # nf4 may differ from int8/fp16 by noise, never gain a lot


def _fail(msg: str) -> None:
    print(f"[smoke] FAIL: {msg}")
    sys.exit(1)


def _argmax(xs: list[float]) -> int:
    bi, bv = 0, -float("inf")
    for i, v in enumerate(xs):
        if v > bv:
            bv, bi = v, i
    return bi


def _acc(records: list[dict], key: str) -> float:
    vals = [int(_argmax(r[key]) == r["gold"]) for r in records if r.get(key)]
    return sum(vals) / len(vals) if vals else 0.0


def assert_nf4_sane(records: list[dict]) -> None:
    """Physical-plausibility guard on the nf4 scoring path.

    4-bit nf4 is a quantization of fp16/int8, so its accuracy must be a SMALL drop
    (or a tie within noise), never a large gain. A large nf4 gain over int8/fp16
    is the signature of the option-position bug (gold pinned at index 0 + an "A"
    bias) that this fix removes. We assert it on the cultural items, where the bug
    manifested. Runs only when fp16 logprobs are present (offline sweep)."""
    cult = [r for r in records if r["group"] == "cultural"]
    if not cult or not all(r.get("logprobs_fp16") for r in cult):
        print("[smoke] (nf4 sanity skipped: no fp16 sweep in this run)")
        return
    a_fp16 = _acc(cult, "logprobs_fp16")
    a_int8 = _acc(cult, "logprobs_int8")
    a_nf4 = _acc(cult, "logprobs_nf4")
    print(
        f"[smoke] cultural acc fp16={a_fp16:.3f} int8={a_int8:.3f} nf4={a_nf4:.3f}"
    )
    tol = NF4_GAIN_TOL_PP / 100.0
    if a_nf4 > a_int8 + tol:
        _fail(
            f"nf4 cultural acc {a_nf4:.3f} exceeds int8 {a_int8:.3f} by "
            f">{NF4_GAIN_TOL_PP}pp; non-physical nf4 gain (option-position bug?)"
        )
    if a_nf4 > a_fp16 + tol:
        _fail(
            f"nf4 cultural acc {a_nf4:.3f} exceeds fp16 {a_fp16:.3f} by "
            f">{NF4_GAIN_TOL_PP}pp; 4-bit must not beat fp16 by a large margin"
        )
    print(f"[smoke] OK: nf4 is a sane drop/tie (<= {NF4_GAIN_TOL_PP}pp over int8/fp16)")


def main() -> None:
    out_dir = Path(sys.argv[1])

    # 1. build_prompt regression invariant (context default / None / '' identical)
    q, opts = "Qual e a capital da Bahia?", ["Salvador", "B", "C", "D", "E"]
    base = build_prompt(q, opts)
    if build_prompt(q, opts, None) != base or build_prompt(q, opts, "") != base:
        _fail("build_prompt regression invariant broken (context='' != no-context)")
    print("[smoke] OK: build_prompt regression invariant")

    files = sorted(out_dir.glob("gate__*.json"))
    if not files:
        _fail(f"no gate__*.json under {out_dir}")
    blob = json.loads(files[0].read_text())
    records = blob["records"]
    cond = blob["conditions"]

    # 2. int8 + nf4 logprobs present, QDIS computes without NaN
    for r in records:
        for key in ("logprobs_int8", "logprobs_nf4", "logprobs_rag_int8"):
            v = r[key]
            if not v or len(v) != 5 or any(x != x for x in v):
                _fail(f"{r['id']}: bad/NaN {key}")
        for comp in ("flip", "jsd", "margin_min"):
            x = r["qdis"][comp]
            if x != x:
                _fail(f"{r['id']}: NaN QDIS component {comp}")
    print("[smoke] OK: int8/nf4 logprobs + QDIS components, no NaN")

    # 2b. nf4 physical-plausibility guard (no large nf4 gain over int8/fp16)
    assert_nf4_sane(records)

    # 3. BM25 returns a non-empty passage for every cultural query
    for r in records:
        if r["group"] == "cultural" and not r["retrieved"]:
            _fail(f"cultural item {r['id']} got no retrieved passage")
    print("[smoke] OK: BM25 non-empty for every cultural query")

    # 4. gate decisions + correct flags written and re-loadable for all budgets
    for gate in ("quant_aware", "generic"):
        for b in cond["gates"][gate]:
            cell = cond["gates"][gate][b]
            if len(cell["retrieve"]) != len(records) or len(cell["correct"]) != len(records):
                _fail(f"gate {gate}@{b}: bitmask/correct length mismatch")
    print("[smoke] OK: gate decisions + correct flags re-loadable")

    # 5. always_rag - closed_book cultural gap precheck (headroom to gate)
    cult = [i for i, r in enumerate(records) if r["group"] == "cultural"]
    cb = sum(cond["closed_book"][i] for i in cult) / len(cult)
    rag = sum(cond["always_rag"][i] for i in cult) / len(cult)
    gap_pp = 100.0 * (rag - cb)
    print(f"[smoke] cultural closed_book={cb:.2f} always_rag={rag:.2f} gap={gap_pp:+.1f}pp")
    if gap_pp < GAP_FLOOR_PP:
        print(
            f"[smoke] WARN: cultural always_rag-closed_book gap {gap_pp:.1f}pp "
            f"< {GAP_FLOOR_PP}pp; little headroom to gate on this smoke subset "
            "(spec risk: no headroom). Inspect before launching the full grid."
        )
    else:
        print(f"[smoke] OK: cultural gap {gap_pp:.1f}pp exceeds the {GAP_FLOOR_PP}pp floor")

    print("[smoke] all data-contract checks passed")


if __name__ == "__main__":
    main()
