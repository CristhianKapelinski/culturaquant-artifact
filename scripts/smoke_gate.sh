#!/usr/bin/env bash
# Smoke test for the QDIS selective-retrieval gate (EXP_A, spec section 8).
# Blocks the full grid until green. Target: < 3 min on one RTX 3060 (Ampere sm_86,
# where bitsandbytes int8/NF4 kernels are best supported).
#
# 1 model (Qwen3-1.7B), ~10 items (6 cultural across strata + 4 control), all
# conditions + oracle, BM25 k=1, single budget B=0.3 (all budgets are derived
# post-hoc, so 0.3 is just the headline reported one). Proves:
#   - build_prompt(context='') is byte-identical to the no-context prompt
#   - int8 + NF4 passes each yield 5 option logprobs; QDIS has no NaN
#   - BM25 returns a non-empty passage for every cultural query; RAG scores
#   - gate decisions / retrieved ids / per-condition correct flags are re-loadable
#   - a re-run skips the model (idempotency)
#   - the always_rag - closed_book cultural gap precheck (headroom to gate)
#
# Runs in the pinned Docker image on a GPU host; growing data goes to the
# bind-mounted $CQ_DATA dir, never the image. Usage:
#   CQ_DATA=~/cq_data ./scripts/smoke_gate.sh
# CPU-only dry run of the data-contract asserts (no GPU, mocked scorer) is the
# unit test tests/test_run_gate_idempotent.py; this script exercises the real
# quantized passes.
set -euo pipefail

IMAGE="${IMAGE:-culturaquant:1.0}"
CQ_DATA="${CQ_DATA:-$HOME/cq_data}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
SEED="${SEED:-20260607}"
# Cap the corpus scan so the BM25 build stays well under a minute on the smoke run.
MAX_ARTICLES="${MAX_ARTICLES:-4000}"

OUT="/data/out/gate_smoke"
CORPUS="/data/out/corpus_smoke"
mkdir -p "$CQ_DATA/out" "$CQ_DATA/hf"

run_gate() {
  docker run --rm --gpus all \
    -v "$CQ_DATA/out:/data/out" -v "$CQ_DATA/hf:/data/hf" \
    -e HF_HOME=/data/hf \
    "$IMAGE" python -m culturaquant.run_gate \
    --models "$MODEL" \
    --data-dir /app/data --out-dir "$OUT" --corpus-dir "$CORPUS" \
    --retriever bm25 --top-k 1 --chunk-tokens 200 \
    --max-articles "$MAX_ARTICLES" --max-cultural 6 --max-control 4 --seed "$SEED"
}

echo "[smoke] first run (scores the model, builds corpus, derives all conditions)"
t0=$(date +%s)
FIRST_LOG="$(run_gate)"
echo "$FIRST_LOG"
echo "$FIRST_LOG" | grep -q "'status': 'done'" || { echo "[smoke] FAIL: first run did not score"; exit 1; }

echo "[smoke] data-contract assertions"
docker run --rm -v "$CQ_DATA/out:/data/out" "$IMAGE" \
  python /app/scripts/_smoke_assert.py "$OUT"

echo "[smoke] second run (must skip; idempotency)"
SECOND_LOG="$(run_gate)"
echo "$SECOND_LOG"
echo "$SECOND_LOG" | grep -q "'status': 'skipped'" || {
  echo "[smoke] FAIL: re-run did not skip the model (idempotency broken)"; exit 1; }

t1=$(date +%s)
echo "[smoke] PASS in $((t1 - t0))s (target < 180s on an RTX 3060)"
