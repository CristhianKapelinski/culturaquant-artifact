#!/usr/bin/env bash
# Run the QDIS gate experiment (EXP_A) in the pinned Docker image on a GPU host.
# Resumable: per-model output files are skipped when their unit hash matches, so a
# re-run resumes from the first missing model. Growing data (HF cache, corpus,
# per-model gate JSONLs) goes to the bind-mounted $CQ_DATA dir, never the image.
#
# Gating is POST-HOC: the four forward sweeps (fp16 + int8 + nf4 closed-book +
# int8 always-RAG) run once per model and ALL budgets {0.10..0.50} and both gates
# (QDIS, generic margin) are derived from the cached scores. The five budgets cost
# no extra forward passes.
#
# Host assignment: set MODELS to each host's slice and run the script there. The grid
# parallelizes across any set of ssh-reachable GPU hosts; a CPU-only host can build the
# corpus and run retrieval (CORPUS_ONLY=1). Example split across four GPU hosts plus one
# CPU host:
#   gpu-1:  Qwen/Qwen3-4B  google/gemma-2-2b-it          (+ bge-m3 embed, wave 2)
#   gpu-2:  Qwen/Qwen2.5-3B-Instruct  meta-llama/Llama-3.2-3B-Instruct
#   gpu-3:  Qwen/Qwen3-1.7B  Qwen/Qwen3-0.6B             (smoke test runs here first)
#   gpu-4:  Qwen/Qwen2.5-1.5B-Instruct
#   cpu-1:  corpus build + retrieval only (CORPUS_ONLY=1)
#
# bnb note: int8/NF4 kernels are best-supported on Ampere. If bitsandbytes is unconfirmed
# on a Blackwell (sm_120) host at smoke time, route the quantized passes (int8/nf4/
# always-RAG) to an Ampere host and use Blackwell only for the FP16 pass + dense embedding.
#
# Usage:
#   CQ_DATA=~/cq_data MODELS="Qwen/Qwen3-1.7B Qwen/Qwen3-0.6B" ./scripts/run_gate_waves.sh
#   CQ_DATA=~/cq_data CORPUS_ONLY=1 ./scripts/run_gate_waves.sh          # CPU host: build corpus
#   CQ_DATA=~/cq_data WAVE=2 DENSE=1 MODELS="..." ./scripts/run_gate_waves.sh
#   CQ_DATA=~/cq_data HF_TOKEN=hf_xxx MODELS="google/gemma-2-2b-it" ./scripts/run_gate_waves.sh
set -euo pipefail

IMAGE="${IMAGE:-culturaquant:1.0}"
CQ_DATA="${CQ_DATA:-$HOME/cq_data}"
SEED="${SEED:-20260607}"
WAVE="${WAVE:-1}"
TOPK="${TOPK:-1}"
CHUNK="${CHUNK:-200}"
# Full wave-1 grid default; override MODELS per host from the assignment above.
MODELS="${MODELS:-Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct \
  Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B}"
PRECISIONS="${PRECISIONS:-fp16 int8 nf4}"
RETRIEVER="${RETRIEVER:-bm25}"

OUT="/data/out/gate_wave${WAVE}"
CORPUS="/data/out/corpus_cultural"
mkdir -p "$CQ_DATA/out" "$CQ_DATA/hf"

# Wave 2 switches on the dense bge-m3 retriever as a robustness check.
if [[ "${WAVE}" == "2" && "${DENSE:-0}" == "1" ]]; then
  RETRIEVER="dense"
  OUT="/data/out/gate_wave2_dense"
fi

# HF token (for the gated Gemma/Llama models) is mounted via env only if set.
HF_ENV=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  HF_ENV=(-e "HF_TOKEN=${HF_TOKEN}" -e "HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}")
fi

# a CPU host builds the corpus + the BM25 index once; GPU hosts then reuse it.
if [[ "${CORPUS_ONLY:-0}" == "1" ]]; then
  echo "[gate-waves] CPU corpus build only"
  docker run --rm \
    -v "$CQ_DATA/out:/data/out" -v "$CQ_DATA/hf:/data/hf" -e HF_HOME=/data/hf \
    "$IMAGE" python -m culturaquant.run_gate \
    --models __none__ --data-dir /app/data --out-dir "$OUT" --corpus-dir "$CORPUS" \
    --retriever bm25 --top-k "$TOPK" --chunk-tokens "$CHUNK" --seed "$SEED" || true
  echo "[gate-waves] corpus at $CQ_DATA/out/${CORPUS##*/}"
  exit 0
fi

echo "[gate-waves] wave=$WAVE retriever=$RETRIEVER models: $MODELS"
docker run --rm --gpus all \
  -v "$CQ_DATA/out:/data/out" -v "$CQ_DATA/hf:/data/hf" \
  -e HF_HOME=/data/hf "${HF_ENV[@]}" \
  "$IMAGE" python -m culturaquant.run_gate \
  --models $MODELS \
  --data-dir /app/data --out-dir "$OUT" --corpus-dir "$CORPUS" \
  --precisions $PRECISIONS --retriever "$RETRIEVER" \
  --top-k "$TOPK" --chunk-tokens "$CHUNK" --seed "$SEED"

echo "[gate-waves] gate JSONLs written to $CQ_DATA/out/${OUT##*/}"
echo "[gate-waves] aggregate with:"
echo "  docker run --rm -v $CQ_DATA/out:/data/out $IMAGE \\"
echo "    python -m culturaquant.analyze_gate --out-dir $OUT \\"
echo "    --macro-out $OUT/results_gate_macros.tex --run-date \$(date +%F)"
