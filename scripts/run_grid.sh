#!/usr/bin/env bash
# Run the full model x precision scoring grid in the pinned Docker image on a GPU host.
# Growing data (HF cache + predictions) goes to the bind-mounted $CQ_DATA dir, never
# inside the container or the image. Usage:
#   CQ_DATA=~/cq_data ./scripts/run_grid.sh           # full grid (8 models x 3 precisions)
#   CQ_DATA=~/cq_data QUICK=1 ./scripts/run_grid.sh   # quick: 2 models, fp16+nf4, 0 proverbs
set -euo pipefail

IMAGE="${IMAGE:-culturaquant:1.0}"
CQ_DATA="${CQ_DATA:-$HOME/cq_data}"
SEED="${SEED:-20260607}"
mkdir -p "$CQ_DATA/out" "$CQ_DATA/hf"

if [[ "${QUICK:-0}" == "1" ]]; then
  MODELS="Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct"
  PRECISIONS="fp16 nf4"
  NPROV=0
  OUT="grid_quick"
else
  MODELS="Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct \
          Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B \
          TucanoBR/Tucano-1b1 TucanoBR/Tucano-2b4"
  PRECISIONS="fp16 int8 nf4"
  NPROV=80
  OUT="grid"
fi

docker run --rm --gpus all \
  -v "$CQ_DATA/out:/data/out" -v "$CQ_DATA/hf:/data/hf" \
  -e HF_HOME=/data/hf \
  "$IMAGE" python -m culturaquant.run \
  --models $MODELS --precisions $PRECISIONS \
  --data-dir /app/data --out-dir "/data/out/$OUT" --n-proverbs "$NPROV" --seed "$SEED"

echo "[run_grid] predictions written to $CQ_DATA/out/$OUT"
