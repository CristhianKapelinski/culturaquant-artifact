#!/usr/bin/env bash
# Analyze a prediction grid and emit the LaTeX results-macros block.
# Pure-Python, no GPU; runs inside the same image. Usage:
#   CQ_DATA=~/cq_data ./scripts/emit_macros.sh [grid|grid_quick]
set -euo pipefail

IMAGE="${IMAGE:-culturaquant:1.0}"
CQ_DATA="${CQ_DATA:-$HOME/cq_data}"
GRID="${1:-grid}"

docker run --rm \
  -v "$CQ_DATA/out:/data/out" \
  "$IMAGE" python -m culturaquant.analyze \
  --out-dir "/data/out/$GRID" \
  --macro-out "/data/out/$GRID/results_macros.tex" \
  --run-date "$(date +%F)"

echo "[emit_macros] wrote $CQ_DATA/out/$GRID/results_macros.tex"
