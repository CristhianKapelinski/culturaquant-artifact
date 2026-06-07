#!/usr/bin/env bash
# Regenerate every paper macro from the committed run of record and verify the
# output is byte-identical to data/results/results_macros.tex. No GPU, no network.
# Host path (uv-managed) by default; pass --docker to run in the pinned image
# (baked environment, no bind mount over /app).
set -euo pipefail

IMAGE="${IMAGE:-culturaquant:1.0}"
GRID="data/results/grid"
REF="data/results/results_macros.tex"
OUT="${OUT:-/tmp/cq_macros_repro.tex}"

run() {
  if [[ "${1:-}" == "--docker" ]]; then
    docker run --rm "$IMAGE" bash -c "
      python -m culturaquant.analyze --out-dir '$GRID' --macro-out /tmp/m.tex --run-date 2026-06-07 >/dev/null &&
      diff <(sort /tmp/m.tex) <(sort '$REF') && echo OK_MACROS_REPRODUCED"
  else
    uv run python -m culturaquant.analyze --out-dir "$GRID" --macro-out "$OUT" --run-date 2026-06-07 >/dev/null
    diff <(sort "$OUT") <(sort "$REF") && echo OK_MACROS_REPRODUCED
  fi
}

run "${1:-}"
