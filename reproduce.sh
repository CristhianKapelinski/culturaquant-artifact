#!/usr/bin/env bash
# Form 1 (no GPU): reproduce every paper number from the committed predictions.
# Recomputes the differential-erosion CIs, the position-bias RStd, the measurable
# set and the calibration gradient from data/predictions/, then regenerates the
# LaTeX macros and asserts they are byte-identical to results/results_macros.tex.
# Pure Python standard library; no GPU, no network, no install needed.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"

echo "== differential erosion + 95% CIs =="
"$PY" "$HERE/analysis/cq_ci.py"
echo
echo "== position-bias RStd =="
"$PY" "$HERE/analysis/cq_rstd.py"
echo
echo "== measurable set + calibration gradient =="
"$PY" "$HERE/analysis/cq_full_analysis.py"
"$PY" "$HERE/analysis/cq_calib.py"
echo
echo "== regenerate macros and verify byte-identical =="
REF="$HERE/results/results_macros.tex"
TMP="$(mktemp)"
"$PY" "$HERE/analysis/gen_macros.py" "$TMP" >/dev/null
if diff <(grep newcommand "$TMP" | sort) <(grep newcommand "$REF" | sort) >/dev/null; then
  echo "OK_MACROS_REPRODUCED (all macros byte-identical to results/results_macros.tex)"
else
  echo "MISMATCH: regenerated macros differ from results/results_macros.tex" >&2
  diff <(grep newcommand "$TMP" | sort) <(grep newcommand "$REF" | sort) >&2
  exit 1
fi
rm -f "$TMP"
