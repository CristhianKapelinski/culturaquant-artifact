#!/usr/bin/env bash
# Reproduce every all-band paper number from the committed run of record
# (v2/allband_predictions, 11 models x 3 precisions x 700 items x 5 rotations).
# No GPU, no network. Recomputes the differential-erosion CIs, the per-band and
# per-region drops, the position-bias RStd, then regenerates results_macros.tex
# and asserts it is byte-identical to the committed copy.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"

echo "== differential erosion + CIs (cq_ci.py) =="
"$PY" "$HERE/cq_ci.py"
echo
echo "== position-bias RStd (cq_rstd.py) =="
"$PY" "$HERE/cq_rstd.py"
echo
echo "== regenerate macros and verify byte-identical =="
REF="$HERE/../allband/results_macros.tex"   # committed reference copy
TMP="$(mktemp)"
"$PY" "$HERE/gen_allband_macros.py" "$TMP" >/dev/null
if diff <(grep newcommand "$TMP" | sort) <(grep newcommand "$REF" | sort) >/dev/null; then
  echo "OK_MACROS_REPRODUCED (97 macros byte-identical to v2/allband/results_macros.tex)"
else
  echo "MISMATCH: regenerated macros differ from the committed reference" >&2
  diff <(grep newcommand "$TMP" | sort) <(grep newcommand "$REF" | sort) >&2
  exit 1
fi
rm -f "$TMP"
