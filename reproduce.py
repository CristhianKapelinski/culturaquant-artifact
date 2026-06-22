#!/usr/bin/env python3
"""One no-GPU entry point: re-analyze the committed predictions and print the
paper's tables (differential erosion by quantization, the null result, the CIs),
then regenerate the LaTeX macros and assert they match the committed reference.

Pure Python standard library. No GPU, no network, no install needed:

    python3 reproduce.py

It runs the same four analyses as reproduce.sh, in order:
  1. analysis/cq_ci.py            per-model + pooled differential erosion with 95% CIs, MDE
  2. analysis/cq_rstd.py          position-choice spread (RStd) per measurable model
  3. analysis/cq_full_analysis.py measurable set + per-band / per-region cultural drops
  4. analysis/cq_calib.py         full-precision calibration gradient by rarity band
and then regenerates results/results_macros.tex into a temp file and checks every
\\newcommand is byte-identical to the committed reference. On success it prints
OK_MACROS_REPRODUCED and exits 0; on any drift it prints the diff and exits 1.
"""
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.join(HERE, "analysis")
PY = sys.executable or "python3"


def run(label, *argv):
    print(f"== {label} ==", flush=True)
    subprocess.run([PY, *argv], check=True)
    print(flush=True)


def newcommands(path):
    with open(path, encoding="utf-8") as fh:
        return sorted(ln for ln in fh if "newcommand" in ln)


def main():
    run("differential erosion + 95% CIs", os.path.join(ANALYSIS, "cq_ci.py"))
    run("position-bias RStd", os.path.join(ANALYSIS, "cq_rstd.py"))
    run("measurable set + per-band / per-region drops",
        os.path.join(ANALYSIS, "cq_full_analysis.py"))
    run("full-precision calibration gradient", os.path.join(ANALYSIS, "cq_calib.py"))

    print("== regenerate macros and verify byte-identical ==", flush=True)
    reference = os.path.join(HERE, "results", "results_macros.tex")
    fd, tmp = tempfile.mkstemp(suffix=".tex")
    os.close(fd)
    try:
        subprocess.run([PY, os.path.join(ANALYSIS, "gen_macros.py"), tmp],
                       check=True, stdout=subprocess.DEVNULL)
        if newcommands(tmp) == newcommands(reference):
            print("OK_MACROS_REPRODUCED "
                  "(all macros byte-identical to results/results_macros.tex)")
            return 0
        print("MISMATCH: regenerated macros differ from results/results_macros.tex",
              file=sys.stderr)
        regen, ref = set(newcommands(tmp)), set(newcommands(reference))
        for ln in sorted(regen - ref):
            print("regenerated only: " + ln.rstrip(), file=sys.stderr)
        for ln in sorted(ref - regen):
            print("reference only:   " + ln.rstrip(), file=sys.stderr)
        return 1
    finally:
        os.remove(tmp)


if __name__ == "__main__":
    raise SystemExit(main())
