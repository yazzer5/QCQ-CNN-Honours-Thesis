#!/bin/bash
# SMOKE TEST — one full-scale job before the grid. Non-negotiable workflow rule.
#
#   ./submit_run9_smoke.sh 2,4
#
# Submits exactly ONE job: M2 / seed 42 / the given class pair, at the full
# locked protocol. It uses run9_fashion_grid.pbs -- the SAME script the grid
# uses -- so the smoke test exercises the identical code path rather than a
# parallel one that could drift from it.
#
# Why M2 / seed 42:
#   * M2 is the slowest model and exercises the most machinery (trainable
#     quantum front-end AND quantum back-end), so it is the job most likely
#     to expose a walltime or device problem.
#   * Seed 42 has independent reference values already on record: M2/seed42
#     came in at 86.30% (v8 smoke, 2,6) and 86.74% was the 2,6 M2 5-seed
#     mean, so the result has something to be checked against rather than
#     being read in isolation. 2,4 is the easier pair, so expect >= that.
#
# BEFORE READING ANY ACCURACY NUMBER, check the integrity outputs:
#   grep 'CLASS PAIR'   v9_fash.o*    -> must show the pair you asked for
#   grep 'CODE VERSION' v9_fash.o*    -> must show v9
#   grep 'PARAM'        v9_fash.o*    -> M2 must be 1,610
#   grep 'FULL PROTOCOL' v9_fash.o*   -> must NOT say REDUCED-SCALE
#   grep -c 'INVERSION' v9_fash.o*    -> must be 0
#   grep 'GAP@BEST'     v9_fash.o*    -> must be present
# Only then look at FINAL test_acc.
set -e
CLASSES="${1:-2,4}"
CLASSPAIR="${CLASSES//,/_}"

case "$CLASSES" in
    [0-9],[0-9]) ;;
    *) echo "FATAL: class pair must be 'a,b' with single digits (got '$CLASSES')"; exit 1 ;;
esac
[ "${CLASSES%,*}" = "${CLASSES#*,}" ] && { echo "FATAL: class pair must be two DIFFERENT classes"; exit 1; }

echo "SMOKE TEST — M2 / seed 42 / classes=$CLASSES (passed as CLASSPAIR=$CLASSPAIR)"
qsub -v MODEL=M2,DATASET=fashion_mnist,SEED=42,CLASSPAIR=$CLASSPAIR run9_fashion_grid.pbs
echo ""
echo "Do NOT submit the grid until this job returns rc=0 with 0 inversions,"
echo "the correct class pair, 1,610 params, FULL PROTOCOL, and a GAP@BEST line."
