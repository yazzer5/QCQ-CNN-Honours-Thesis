#!/bin/bash
# SMOKE TEST — one full-scale job before the grid. Non-negotiable workflow rule.
#
#   ./submit_run8_smoke.sh 2,6
#
# Submits exactly ONE job: M2 / seed 42 / the given class pair, at the full
# locked protocol. It uses run8_fashion_grid.pbs -- the SAME script the grid
# uses -- so the smoke test exercises the identical code path rather than a
# parallel one that could drift from it.
#
# Why M2 / seed 42:
#   * M2 is the slowest model and exercises the most machinery (trainable
#     quantum front-end AND quantum back-end), so it is the job most likely
#     to expose a walltime or device problem.
#   * Seed 42 for 0,6 has an independent reference value already on record
#     (v7 smoke 83.60%, grid 82.80%), and the 2,6 M4 pilot gave 87.10%, so
#     the result has something to be checked against rather than being read
#     in isolation.
#
# BEFORE READING ANY ACCURACY NUMBER, check the integrity outputs:
#   grep 'CLASS PAIR'   v8_fash.o*    -> must show the pair you asked for
#   grep 'CODE VERSION' v8_fash.o*    -> must show v8
#   grep 'PARAM'        v8_fash.o*    -> M2 must be 1,610
#   grep 'FULL PROTOCOL' v8_fash.o*   -> must NOT say REDUCED-SCALE
#   grep -c 'INVERSION' v8_fash.o*    -> must be 0
#   grep 'GAP@BEST'     v8_fash.o*    -> must be present (this is new in v8)
# Only then look at FINAL test_acc.
set -e
CLASSES="${1:-2,6}"
CLASSPAIR="${CLASSES//,/_}"

case "$CLASSES" in
    [0-9],[0-9]) ;;
    *) echo "FATAL: class pair must be 'a,b' with single digits (got '$CLASSES')"; exit 1 ;;
esac
[ "${CLASSES%,*}" = "${CLASSES#*,}" ] && { echo "FATAL: class pair must be two DIFFERENT classes"; exit 1; }

echo "SMOKE TEST — M2 / seed 42 / classes=$CLASSES (passed as CLASSPAIR=$CLASSPAIR)"
qsub -v MODEL=M2,DATASET=fashion_mnist,SEED=42,CLASSPAIR=$CLASSPAIR run8_fashion_grid.pbs
echo ""
echo "Do NOT submit the grid until this job returns rc=0 with 0 inversions,"
echo "the correct class pair, 1,610 params, FULL PROTOCOL, and a GAP@BEST line."
