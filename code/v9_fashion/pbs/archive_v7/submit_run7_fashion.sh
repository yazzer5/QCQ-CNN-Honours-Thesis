#!/bin/bash
# Fashion-MNIST grid: 4 models x 5 seeds = 20 jobs for one class pair.
#
#   ./submit_run7_fashion.sh 0,6   <- hardest pair, ~16 pp headroom (RECOMMENDED)
#   ./submit_run7_fashion.sh 0,1   <- the pair Thesis A specifies (saturated)
#
# Run 0,6 for the scientific question and 0,1 for protocol fidelity. They
# write to separate results directories and cannot be mixed.
#
# v7 CHANGES vs v6:
#   1. The class pair is passed to PBS as CLASSPAIR=0_6, not CLASSES=0,6.
#      `qsub -v` takes a COMMA-separated list, so CLASSES=0,6 was parsed as
#      CLASSES=0 plus a stray variable "6". The job script converts the
#      underscore back to a comma.
#   2. The rc=99 continuation branch resubmits run7_fashion_grid.pbs and
#      carries CLASSPAIR through. v6 resubmitted run6_full_grid.pbs without
#      the class pair, silently reverting to an MNIST job.
set -e
CLASSES="${1:-0,6}"
CLASSPAIR="${CLASSES//,/_}"
CTAG="${CLASSES//,/}"

case "$CLASSES" in
    [0-9],[0-9]) ;;
    *) echo "FATAL: class pair must be 'a,b' with single digits (got '$CLASSES')"; exit 1 ;;
esac

echo "Class pair: $CLASSES  (passed to PBS as CLASSPAIR=$CLASSPAIR)"
[ "$CLASSES" = "0,1" ] && echo "  NOTE: this pair is saturated (M4 98.8%). Expect a null result."

for MODEL in M1 M2 M3 M4; do
    for SEED in 42 43 44 45 46; do
        echo "Submitting $MODEL / fashion_mnist / classes=$CLASSES / seed=$SEED"
        qsub -v MODEL=$MODEL,DATASET=fashion_mnist,SEED=$SEED,CLASSPAIR=$CLASSPAIR run7_fashion_grid.pbs
    done
done
echo ""
echo "20 jobs submitted. Monitor:  qstat -u \$USER"
echo "Sanity (MUST be 0):          grep -cE 'INVERSION' v7_fash.o*"
echo "Check the pair actually took: grep -h 'CLASSES=' v7_fash.o* | sort -u"
echo "Collect:                     python scripts/collect_results.py results_v7_fashion_c${CTAG}"
