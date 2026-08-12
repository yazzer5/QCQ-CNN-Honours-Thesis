#!/bin/bash
# Fashion-MNIST grid: 4 models x 5 seeds = 20 jobs for one class pair.
#
#   ./submit_run6_fashion.sh 0,6   <- hardest pair, ~16 pp headroom (RECOMMENDED)
#   ./submit_run6_fashion.sh 0,1   <- the pair Thesis A specifies (saturated)
#
# Run 0,6 for the scientific question and 0,1 for protocol fidelity. They
# write to separate results directories and cannot be mixed.
set -e
CLASSES="${1:-0,6}"
echo "Class pair: $CLASSES"
[ "$CLASSES" = "0,1" ] && echo "  NOTE: this pair is saturated (M4 98.8%). Expect a null result."
for MODEL in M1 M2 M3 M4; do
    for SEED in 42 43 44 45 46; do
        echo "Submitting $MODEL / fashion_mnist / classes=$CLASSES / seed=$SEED"
        qsub -v MODEL=$MODEL,DATASET=fashion_mnist,SEED=$SEED,CLASSES=$CLASSES run6_fashion_grid.pbs
    done
done
echo ""
echo "Collect:  python scripts/collect_results.py results_v6_fashion_c${CLASSES//,/}"
