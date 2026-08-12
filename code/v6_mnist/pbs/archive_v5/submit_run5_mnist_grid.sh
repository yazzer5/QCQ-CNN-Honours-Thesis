#!/bin/bash
# STEP 3 OF 3 — full MNIST 3-vs-5 grid: 4 models x 5 seeds = 20 jobs.
# Only run this after the timing run and the batch-size control.
# Usage: ./submit_run5_mnist_grid.sh
set -e
DATASET=mnist
for MODEL in M1 M2 M3 M4; do
    for SEED in 42 43 44 45 46; do
        echo "Submitting $MODEL / $DATASET / seed=$SEED"
        qsub -v MODEL=$MODEL,DATASET=$DATASET,SEED=$SEED run5_mnist_grid.pbs
    done
done
echo ""
echo "20 jobs submitted. Check with: qstat -u \$USER"
echo "Collect with: python scripts/collect_results.py results_v5_mnist"
