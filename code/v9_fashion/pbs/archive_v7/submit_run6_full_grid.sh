#!/bin/bash
# FULL MNIST 3-vs-5 GRID: 4 models x 5 seeds = 20 jobs, locked protocol.
# These are the real results -- no overrides.
# Usage: ./submit_run6_full_grid.sh [dataset]
set -e
DATASET="${1:-mnist}"
for MODEL in M1 M2 M3 M4; do
    for SEED in 42 43 44 45 46; do
        echo "Submitting $MODEL / $DATASET / seed=$SEED"
        qsub -v MODEL=$MODEL,DATASET=$DATASET,SEED=$SEED run6_full_grid.pbs
    done
done
echo ""
echo "20 jobs submitted. Monitor:  qstat -u \$USER"
echo "Collect:     python scripts/collect_results.py results_v6_mnist"
echo "Sanity:      grep -E 'INVERSION|FINAL' v6_grid.o*"
