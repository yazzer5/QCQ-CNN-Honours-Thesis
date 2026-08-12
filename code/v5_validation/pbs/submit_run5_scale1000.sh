#!/bin/bash
# SCALE-1000: 1000/250/200, seed 42, all four models, dropout 0.0, bs 32.
# Intermediate validation step before the full 5000/1250/1000 grid.
# Usage: ./submit_run5_scale1000.sh
set -e
for MODEL in M1 M2 M3 M4; do
    echo "Submitting $MODEL / mnist / seed=42 (scale-1000)"
    qsub -v MODEL=$MODEL,DATASET=mnist,SEED=42,DROPOUT=0.0 run5_scale1000.pbs
done
echo ""
echo "Read with:"
echo "  grep -E 'MODEL=|INVERSION|fe_gradvar|FINAL' v5_s1000.o*"
