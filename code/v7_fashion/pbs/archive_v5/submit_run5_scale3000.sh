#!/bin/bash
# SCALE-3000: 3000/750/600, seed 42, all four models.
# Final rehearsal before the full 5000/1250/1000 grid.
# Usage: ./submit_run5_scale3000.sh
set -e
for MODEL in M1 M2 M3 M4; do
    echo "Submitting $MODEL / mnist / seed=42 (scale-3000)"
    qsub -v MODEL=$MODEL,DATASET=mnist,SEED=42,DROPOUT=0.0 run5_scale3000.pbs
done
echo ""
echo "Read with:"
echo "  grep -E 'MODEL=|INVERSION|FINAL' v5_s3000.o*"
echo "  grep -E 'epoch' v5_s3000.o<M2jobid> | head -20   # check plateau exits ~epoch 2"
