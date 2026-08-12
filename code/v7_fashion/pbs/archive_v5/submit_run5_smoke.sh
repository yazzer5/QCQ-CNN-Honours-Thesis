#!/bin/bash
# SMOKE-SCALE run: 500/125/100, seed 42 only, all 4 models.
# Confirms the pipeline before the full grid -- not a result.
# Usage: ./submit_run5_smoke.sh
set -e
for MODEL in M1 M2 M3 M4; do
    echo "Submitting smoke run: $MODEL / mnist / seed=42"
    qsub -v MODEL=$MODEL,DATASET=mnist,SEED=42 run5_smoke.pbs
done
echo ""
echo "Check with: qstat -u \$USER"
echo "Read results: grep -E '\[GRADCHECK\]|epoch|FINAL|Peak GPU' *.o*"
echo "If this looks healthy, move to submit_run5_timing.sh, then the full grid."
