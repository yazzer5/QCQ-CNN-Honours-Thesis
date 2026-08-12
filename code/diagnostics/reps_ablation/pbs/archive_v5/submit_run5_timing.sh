#!/bin/bash
# STEP 1 OF 3 — timing run, full data volume, 3 epochs.
# Usage: ./submit_run5_timing.sh [dataset] [seed]
set -e
DATASET="${1:-mnist}"
SEED="${2:-42}"
for MODEL in M1 M2 M3 M4; do
    echo "Submitting timing run: $MODEL / $DATASET / seed=$SEED"
    qsub -v MODEL=$MODEL,DATASET=$DATASET,SEED=$SEED run5_timing.pbs
done
echo ""
echo "Then, from the repo root:"
echo "  grep -E '\[GRADCHECK\]|epoch|FINAL' *.o*"
echo "Read the per-epoch time before submitting the grid."
