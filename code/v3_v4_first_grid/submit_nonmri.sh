#!/bin/bash
# submit_nonmri.sh — resubmit Phase 1 non-MRI jobs only (40 jobs)
# Use after pre-downloading datasets on the login node.

set -e

MODELS=("M1" "M2" "M3" "M4")
DATASETS=("mnist" "fashion_mnist")
SEEDS=(42 43 44 45 46)
ALPHA=0.0

LOGDIR=/g/data/wa66/yh2481/qcq_cnn/logs
mkdir -p "$LOGDIR"

count=0
for MODEL in "${MODELS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      count=$((count+1))
      JOBNAME="${MODEL}_${DATASET}_s${SEED}"
      qsub -N "$JOBNAME" \
           -v MODEL="$MODEL",DATASET="$DATASET",SEED="$SEED",ALPHA="$ALPHA" \
           run_single.pbs
      echo "[$count/40] queued $JOBNAME"
    done
  done
done

echo "Done. Submitted $count jobs. Check with: qstat -u yh2481"
