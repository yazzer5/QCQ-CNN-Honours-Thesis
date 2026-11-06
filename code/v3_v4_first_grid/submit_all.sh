#!/bin/bash
# submit_all.sh — Phase 1: 4 models x 3 datasets x 5 seeds x alpha=0.0 = 60 jobs
# PBS job arrays are NOT permitted on Gadi, so we qsub each combo separately.

set -e

MODELS=("M1" "M2" "M3" "M4")
DATASETS=("mnist" "fashion_mnist" "mri")
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
      echo "[$count/60] queued $JOBNAME"
    done
  done
done

echo "Done. Submitted $count jobs. Check with: qstat -u yh2481"
