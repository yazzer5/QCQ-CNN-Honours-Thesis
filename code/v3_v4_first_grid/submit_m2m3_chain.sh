#!/bin/bash
# submit_m2m3_chain.sh — launch M2 + M3 on mnist + fashion_mnist (20 combos)
# as PBS-native dependency chains, per NCI's documented pattern:
# https://opus.nci.org.au/display/Help/Job+Dependencies
#
# For each combo, this submits CHAIN_MAX links UP FRONT, each depending on
# the previous via `-W depend=afterany:<prev_jobid>`, so link N+1 only
# starts once link N has ended (converged early, hit its time budget, or
# crashed) -- never in parallel with it. All qsub calls happen here, from
# the login node, seconds apart at launch time -- no job resubmits itself
# from a compute node.
#
# MRI and the missing M1 seeds are deliberately OUT OF SCOPE here -- run
# those separately once M2/M3 MNIST/Fashion are done.

set -e

MODELS=("M2" "M3")
DATASETS=("mnist" "fashion_mnist")
SEEDS=(42 43 44 45 46)
ALPHA=0.0
CHAIN_MAX=8   # ceiling of 8 x 24h = 192h RESERVED per combo -- not what you
              # expect to actually burn. Once a combo converges, every
              # remaining pre-submitted link for it sees the result file
              # already exists (run_chain_link.pbs skip-check) and exits in
              # seconds, returning the unused walltime.

PROJECT_DIR=/g/data/wa66/yh2481/qcq_cnn
cd "$PROJECT_DIR"
mkdir -p results_phase1 checkpoints logs

count=0
for MODEL in "${MODELS[@]}"; do
  for DATASET in "${DATASETS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
      count=$((count+1))
      RESULT_FILE="results_phase1/result_${MODEL}_${DATASET}_seed${SEED}_alpha${ALPHA}.json"
      if [ -f "$RESULT_FILE" ]; then
        echo "[$count/20] SKIP ${MODEL}_${DATASET}_s${SEED} -- result already exists"
        continue
      fi

      PREV_JOBID=""
      for LINK in $(seq 1 "$CHAIN_MAX"); do
        JOBNAME="${MODEL}_${DATASET}_s${SEED}_c${LINK}"
        if [ -z "$PREV_JOBID" ]; then
          JOBID=$(qsub -N "$JOBNAME" \
                       -v MODEL="$MODEL",DATASET="$DATASET",SEED="$SEED",ALPHA="$ALPHA" \
                       run_chain_link.pbs)
        else
          JOBID=$(qsub -N "$JOBNAME" \
                       -W depend=afterany:"$PREV_JOBID" \
                       -v MODEL="$MODEL",DATASET="$DATASET",SEED="$SEED",ALPHA="$ALPHA" \
                       run_chain_link.pbs)
        fi
        echo "[$count/20] queued $JOBNAME  (jobid=$JOBID, depends_on=${PREV_JOBID:-none})"
        PREV_JOBID="$JOBID"
      done
    done
  done
done

echo ""
echo "Done. $count combo(s) x up to $CHAIN_MAX chained links each submitted."
echo "Monitor with: qstat -u yh2481"
echo "Checkpoints:  ls ${PROJECT_DIR}/checkpoints/"
echo "Results:      ls ${PROJECT_DIR}/results_phase1/"
