#!/bin/bash
# v15.5 SMOKE: seed 42 at every (K, subset, sigma, model) cell = 75 jobs, one submission.
# The sigma=0 jobs double as the reproductions (must equal v14 / v13 / v13.5 exactly).
# Resumable: skips cells with a result or a queued job.
set -e
cd "$(dirname "$0")"
source ./config155.sh; source ./names155.sh
grep -q "ALL CHECKS PASSED" ../verify_v155.log 2>/dev/null || { echo "REFUSED: ../verify_v155.log does not end in ALL CHECKS PASSED (run setup_v155.sh)"; exit 1; }
Q=$(queued_names)
todo=()
for K in $GRID_KS; do for TAG in $(tags_for $K); do for SG in $GRID_SIGMAS; do for M in $MODELS; do
  NAME=$(smoke_name $K $TAG $SG $M); OUT=$(out_dir $K $TAG $SG)
  if ls ../$OUT/result_${M}_*seed${SMOKE_SEED}_*.json >/dev/null 2>&1 || grep -qx "$NAME" <<<"$Q"; then continue; fi
  todo+=("$K $TAG $SG $M")
done; done; done; done
inode_guard ${#todo[@]} || exit 1
n=0
for t in "${todo[@]}"; do
  set -- $t; K=$1; TAG=$2; SG=$3; M=$4
  NAME=$(smoke_name $K $TAG $SG $M); OUT=$(out_dir $K $TAG $SG)
  qsub -N "$NAME" -l walltime=$WALLTIME -v K=$K,TAG=$TAG,SG=$SG,MODEL=$M,SEED=$SMOKE_SEED,OUT=$OUT,CONT=0 run155.pbs >/dev/null
  n=$((n+1))
done
echo "Submitted $n smoke jobs (the rest of the cells already done or queued)."
echo "When qstat is empty: ./check_v155.sh smoke"
