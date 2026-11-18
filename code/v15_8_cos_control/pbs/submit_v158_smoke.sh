#!/bin/bash
# v15.8 SMOKE (9 jobs): M6 seed 42 at all 6 (cell, sigma_train) + M3/M4/M5 seed 42 at K=5 A, sigma=0.3
# (reproductions of v15.5: must be bit-exact). Resumable: skips done or queued jobs.
set -e
cd "$(dirname "$0")"
source ./config158.sh; source ./names158.sh
grep -q "ALL CHECKS PASSED" ../verify_v158.log 2>/dev/null || { echo "REFUSED: ../verify_v158.log does not end in ALL CHECKS PASSED (run setup_v158.sh)"; exit 1; }
Q=$(queued_names)
todo=()
add() {   # K TAG SG MODEL
  local NAME OUT; NAME=$(smoke_name $1 $2 $3 $4); OUT=$(out_dir $1 $2 $3)
  if ls ../$OUT/result_${4}_*seed${SMOKE_SEED}_*.json >/dev/null 2>&1 || grep -qx "$NAME" <<<"$Q"; then return; fi
  todo+=("$1 $2 $3 $4")
}
for K in $GRID_KS; do for TAG in $(tags_for $K); do for SG in $GRID_SIGMAS; do for M in $NEW_MODELS; do
  add $K $TAG $SG $M
done; done; done; done
for M in $REUSED_MODELS; do add $REPRO_K $REPRO_TAG $REPRO_SIGMA $M; done
inode_guard ${#todo[@]} || exit 1
n=0
for t in "${todo[@]}"; do
  set -- $t; K=$1; TAG=$2; SG=$3; M=$4
  NAME=$(smoke_name $K $TAG $SG $M); OUT=$(out_dir $K $TAG $SG)
  qsub -N "$NAME" -l walltime=$WALLTIME -v K=$K,TAG=$TAG,SG=$SG,MODEL=$M,SEED=$SMOKE_SEED,OUT=$OUT,CONT=0 run158.pbs >/dev/null
  n=$((n+1))
done
echo "Submitted $n smoke jobs (the rest already done or queued)."
echo "When qstat is empty: ./check_v158.sh smoke"
