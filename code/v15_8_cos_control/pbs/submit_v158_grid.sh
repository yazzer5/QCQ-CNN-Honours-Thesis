#!/bin/bash
# v15.8 GRID: M6 at every other seed x 6 (cell, sigma_train). Refuses without a passed smoke gate.
# Resumable: re-run if qsub stops at a queue limit.
set -e
cd "$(dirname "$0")"
source ./config158.sh; source ./names158.sh
[ -f .smoke_ok ] || { echo "REFUSED: the smoke gate has not passed (./check_v158.sh smoke)"; exit 1; }
Q=$(queued_names)
todo=(); skip=0
for K in $GRID_KS; do for TAG in $(tags_for $K); do for SG in $GRID_SIGMAS; do for M in $NEW_MODELS; do
  for S in $GRID_SEEDS; do
    [ "$S" = "$SMOKE_SEED" ] && continue
    NAME=$(grid_name $K $TAG $SG $M $S); OUT=$(out_dir $K $TAG $SG)
    if ls ../$OUT/result_${M}_*seed${S}_*.json >/dev/null 2>&1 || grep -qx "$NAME" <<<"$Q"; then skip=$((skip+1)); continue; fi
    todo+=("$K $TAG $SG $M $S")
  done
done; done; done; done
inode_guard ${#todo[@]} || exit 1
n=0; fail=0
for t in "${todo[@]}"; do
  set -- $t; K=$1; TAG=$2; SG=$3; M=$4; S=$5
  NAME=$(grid_name $K $TAG $SG $M $S); OUT=$(out_dir $K $TAG $SG)
  if qsub -N "$NAME" -l walltime=$WALLTIME -v K=$K,TAG=$TAG,SG=$SG,MODEL=$M,SEED=$S,OUT=$OUT,CONT=0 run158.pbs >/dev/null; then
    n=$((n+1))
  else
    fail=1; break
  fi
done
echo "Submitted $n grid jobs ($skip already done or queued)."
[ $fail -eq 1 ] && echo "qsub refused (queue limit?). Wait for jobs to drain, then run ./submit_v158_grid.sh again."
echo "When qstat is empty: ./check_v158.sh grid"
