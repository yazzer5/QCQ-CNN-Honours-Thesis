#!/bin/bash
# v16a GRID: arm c seeds 43-51 (54 jobs) + arm z seeds 43-44 (8 jobs) = 62.
# Refuses without a passed smoke gate. Resumable: skips finished and queued jobs.
set -e
cd "$(dirname "$0")"
source ./config16a.sh; source ./names16a.sh
[ -f .smoke_ok ] || { echo "REFUSED: the smoke gate has not passed (./check_v16a.sh smoke)"; exit 1; }
Q=$(queued_names)
n=0; skip=0; fail=0
sub() {   # ARM K TAG N MODEL SEED
  local OUT NAME TM; OUT=$(out_dir $1 $2 $3 $4); NAME=$(grid_name $1 $2 $3 $4 $5 $6)
  TM=$5; [ "$TM" = "M2c" ] && TM=M2
  if ls ../$OUT/result_${TM}_*seed${6}_*.json >/dev/null 2>&1 || grep -qx "$NAME" <<<"$Q"; then skip=$((skip+1)); return 0; fi
  if qsub -N "$NAME" -l walltime=$WALLTIME -v ARM=$1,K=$2,TAG=$3,N=$4,MODEL=$5,SEED=$6,OUT=$OUT,CONT=0 run16a.pbs >/dev/null; then
    n=$((n+1)); return 0
  fi
  fail=1; return 1
}
for K in $C_KS; do for TAG in $(tags_for $K); do for N in $C_NS; do for S in $GRID_SEEDS; do
  [ "$S" = "$SMOKE_SEED" ] && continue
  sub c $K $TAG $N M2c $S || break 4
done; done; done; done
if [ $fail -eq 0 ]; then
  for N in $Z_NS; do for M in $Z_MODELS; do for S in $Z_SEEDS; do
    [ "$S" = "$SMOKE_SEED" ] && continue
    sub z $Z_K all $N $M $S || break 3
  done; done; done
fi
echo "Submitted $n grid jobs ($skip already done or queued). Expected 62 in total."
[ $fail -eq 1 ] && echo "qsub refused (queue limit?). Wait for jobs to drain, then run ./submit_v16a_grid.sh again."
echo "When qstat is empty: ./check_v16a.sh grid"
