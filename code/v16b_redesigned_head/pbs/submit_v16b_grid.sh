#!/bin/bash
# v16b GRID: BFR seeds 43-51 on every (K, subset, N) cell = 54 jobs.
# Refuses without a passed smoke gate. Resumable: skips finished and queued jobs.
set -e
cd "$(dirname "$0")"
source ./config16b.sh; source ./names16b.sh
[ -f .smoke_ok ] || { echo "REFUSED: the smoke gate has not passed (./check_v16b.sh smoke)"; exit 1; }
Q=$(queued_names)
n=0; skip=0; fail=0
for K in $B_KS; do for TAG in $(tags_for $K); do for N in $B_NS; do for S in $GRID_SEEDS; do
  [ "$S" = "$SMOKE_SEED" ] && continue
  OUT=$(out_dir b $K $TAG $N); NAME=$(grid_name $K $TAG $N $S)
  if ls ../$OUT/result_M4_*seed${S}_*.json >/dev/null 2>&1 || grep -qx "$NAME" <<<"$Q"; then skip=$((skip+1)); continue; fi
  if qsub -N "$NAME" -l walltime=$WALLTIME -v ARM=b,K=$K,TAG=$TAG,N=$N,MODEL=BFR,SEED=$S,OUT=$OUT,CONT=0 run16b.pbs >/dev/null; then
    n=$((n+1))
  else
    fail=1; break 4
  fi
done; done; done; done
echo "Submitted $n grid jobs ($skip already done or queued). Expected 54 in total."
[ $fail -eq 1 ] && echo "qsub refused (queue limit?). Wait for jobs to drain, then run ./submit_v16b_grid.sh again."
echo "When qstat is empty: ./check_v16b.sh grid"
