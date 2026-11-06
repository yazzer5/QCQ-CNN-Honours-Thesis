#!/bin/bash
# v15 GRID: every other seed (43-51) at every cell. Refuses without a passed smoke gate.
# Resumable: re-run it if qsub stops at a queue limit; it skips finished and queued jobs.
set -e
cd "$(dirname "$0")"
source ./config15.sh; source ./names15.sh
[ -f .smoke_ok ] || { echo "REFUSED: the smoke gate has not passed (./check_v15.sh smoke)"; exit 1; }
Q=$(queued_names)
n=0; skip=0; fail=0
for K in $GRID_KS; do for TAG in $(tags_for $K); do for A in $GRID_ALPHAS; do for M in $(models_for_alpha $A); do
  for S in $GRID_SEEDS; do
    [ "$S" = "$SMOKE_SEED" ] && continue
    NAME=$(grid_name $K $TAG $A $M $S); OUT=$(out_dir $K $TAG $A)
    if ls ../$OUT/result_${M}_*seed${S}_*.json >/dev/null 2>&1 || grep -qx "$NAME" <<<"$Q"; then skip=$((skip+1)); continue; fi
    if qsub -N "$NAME" -l walltime=$WALLTIME -v K=$K,TAG=$TAG,A=$A,MODEL=$M,SEED=$S,OUT=$OUT,CONT=0 run15.pbs >/dev/null; then
      n=$((n+1))
    else
      fail=1; break 5
    fi
  done
done; done; done; done
echo "Submitted $n grid jobs ($skip already done or queued)."
[ $fail -eq 1 ] && echo "qsub refused (queue limit?). Wait for jobs to drain, then run ./submit_v15_grid.sh again."
echo "When qstat is empty: ./check_v15.sh grid"
