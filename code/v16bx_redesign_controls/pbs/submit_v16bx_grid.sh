#!/bin/bash
# v16b-ext GRID: M4-trim and BFR-ZZ, seeds 43-51 = 18 jobs. Refuses without the smoke gate. Resumable.
set -e
cd "$(dirname "$0")"
source ./config16bx.sh; source ./names16bx.sh
[ -f .smoke_ok ] || { echo "REFUSED: the smoke gate has not passed (./check_v16bx.sh smoke)"; exit 1; }
Q=$(queued_names)
n=0; skip=0; fail=0
for M in $X_MODELS; do for S in $GRID_SEEDS; do
  [ "$S" = "$SMOKE_SEED" ] && continue
  OUT=$(out_dir x $M); NAME=$(grid_name $M $S)
  if ls ../$OUT/result_M4_*seed${S}_*.json >/dev/null 2>&1 || grep -qx "$NAME" <<<"$Q"; then skip=$((skip+1)); continue; fi
  if qsub -N "$NAME" -l walltime=$WALLTIME -v ARM=x,K=$XK,TAG=$XTAG,N=$XN,MODEL=$M,SEED=$S,OUT=$OUT,CONT=0 run16bx.pbs >/dev/null; then
    n=$((n+1))
  else
    fail=1; break 2
  fi
done; done
echo "Submitted $n grid jobs ($skip already done or queued). Expected 18 in total."
[ $fail -eq 1 ] && echo "qsub refused (queue limit?). Run ./submit_v16bx_grid.sh again later."
echo "When qstat is empty: ./check_v16bx.sh grid"
