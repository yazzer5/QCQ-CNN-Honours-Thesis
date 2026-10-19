#!/bin/bash
# STEP 1. Full-scale jobs, seed 42:
#   (a) REPRODUCTION: M3, K=5 subset A, width 3, in the PATCHED repo. Must match
#       v12's M3 seed-42 result EXACTLY. Proves the patch left M3 (and so M5's
#       starting point) untouched.
#   (b) M5 at every cell (13 jobs). Kept and reused by the grid.
#   ./submit_v135_smoke.sh          -> 14 jobs (all K)
#   ./submit_v135_smoke.sh 5        -> one K only (the repro job goes with K=5)
set -e
cd "$(dirname "$0")"
source ./config135.sh
KLIST="${1:-$KS}"
REF=$(ref_result)
[ -f "$REF" ] || { echo "REFUSING: reference result not found: $REF (fix V12_ROOT in config135.sh)"; exit 1; }
N=0
for K in $KLIST; do
  WT=$(walltime_for "$K")
  if [ "$K" = "$REPRO_K" ]; then
    qsub -N "v135s_k5Ar" -l walltime=$WT \
         -v K=$REPRO_K,TAG=$REPRO_TAG,W=$REPRO_W,MODEL=$REPRO_MODEL,SEED=42,OUT=$(repro_dir),CONT=0 run135.pbs
    N=$((N+1))
  fi
  for TAG in $(subset_tags "$K"); do
    for W in $(widths_for "$K"); do
      qsub -N "v135s_k${K}${TAG}w${W}" -l walltime=$WT \
           -v K=$K,TAG=$TAG,W=$W,MODEL=$GRID_MODEL,SEED=$SMOKE_SEED,OUT=$(results_dir $K $TAG $W),CONT=0 run135.pbs
      N=$((N+1))
    done
  done
done
echo
echo "Submitted $N smoke job(s) for K = $KLIST."
echo "When all have left qstat:  ./check_v135.sh smoke     (or ./check_v135.sh smoke K)"
echo "Only if it prints ALL CHECKS PASSED:  ./submit_v135_grid.sh"
