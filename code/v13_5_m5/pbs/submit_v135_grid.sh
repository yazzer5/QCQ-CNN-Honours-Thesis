#!/bin/bash
# STEP 2. Refuses any K whose smoke gate has not passed (.smoke_ok_k$K), and
# K=5 additionally needs the reproduction to have passed (.repro_ok).
#   K=3 : 2 subsets x w3 x 10 seeds - 2 smoke            = 18 jobs
#   K=4 : same                                            = 18
#   K=5 : 2 subsets x {w3,w6,w15} x 10 seeds - 6 smoke    = 54
#   K=10: {w3,w6,w15} x 10 seeds - 3 smoke                = 27
#   all                                                   = 117 jobs
#   ./submit_v135_grid.sh            (all K)      ./submit_v135_grid.sh 3 4   (some K)
set -e
cd "$(dirname "$0")"
source ./config135.sh
KLIST="${*:-$KS}"
for K in $KLIST; do
  [ -f ".smoke_ok_k$K" ] || { echo "REFUSING: smoke gate for K=$K has not passed (./check_v135.sh smoke $K)."; exit 1; }
done
[ -f ".repro_ok" ] || { echo "REFUSING: reproduction (M3 K=5 A w3 seed 42) has not passed."; exit 1; }
N=0
for K in $KLIST; do
  WT=$(walltime_for "$K")
  for TAG in $(subset_tags "$K"); do
    for W in $(widths_for "$K"); do
      for SEED in $SEEDS; do
        [ "$SEED" = "$SMOKE_SEED" ] && continue
        qsub -N "v135_k${K}${TAG}w${W}" -l walltime=$WT \
             -v K=$K,TAG=$TAG,W=$W,MODEL=$GRID_MODEL,SEED=$SEED,OUT=$(results_dir $K $TAG $W),CONT=0 run135.pbs >/dev/null
        N=$((N+1))
      done
    done
  done
done
echo "Submitted $N jobs for K = $KLIST.   Check with:  ./check_v135.sh grid"
