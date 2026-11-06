#!/bin/bash
# STEP 2 for a class count. Refuses unless the smoke gate has passed
# (it writes .smoke_ok_k$K).
#   K=5 : 2 subsets x {w6,w15} x {M2,M3} x 10 seeds = 80, minus 4 smoke = 76 jobs
#   K=10: {w6,w15} x {M2,M3} x 10 seeds = 40, minus 2 smoke = 38 jobs
#         + width-3 top-up: M1-M4 x TOPUP_SEEDS (5 seeds) = 20 jobs
set -e
cd "$(dirname "$0")"
source ./config13.sh
K="${1:?usage: ./submit_v13_grid.sh K}"
[ -f ".smoke_ok_k$K" ] || { echo "REFUSING: smoke gate for K=$K has not passed (./check_v13.sh $K smoke)."; exit 1; }
WT=$(walltime_for "$K")
N=0
for TAG in $(subset_tags "$K"); do
  for W in $WIDTHS_NEW; do
    for MODEL in $MODELS_NEW; do
      for SEED in $SEEDS; do
        [ "$MODEL" = "M2" ] && [ "$SEED" = "42" ] && continue
        qsub -N "v13_k${K}${TAG}w${W}" -l walltime=$WT \
             -v K=$K,TAG=$TAG,W=$W,MODEL=$MODEL,SEED=$SEED,OUT=$(results_dir $K $TAG $W),CONT=0 run13.pbs >/dev/null
        N=$((N+1))
      done
    done
  done
done
if [ "$K" = "10" ]; then
  for MODEL in $TOPUP_MODELS; do
    for SEED in $TOPUP_SEEDS; do
      qsub -N "v13_k10allw3" -l walltime=$WT \
           -v K=10,TAG=all,W=3,MODEL=$MODEL,SEED=$SEED,OUT=$(results_dir 10 all 3),CONT=0 run13.pbs >/dev/null
      N=$((N+1))
    done
  done
fi
echo "Submitted $N jobs for K=$K.   Check with:  ./check_v13.sh $K grid"
