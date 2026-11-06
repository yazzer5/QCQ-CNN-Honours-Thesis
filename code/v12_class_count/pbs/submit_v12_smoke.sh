#!/bin/bash
# STEP 1 for a class count: ONE full-scale job per subset (M2, seed 42).
#   ./submit_v12_smoke.sh 3      -> 2 jobs (K=3 subset A and B)
# These results are kept and reused by the grid (grid skips M2/seed42).
set -e
cd "$(dirname "$0")"
source ./subsets.sh
K="${1:?usage: ./submit_v12_smoke.sh K   (K = 3, 4 or 5)}"
for TAG in $SUBSET_TAGS; do
  CL=$(subset_classes "$K" "$TAG")
  echo "K=$K subset $TAG classes=[$CL]"
  qsub -N "v12s_k${K}${TAG}" -v K=$K,TAG=$TAG,MODEL=M2,SEED=42,CONT=0 run12_classes.pbs
done
echo
echo "When both finish:  ./check_v12.sh $K smoke"
echo "Only if it prints ALL CHECKS PASSED:  ./submit_v12_grid.sh $K"
