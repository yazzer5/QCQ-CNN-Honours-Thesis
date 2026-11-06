#!/bin/bash
# STEP 1 for a class count. Full-scale jobs, M2 seed 42:
#   (a) REPRODUCTION: width 3, subset A (K=5) / all (K=10). Must match the
#       existing v12/v10 result EXACTLY. Proves the patch left the width-3
#       path untouched and that the old results can be pooled.
#   (b) NEW PATH: width 6 and 15 on every subset. Kept and reused by the grid.
#   ./submit_v13_smoke.sh 5     -> 5 jobs      ./submit_v13_smoke.sh 10   -> 3 jobs
set -e
cd "$(dirname "$0")"
source ./config13.sh
K="${1:?usage: ./submit_v13_smoke.sh K   (K = 5 or 10)}"
WT=$(walltime_for "$K")
RTAG=$( [ "$K" = "10" ] && echo all || echo A )
REF=$(ref_result "$K" "$RTAG")
[ -f "$REF" ] || { echo "REFUSING: reference result not found: $REF (fix V12_ROOT / V10_RES in config13.sh)"; exit 1; }

qsub -N "v13s_k${K}${RTAG}r" -l walltime=$WT \
     -v K=$K,TAG=$RTAG,W=3,MODEL=M2,SEED=42,OUT=$(repro_dir $K $RTAG),CONT=0 run13.pbs
for TAG in $(subset_tags "$K"); do
  for W in $WIDTHS_NEW; do
    qsub -N "v13s_k${K}${TAG}w${W}" -l walltime=$WT \
         -v K=$K,TAG=$TAG,W=$W,MODEL=M2,SEED=42,OUT=$(results_dir $K $TAG $W),CONT=0 run13.pbs
  done
done
echo
echo "When all have left qstat:  ./check_v13.sh $K smoke"
echo "Only if it prints ALL CHECKS PASSED:  ./submit_v13_grid.sh $K"
