#!/bin/bash
# v15 SMOKE: seed 42 at every (K, subset, alpha, model) cell = 63 jobs, one submission.
# The alpha=0 jobs double as the reproductions (must equal v14 / v13 / v13.5 exactly).
# Resumable: skips cells with a result or a queued job.
set -e
cd "$(dirname "$0")"
source ./config15.sh; source ./names15.sh
grep -q "ALL CHECKS PASSED" ../verify_v15.log 2>/dev/null || { echo "REFUSED: ../verify_v15.log does not end in ALL CHECKS PASSED (run setup_v15.sh)"; exit 1; }
Q=$(queued_names)
n=0; skip=0
for K in $GRID_KS; do for TAG in $(tags_for $K); do for A in $GRID_ALPHAS; do for M in $(models_for_alpha $A); do
  NAME=$(smoke_name $K $TAG $A $M); OUT=$(out_dir $K $TAG $A)
  if ls ../$OUT/result_${M}_*seed${SMOKE_SEED}_*.json >/dev/null 2>&1 || grep -qx "$NAME" <<<"$Q"; then skip=$((skip+1)); continue; fi
  qsub -N "$NAME" -l walltime=$WALLTIME -v K=$K,TAG=$TAG,A=$A,MODEL=$M,SEED=$SMOKE_SEED,OUT=$OUT,CONT=0 run15.pbs >/dev/null
  n=$((n+1))
done; done; done; done
echo "Submitted $n smoke jobs ($skip already done or queued)."
echo "When qstat is empty: ./check_v15.sh smoke"
