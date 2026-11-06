#!/bin/bash
# THE grid. Refuses without a passed smoke gate (.smoke_ok).
#   K=5 : 2 subsets x (3 N x 5 models + N=500 x 2 models) x 9 seeds = 306
#   K=10: 1 subset  x (3 N x 5 models + N=500 x 2 models) x 9 seeds = 153    total 459
# Safe to re-run: skips any job whose result JSON exists or whose name is already
# in qstat (use it to resume if qsub hits a queue limit part-way).
#   ./submit_v14_grid.sh          (every K in GRID_KS)
#   ./submit_v14_grid.sh 5        (one K; e.g. if the per-user queue limit bites)
set -e
cd "$(dirname "$0")"
source ./config14.sh; source ./names14.sh
[ -f .smoke_ok ] || { echo "REFUSING: smoke gate has not passed (./check_v14.sh smoke)."; exit 1; }
KLIST="${*:-$GRID_KS}"
QUEUED=$(qstat -u "$USER" -f 2>/dev/null | awk '/Job_Name/{print $3}' | sort -u)
N_SUB=0; N_SKIP=0
for K in $KLIST; do for TAG in $(subset_tags "$K"); do for N in $GRID_NS; do for M in $(models_for $N); do
  for SEED in $GRID_SEEDS; do
    [ "$SEED" = "$SMOKE_SEED" ] && continue
    NAME=$(grid_name $K $TAG $N $M $SEED); OUT=$(results_dir $K $TAG $N)
    if [ -f "../$OUT/result_${M}_mnist_seed${SEED}_alpha0.0.json" ] || grep -qx "$NAME" <<<"$QUEUED"; then
      N_SKIP=$((N_SKIP+1)); continue
    fi
    qsub -N "$NAME" -l walltime=$(walltime_for $N) \
         -v K=$K,TAG=$TAG,N=$N,MODEL=$M,SEED=$SEED,OUT=$OUT,CONT=0 run14.pbs >/dev/null \
      || { echo "qsub failed after $N_SUB submissions (queue limit?). Wait for jobs to drain, then re-run this script: it resumes."; exit 1; }
    N_SUB=$((N_SUB+1))
  done
done; done; done; done
echo "Submitted $N_SUB grid jobs for K = $KLIST (skipped $N_SKIP already done or queued)."
echo "When qstat is empty:  ./check_v14.sh grid"
