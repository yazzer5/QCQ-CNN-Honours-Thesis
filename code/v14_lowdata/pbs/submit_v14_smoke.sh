#!/bin/bash
# THE smoke test (one submission, one gate). Full-scale, seed 42:
#   (a) every (K, subset, N, model) cell of the grid: 51 jobs, kept and reused by the grid
#       (N=50/100/250: M1-M5; N=500: M1, M4 -- the two never run at w15)
#   (b) 2 reproductions at N=500, w15, K=5 A, seed 42, in the PATCHED repo:
#       M3 must equal v13 exactly, M5 must equal v13.5 exactly.
#   ./submit_v14_smoke.sh      -> 53 jobs
set -e
cd "$(dirname "$0")"
source ./config14.sh; source ./names14.sh
for M in $REPRO_MODELS; do
  REF=$(repro_ref "$M")
  [ -f "$REF" ] || { echo "REFUSING: reproduction reference not found: $REF (fix the *_ROOT paths in config14.sh)"; exit 1; }
done
[ -f .smoke_submitted ] && { echo "REFUSING: smoke already submitted ($(cat .smoke_submitted)). Delete .smoke_submitted only if you mean to resubmit."; exit 1; }
N_JOBS=0
for M in $REPRO_MODELS; do
  qsub -N "$(repro_name $M)" -l walltime=$(walltime_for 500) \
       -v K=5,TAG=A,N=500,MODEL=$M,SEED=42,OUT=$(repro_dir $M),CONT=0 run14.pbs
  N_JOBS=$((N_JOBS+1))
done
for K in $GRID_KS; do for TAG in $(subset_tags "$K"); do for N in $GRID_NS; do for M in $(models_for $N); do
  qsub -N "$(smoke_name $K $TAG $N $M)" -l walltime=$(walltime_for $N) \
       -v K=$K,TAG=$TAG,N=$N,MODEL=$M,SEED=$SMOKE_SEED,OUT=$(results_dir $K $TAG $N),CONT=0 run14.pbs >/dev/null
  N_JOBS=$((N_JOBS+1))
done; done; done; done
date > .smoke_submitted
echo
echo "Submitted $N_JOBS smoke jobs (K = $GRID_KS, N = $GRID_NS, width $W_OBS, seed $SMOKE_SEED, + $REPRO_MODELS reproductions)."
echo "When qstat is empty:  ./check_v14.sh smoke"
