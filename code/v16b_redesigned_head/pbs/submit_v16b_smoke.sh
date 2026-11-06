#!/bin/bash
# v16b SMOKE (one submission, one gate): seed 42, full scale.
#   one BFR job per (K, subset, N) cell = 6;  repro: M4, K=5 A, N=500 (must equal v14 exactly) = 1
# Refuses without the verify log, the pre-registration in the repo, and every stored comparator.
# Resumable: skips cells with a result or a queued job.
set -e
cd "$(dirname "$0")"
source ./config16b.sh; source ./names16b.sh
grep -q "ALL CHECKS PASSED" ../verify_v16b.log 2>/dev/null || { echo "REFUSED: ../verify_v16b.log does not end in ALL CHECKS PASSED (run setup_v16b.sh)"; exit 1; }
[ -f ../PREREGISTRATION_v16b.md ] || { echo "REFUSED: PREREGISTRATION_v16b.md missing from the repo"; exit 1; }
miss=0
for K in $B_KS; do for TAG in $(tags_for $K); do for N in $B_NS; do for S in $GRID_SEEDS; do
  f="$(stored_m4_dir $K $TAG $N)/result_M4_mnist_seed${S}_alpha0.0.json"; [ -f "$f" ] || { echo "MISSING stored M4: $f"; miss=$((miss+1)); }
  f="$(stored_m2_dir $K $TAG $N)/result_M2_mnist_seed${S}_alpha0.0.json"; [ -f "$f" ] || { echo "MISSING stored M2: $f"; miss=$((miss+1)); }
done; done; done; done
[ $miss -eq 0 ] || { echo "REFUSED: $miss stored comparator result(s) missing. Fix V13_ROOT / V14_ROOT in config16b.sh."; exit 1; }
echo "Stored comparators: all 120 present (M4: v14 N=50/500; M2-w15: v14 N=50, v13 N=500)."
Q=$(queued_names)
n=0; skip=0
sub() {   # ARM K TAG N MODEL NAME
  local OUT; OUT=$(out_dir $1 $2 $3 $4)
  if ls ../$OUT/result_M4_*seed${SMOKE_SEED}_*.json >/dev/null 2>&1 || grep -qx "$6" <<<"$Q"; then skip=$((skip+1)); return; fi
  qsub -N "$6" -l walltime=$WALLTIME -v ARM=$1,K=$2,TAG=$3,N=$4,MODEL=$5,SEED=$SMOKE_SEED,OUT=$OUT,CONT=0 run16b.pbs >/dev/null
  n=$((n+1))
}
for K in $B_KS; do for TAG in $(tags_for $K); do for N in $B_NS; do
  sub b $K $TAG $N BFR "$(smoke_name $K $TAG $N)"
done; done; done
sub r 5 A 500 M4 "v16br_M4"
echo "Submitted $n smoke jobs ($skip already done or queued). Expected 7 in total."
echo "When qstat is empty: ./check_v16b.sh smoke"
