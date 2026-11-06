#!/bin/bash
# v16a SMOKE (one submission, one gate): seed 42, full scale.
#   arm c: one M2c job per (K, subset, N) cell        = 6
#   arm z: one M2 and one M3 job per N at K=10 (--save_z; reproductions of v13/v14) = 4
#   repro: M2, K=5 A, N=500 (must equal v13 exactly) = 1
# Resumable: skips cells with a result or a queued job.
set -e
cd "$(dirname "$0")"
source ./config16a.sh; source ./names16a.sh
grep -q "ALL CHECKS PASSED" ../verify_v16a.log 2>/dev/null || { echo "REFUSED: ../verify_v16a.log does not end in ALL CHECKS PASSED (run setup_v16a.sh)"; exit 1; }
[ -f ../PREREGISTRATION_v16a.md ] || { echo "REFUSED: PREREGISTRATION_v16a.md missing from the repo"; exit 1; }
miss=0
for K in $C_KS; do for TAG in $(tags_for $K); do for N in $C_NS; do for S in $GRID_SEEDS; do for M in M2 M3; do
  f="$(stored_dir $K $TAG $N)/result_${M}_mnist_seed${S}_alpha0.0.json"
  [ -f "$f" ] || { echo "MISSING stored comparator: $f"; miss=$((miss+1)); }
done; done; done; done; done
[ $miss -eq 0 ] || { echo "REFUSED: $miss stored M2/M3 result(s) missing. Fix V13_ROOT / V14_ROOT in config16a.sh."; exit 1; }
Q=$(queued_names)
n=0; skip=0
sub() {   # ARM K TAG N MODEL NAME
  local OUT; OUT=$(out_dir $1 $2 $3 $4)
  local TM=$5; [ "$TM" = "M2c" ] && TM=M2
  if ls ../$OUT/result_${TM}_*seed${SMOKE_SEED}_*.json >/dev/null 2>&1 || grep -qx "$6" <<<"$Q"; then skip=$((skip+1)); return; fi
  qsub -N "$6" -l walltime=$WALLTIME -v ARM=$1,K=$2,TAG=$3,N=$4,MODEL=$5,SEED=$SMOKE_SEED,OUT=$OUT,CONT=0 run16a.pbs >/dev/null
  n=$((n+1))
}
for K in $C_KS; do for TAG in $(tags_for $K); do for N in $C_NS; do
  sub c $K $TAG $N M2c "$(smoke_name c $K $TAG $N)"
done; done; done
for N in $Z_NS; do for M in $Z_MODELS; do
  sub z $Z_K all $N $M "$(smoke_name z $Z_K all $N $M)"
done; done
sub r 5 A 500 M2 "v16r_M2"
echo "Submitted $n smoke jobs ($skip already done or queued). Expected 11 in total."
echo "When qstat is empty: ./check_v16a.sh smoke"
