#!/bin/bash
# v16b-ext SMOKE: M4-trim and BFR-ZZ at seed 42 (2) + reproduction BFR K=10 N=50 seed 42 == v16b (1) = 3 jobs.
set -e
cd "$(dirname "$0")"
source ./config16bx.sh; source ./names16bx.sh
grep -q "^ALL CHECKS PASSED" ../verify_v16bx.log 2>/dev/null || { echo "REFUSED: ../verify_v16bx.log lacks ALL CHECKS PASSED (run setup_v16bx.sh)"; exit 1; }
grep -q "A3 (2026-10-22" ../PREREGISTRATION_v16b.md 2>/dev/null || { echo "REFUSED: amendment A3 missing from ../PREREGISTRATION_v16b.md"; exit 1; }
miss=0
for S in $GRID_SEEDS; do
  for f in "$V14_ROOT/results_v14_k10_all_n50/result_M4_mnist_seed${S}_alpha0.0.json" \
           "$V14_ROOT/results_v14_k10_all_n50/result_M2_mnist_seed${S}_alpha0.0.json" \
           "$V16B_ROOT/results_v16b_bfr_k10_all_n50/result_M4_mnist_seed${S}_alpha0.0.json"; do
    [ -f "$f" ] || { echo "MISSING stored comparator: $f"; miss=$((miss+1)); }
  done
done
[ $miss -eq 0 ] || { echo "REFUSED: $miss stored comparator(s) missing"; exit 1; }
echo "Stored comparators: all 30 present (M4, M2-w15: v14; BFR: v16b; K=10 N=50, seeds 42-51)."
Q=$(queued_names)
n=0; skip=0
sub() {   # ARM MODEL NAME
  local OUT; OUT=$(out_dir $1 $2)
  if ls ../$OUT/result_M4_*seed${SMOKE_SEED}_*.json >/dev/null 2>&1 || grep -qx "$3" <<<"$Q"; then skip=$((skip+1)); return; fi
  qsub -N "$3" -l walltime=$WALLTIME -v ARM=$1,K=$XK,TAG=$XTAG,N=$XN,MODEL=$2,SEED=$SMOKE_SEED,OUT=$OUT,CONT=0 run16bx.pbs >/dev/null
  n=$((n+1))
}
for M in $X_MODELS; do sub x $M "$(smoke_name $M)"; done
sub r BFR "v16xr_BFR"
echo "Submitted $n smoke jobs ($skip already done or queued). Expected 3 in total."
echo "When qstat is empty: ./check_v16bx.sh smoke"
