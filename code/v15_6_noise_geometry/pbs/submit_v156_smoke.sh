#!/bin/bash
# v15.6 SMOKE: seed 42 x {M3,M4,M5} x train sigma {0,0.3} x {K5 A, K5 B, K10} = 18 jobs.
# Each must reproduce the stored v15.5 run bit-exactly (gate_v156.py smoke). Resumable.
set -e
cd "$(dirname "$0")"
source ./config155.sh; source ./names155.sh
MODELS="M3 M4 M5"; GRID_SIGMAS="0 0.3"; FILES_PER_JOB=4          # v15.6 design (+ geom npz)
smoke_name() { echo "v156s_k${1}${2}_s${3}_${4}"; }
grep -q "ALL CHECKS PASSED" ../verify_v156.log 2>/dev/null || { echo "REFUSED: ../verify_v156.log lacks ALL CHECKS PASSED"; exit 1; }
[ -f ../V156_GEOM_ON ] || { echo "REFUSED: ../V156_GEOM_ON marker missing (geometry would not run)"; exit 1; }
grep -q "V156_GEOM begin" ../train_gadi.py || { echo "REFUSED: ../train_gadi.py is not v156-patched"; exit 1; }
Q=$(queued_names)
todo=()
for K in $GRID_KS; do for TAG in $(tags_for $K); do for SG in $GRID_SIGMAS; do for M in $MODELS; do
  NAME=$(smoke_name $K $TAG $SG $M); OUT=$(out_dir $K $TAG $SG)
  if ls ../$OUT/geom_${M}_seed${SMOKE_SEED}.npz >/dev/null 2>&1 || grep -qx "$NAME" <<<"$Q"; then continue; fi
  todo+=("$K $TAG $SG $M")
done; done; done; done
inode_guard ${#todo[@]} || exit 1
n=0
for t in "${todo[@]}"; do
  set -- $t; K=$1; TAG=$2; SG=$3; M=$4
  NAME=$(smoke_name $K $TAG $SG $M); OUT=$(out_dir $K $TAG $SG)
  qsub -N "$NAME" -l walltime=$WALLTIME -v K=$K,TAG=$TAG,SG=$SG,MODEL=$M,SEED=$SMOKE_SEED,OUT=$OUT,CONT=0 run155.pbs >/dev/null
  n=$((n+1))
done
echo "Submitted $n v15.6 smoke jobs."
echo "When qstat is empty: cd .. && python3 gate_v156.py smoke \$PWD ../qcq_cnn_v155"
