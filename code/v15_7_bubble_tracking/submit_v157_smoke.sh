#!/bin/bash
# v15.7 SMOKE: seed 42 x {M3,M4,M5} x train sigma {0,0.3}, K5 A = 6 jobs. Resumable.
set -e
cd "$(dirname "$0")"
source ./config155.sh; source ./names155.sh
MODELS="M3 M4 M5"; GRID_SIGMAS="0 0.3"; FILES_PER_JOB=5; K=5; TAG=A
smoke_name() { echo "v157s_k5A_s${1}_${2}"; }
grep -q "ALL CHECKS PASSED" ../verify_v157.log 2>/dev/null || { echo "REFUSED: ../verify_v157.log lacks ALL CHECKS PASSED"; exit 1; }
[ -f ../V157_TRACK_ON ] && [ -f ../V156_GEOM_ON ] || { echo "REFUSED: marker files missing"; exit 1; }
Q=$(queued_names); todo=()
for SG in $GRID_SIGMAS; do for M in $MODELS; do
  NAME=$(smoke_name $SG $M); OUT=$(out_dir $K $TAG $SG)
  if ls ../$OUT/track_${M}_seed${SMOKE_SEED}.json >/dev/null 2>&1 || grep -qx "$NAME" <<<"$Q"; then continue; fi
  todo+=("$SG $M")
done; done
inode_guard ${#todo[@]} || exit 1
n=0
for t in "${todo[@]}"; do set -- $t; SG=$1; M=$2; OUT=$(out_dir $K $TAG $SG)
  qsub -N "$(smoke_name $SG $M)" -l walltime=$WALLTIME -v K=$K,TAG=$TAG,SG=$SG,MODEL=$M,SEED=$SMOKE_SEED,OUT=$OUT,CONT=0 run155.pbs >/dev/null; n=$((n+1))
done
echo "Submitted $n v15.7 smoke jobs. When qstat is empty, from qcq_cnn_v157: python3 gate_v157.py smoke \$PWD ../qcq_cnn_v156 ../qcq_cnn_v155"
