#!/bin/bash
# v15.7 GRID: seeds 43-51 x {M3,M4,M5} x {0,0.3}, K5 A = 54 jobs. Refuses without ../.smoke_ok. Resumable.
set -e
cd "$(dirname "$0")"
source ./config155.sh; source ./names155.sh
MODELS="M3 M4 M5"; GRID_SIGMAS="0 0.3"; FILES_PER_JOB=5; K=5; TAG=A
grid_name() { echo "v157g_k5A_s${1}_${2}_r${3}"; }
[ -f ../.smoke_ok ] || { echo "REFUSED: smoke gate not passed"; exit 1; }
Q=$(queued_names); todo=(); skip=0
for SG in $GRID_SIGMAS; do for M in $MODELS; do for S in $GRID_SEEDS; do
  [ "$S" = "$SMOKE_SEED" ] && continue
  NAME=$(grid_name $SG $M $S); OUT=$(out_dir $K $TAG $SG)
  if ls ../$OUT/track_${M}_seed${S}.json >/dev/null 2>&1 || grep -qx "$NAME" <<<"$Q"; then skip=$((skip+1)); continue; fi
  todo+=("$SG $M $S")
done; done; done
inode_guard ${#todo[@]} || exit 1
n=0; fail=0
for t in "${todo[@]}"; do set -- $t; SG=$1; M=$2; S=$3; OUT=$(out_dir $K $TAG $SG)
  if qsub -N "$(grid_name $SG $M $S)" -l walltime=$WALLTIME -v K=$K,TAG=$TAG,SG=$SG,MODEL=$M,SEED=$S,OUT=$OUT,CONT=0 run155.pbs >/dev/null; then n=$((n+1)); else fail=1; break; fi
done
echo "Submitted $n v15.7 grid jobs ($skip done or queued)."
[ $fail -eq 1 ] && echo "qsub refused (queue limit?). Re-run this script later."
echo "When qstat is empty, from qcq_cnn_v157: python3 gate_v157.py grid \$PWD ../qcq_cnn_v156 ../qcq_cnn_v155"
