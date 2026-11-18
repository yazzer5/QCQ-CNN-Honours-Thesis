#!/bin/bash
# v15.6 GRID: seeds 43-51 x {M3,M4,M5} x {0,0.3} x {K5 A, K5 B, K10} = 162 jobs. Refuses without ../.smoke_ok.
# Resumable: re-run after a qsub queue-limit stop; skips finished and queued jobs.
set -e
cd "$(dirname "$0")"
source ./config155.sh; source ./names155.sh
MODELS="M3 M4 M5"; GRID_SIGMAS="0 0.3"; FILES_PER_JOB=4
grid_name() { echo "v156g_k${1}${2}_s${3}_${4}_r${5}"; }
[ -f ../.smoke_ok ] || { echo "REFUSED: smoke gate not passed (python3 gate_v156.py smoke ...)"; exit 1; }
Q=$(queued_names)
todo=(); skip=0
for K in $GRID_KS; do for TAG in $(tags_for $K); do for SG in $GRID_SIGMAS; do for M in $MODELS; do
  for S in $GRID_SEEDS; do
    [ "$S" = "$SMOKE_SEED" ] && continue
    NAME=$(grid_name $K $TAG $SG $M $S); OUT=$(out_dir $K $TAG $SG)
    if ls ../$OUT/geom_${M}_seed${S}.npz >/dev/null 2>&1 || grep -qx "$NAME" <<<"$Q"; then skip=$((skip+1)); continue; fi
    todo+=("$K $TAG $SG $M $S")
  done
done; done; done; done
inode_guard ${#todo[@]} || exit 1
n=0; fail=0
for t in "${todo[@]}"; do
  set -- $t; K=$1; TAG=$2; SG=$3; M=$4; S=$5
  NAME=$(grid_name $K $TAG $SG $M $S); OUT=$(out_dir $K $TAG $SG)
  if qsub -N "$NAME" -l walltime=$WALLTIME -v K=$K,TAG=$TAG,SG=$SG,MODEL=$M,SEED=$S,OUT=$OUT,CONT=0 run155.pbs >/dev/null; then
    n=$((n+1))
  else
    fail=1; break
  fi
done
echo "Submitted $n v15.6 grid jobs ($skip already done or queued)."
[ $fail -eq 1 ] && echo "qsub refused (queue limit?). Wait, then run ./submit_v156_grid.sh again."
echo "When qstat is empty: cd .. && python3 gate_v156.py grid \$PWD ../qcq_cnn_v155"
