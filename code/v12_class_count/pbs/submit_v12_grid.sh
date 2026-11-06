#!/bin/bash
# STEP 2 for a class count: full grid = 2 subsets x 4 models x 10 seeds
# = 80 runs, minus the 2 smoke runs already done = 78 jobs.
# Refuses to run unless the smoke results exist and are stamped loss=ce.
set -e
cd "$(dirname "$0")"
source ./subsets.sh
K="${1:?usage: ./submit_v12_grid.sh K}"

for TAG in $SUBSET_TAGS; do
  SMOKE="../$(results_dir $K $TAG)/result_M2_mnist_seed42_alpha0.0.json"
  if [ ! -f "$SMOKE" ]; then
    echo "REFUSING: no smoke result at $SMOKE"
    echo "Run ./submit_v12_smoke.sh $K and ./check_v12.sh $K smoke first."
    exit 1
  fi
  LOSS=$(python3 -c "import json;print(json.load(open('$SMOKE')).get('loss'))")
  if [ "$LOSS" != "ce" ]; then
    echo "REFUSING: $SMOKE is stamped loss=$LOSS, expected ce."; exit 1
  fi
done
echo "Smoke results found for K=$K (both subsets, loss=ce). Submitting."

N=0
for TAG in $SUBSET_TAGS; do
  for MODEL in $MODELS; do
    for SEED in $SEEDS; do
      if [ "$MODEL" = "M2" ] && [ "$SEED" = "42" ]; then continue; fi
      qsub -N "v12_k${K}${TAG}" -v K=$K,TAG=$TAG,MODEL=$MODEL,SEED=$SEED,CONT=0 run12_classes.pbs >/dev/null
      N=$((N+1))
    done
  done
done
echo "Submitted $N jobs for K=$K. Expect 80 result JSONs (40 per subset) when done."
echo "Check with:  ./check_v12.sh $K grid"
