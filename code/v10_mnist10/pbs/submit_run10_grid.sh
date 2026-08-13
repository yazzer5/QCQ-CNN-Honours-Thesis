#!/bin/bash
# Full v10 grid: 4 models x 5 seeds = 20 runs, minus M2/seed42 which the
# smoke already produced. Refuses to run without that smoke result.
set -e
cd "$(dirname "$0")"
SMOKE="../results_v10_mnist10/result_M2_mnist_seed42_alpha0.0.json"
if [ ! -f "$SMOKE" ]; then
    echo "REFUSING: no smoke result at $SMOKE"
    echo "Run ./submit_run10_smoke.sh first and check its integrity output."
    exit 1
fi

# The smoke must have used CE. If it did not, the grid would silently mix
# two protocols in one results directory -- 19 CE runs alongside one BCE
# run, indistinguishable once collected into a table.
LOSS=$(python -c "import json;print(json.load(open('$SMOKE')).get('loss'))")
if [ "$LOSS" != "ce" ]; then
    echo "REFUSING: smoke result is stamped loss=$LOSS, expected ce."
    echo "Delete it and re-run the smoke, or the grid mixes two protocols."
    exit 1
fi
echo "Smoke result found, loss=ce. Submitting the remaining 19 jobs."

N=0
for MODEL in M1 M2 M3 M4; do
  for SEED in 42 43 44 45 46; do
    if [ "$MODEL" = "M2" ] && [ "$SEED" = "42" ]; then
        echo "  skip M2/seed42 (smoke result -- do not overwrite)"; continue
    fi
    qsub -v MODEL=$MODEL,SEED=$SEED,CONT=0 run10_mnist10_grid.pbs
    N=$((N+1))
  done
done
echo "Submitted $N jobs. Expect 20 result JSONs when complete."
echo "At ~20 min per job this is well inside the 4 h walltime; no job"
echo "should ever reach the 3.5 h budget or trigger a continuation."
