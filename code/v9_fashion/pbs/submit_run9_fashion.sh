#!/bin/bash
# Fashion-MNIST grid, arm (a) iso-parametric: 4 models x 5 seeds = 20 jobs.
#


#
# RUN ./submit_run9_smoke.sh <pair> FIRST and confirm it came back clean.
# This script refuses to submit until a clean v9 smoke result exists for the
# requested pair, because "smoke test first" has caught six silent failures
# so far and a grid submitted on a broken config wastes ~48 SU and, worse,
# returns plausible-looking numbers.
#
# v9 vs v8: NO code change at all -- version tag only, so that the 2,4 grid
# is attributable to its own release. The gap is measured on the restored
# best checkpoint (gap@best) inside the trainer, as introduced in v8.
# Results go to results_v9_fashion_c<ab>/.
set -e
CLASSES="${1:-2,4}"
CLASSPAIR="${CLASSES//,/_}"
CTAG="${CLASSES//,/}"

case "$CLASSES" in
    [0-9],[0-9]) ;;
    *) echo "FATAL: class pair must be 'a,b' with single digits (got '$CLASSES')"; exit 1 ;;
esac
[ "${CLASSES%,*}" = "${CLASSES#*,}" ] && { echo "FATAL: class pair must be two DIFFERENT classes"; exit 1; }

SMOKE="../results_v9_fashion_c${CTAG}/result_M2_fashion_mnist_seed42_alpha0.0.json"
if [ ! -f "$SMOKE" ]; then
    echo ""
    echo "REFUSING TO SUBMIT: no v9 smoke result found for pair $CLASSES."
    echo "  expected: $SMOKE"
    echo ""
    echo "  Run  ./submit_run9_smoke.sh $CLASSES  first, wait for it to finish,"
    echo "  and verify the integrity outputs before submitting the grid."
    echo "  If you have deliberately decided to skip the smoke test, pass"
    echo "  --force as the second argument."
    [ "$2" != "--force" ] && exit 1
    echo "  --force given; proceeding against the workflow rule."
fi

echo "Class pair: $CLASSES  (passed to PBS as CLASSPAIR=$CLASSPAIR)"
case "$CLASSES" in
    0,1) echo "  NOTE: this pair is saturated (M4 98.8%). Expect a null result." ;;
    0,6) echo "  NOTE: 0,6 arm (a) is already COMPLETE under v7 (M4 84.20%, best)."
         echo "        Re-running gives a v8+ gap@best column; accuracies should"
         echo "        reproduce. Only worth it as a deliberate 10-seed power run." ;;
    2,6) echo "  NOTE: 2,6 arm (a) is already COMPLETE under v8."
         echo "        M3 87.70% (best) > M4 87.00% > M2 86.74% > M1 85.70%."
         echo "        All three attribution contrasts null, all below MDE." ;;
    2,4) echo "  Pullover vs Coat — rank 3 (easiest) of the difficulty gradient."
         echo "  Experiment 3 of 3. M4 pilot was 88.40% (single seed)." ;;
esac

# The smoke test already produced M2/seed42. Re-submitting it would overwrite
# that file with a second, non-identical run (fixed seeds are not
# bit-reproducible on this stack, ~0.8 pp spread), which would silently
# replace the number the grid was authorised on. Skip it by default.
SKIP_M2_42=1

for MODEL in M1 M2 M3 M4; do
    for SEED in 42 43 44 45 46; do
        if [ "$SKIP_M2_42" = "1" ] && [ "$MODEL" = "M2" ] && [ "$SEED" = "42" ] \
           && [ -f "$SMOKE" ]; then
            echo "Skipping M2 / seed=42 — already produced by the smoke test."
            continue
        fi
        echo "Submitting $MODEL / fashion_mnist / classes=$CLASSES / seed=$SEED"
        qsub -v MODEL=$MODEL,DATASET=fashion_mnist,SEED=$SEED,CLASSPAIR=$CLASSPAIR run9_fashion_grid.pbs
    done
done

cat <<EOF

Submitted. Monitor:            qstat -u \$USER

BEFORE reading any accuracy, check the integrity outputs:
  Class pair took (all same):  grep -h 'CLASS PAIR' v9_fash.o* | sort -u
  Code version (all v9):       grep -h 'CODE VERSION' v9_fash.o* | sort -u
  Protocol (no REDUCED-SCALE): grep -c 'REDUCED-SCALE' v9_fash.o*
  Inversions (MUST be 0):      grep -cE 'INVERSION' v9_fash.o*
  gap@best present:            grep -c 'GAP@BEST' v9_fash.o*
  All jobs finished rc=0:      grep -h 'Job end' v9_fash.o* | sort -u

Collect:                       python scripts/collect_results.py results_v9_fashion_c${CTAG}
EOF
