#!/bin/bash
# Fashion-MNIST grid, arm (a) iso-parametric: 4 models x 5 seeds = 20 jobs.
#
#   ./submit_run8_fashion.sh 2,6     <- experiment 2, rank-2 difficulty
#   ./submit_run8_fashion.sh 2,4     <- experiment 3, rank-3 difficulty
#
# RUN ./submit_run8_smoke.sh <pair> FIRST and confirm it came back clean.
# This script refuses to submit until a clean v8 smoke result exists for the
# requested pair, because "smoke test first" has caught six silent failures
# so far and a grid submitted on a broken config wastes ~48 SU and, worse,
# returns plausible-looking numbers.
#
# v8 vs v7: protocol and numerics identical; the generalisation gap is now
# measured on the restored best checkpoint (gap@best) inside the trainer
# rather than reconstructed afterwards from the final epoch. Results go to
# results_v8_fashion_c<ab>/.
set -e
CLASSES="${1:-2,6}"
CLASSPAIR="${CLASSES//,/_}"
CTAG="${CLASSES//,/}"

case "$CLASSES" in
    [0-9],[0-9]) ;;
    *) echo "FATAL: class pair must be 'a,b' with single digits (got '$CLASSES')"; exit 1 ;;
esac
[ "${CLASSES%,*}" = "${CLASSES#*,}" ] && { echo "FATAL: class pair must be two DIFFERENT classes"; exit 1; }

SMOKE="../results_v8_fashion_c${CTAG}/result_M2_fashion_mnist_seed42_alpha0.0.json"
if [ ! -f "$SMOKE" ]; then
    echo ""
    echo "REFUSING TO SUBMIT: no v8 smoke result found for pair $CLASSES."
    echo "  expected: $SMOKE"
    echo ""
    echo "  Run  ./submit_run8_smoke.sh $CLASSES  first, wait for it to finish,"
    echo "  and verify the integrity outputs before submitting the grid."
    echo "  If you have deliberately decided to skip the smoke test, pass"
    echo "  --force as the second argument."
    [ "$2" != "--force" ] && exit 1
    echo "  --force given; proceeding against the workflow rule."
fi

echo "Class pair: $CLASSES  (passed to PBS as CLASSPAIR=$CLASSPAIR)"
case "$CLASSES" in
    0,1) echo "  NOTE: this pair is saturated (M4 98.8%). Expect a null result." ;;
    0,6) echo "  NOTE: 0,6 arm (a) is already complete under v7. Re-running it here"
         echo "        gives a v8 gap@best column but should reproduce the v7 accuracies." ;;
    2,6) echo "  Pullover vs Shirt — rank 2 of the difficulty gradient (M4 pilot 87.10%)." ;;
    2,4) echo "  Pullover vs Coat  — rank 3 of the difficulty gradient (M4 pilot 88.40%)." ;;
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
        qsub -v MODEL=$MODEL,DATASET=fashion_mnist,SEED=$SEED,CLASSPAIR=$CLASSPAIR run8_fashion_grid.pbs
    done
done

cat <<EOF

Submitted. Monitor:            qstat -u \$USER

BEFORE reading any accuracy, check the integrity outputs:
  Class pair took (all same):  grep -h 'CLASS PAIR' v8_fash.o* | sort -u
  Code version (all v8):       grep -h 'CODE VERSION' v8_fash.o* | sort -u
  Protocol (no REDUCED-SCALE): grep -c 'REDUCED-SCALE' v8_fash.o*
  Inversions (MUST be 0):      grep -cE 'INVERSION' v8_fash.o*
  gap@best present:            grep -c 'GAP@BEST' v8_fash.o*
  All jobs finished rc=0:      grep -h 'Job end' v8_fash.o* | sort -u

Collect:                       python scripts/collect_results.py results_v8_fashion_c${CTAG}
EOF
