#!/bin/bash
# reps=1 vs reps=2 ablation on the QNN back-end.
#
#   reps=2 -> 6 variational params -> matches Long's PAPER (depth d=5)
#   reps=1 -> 4 variational params -> matches Long's published CODE
#
# Only M1 and M2 carry a QNN back-end, so only they are run. M3/M4 are
# unaffected by this setting and their existing results remain valid.
#
#   ./submit_reps_ablation.sh 0,6      (default: hardest Fashion pair)
#   ./submit_reps_ablation.sh 3,5 mnist
set -e
CLASSES="${1:-0,6}"
DATASET="${2:-fashion_mnist}"
CLASSPAIR="${CLASSES//,/_}"
CTAG="${CLASSES//,/}"

case "$CLASSES" in
    [0-9],[0-9]) ;;
    *) echo "FATAL: class pair must be 'a,b' (got '$CLASSES')"; exit 1 ;;
esac

echo "Ablation: $DATASET classes=$CLASSES  (M1,M2 x reps{1,2} x 5 seeds = 20 jobs)"
for MODEL in M1 M2; do
    for REPS in 1 2; do
        for SEED in 42 43 44 45 46; do
            echo "  $MODEL reps=$REPS seed=$SEED"
            qsub -v MODEL=$MODEL,DATASET=$DATASET,SEED=$SEED,CLASSPAIR=$CLASSPAIR,REPS=$REPS \
                 run_reps_ablation.pbs
        done
    done
done
echo ""
echo "20 jobs submitted. Monitor:  qstat -u \$USER"
echo "Sanity (MUST be 0):          grep -cE 'INVERSION' reps_abl.o*"
echo "Check both arms present:     grep -h 'REPS=' reps_abl.o* | sort | uniq -c"
echo "Analyse:                     python scripts/analyse_reps.py results_reps_${DATASET}_c${CTAG}"
