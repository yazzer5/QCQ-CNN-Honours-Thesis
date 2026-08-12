#!/bin/bash
# DECISIVE A/B TEST for the M1/M2 global decision inversion.
#
# Hypothesis: Dropout2d(0.5) sits immediately before fc2 -> QNN. Dropout is
# expectation-preserving, which is EXACT for M3/M4's linear head but not for
# M1/M2's nonlinear quantum back-end (E[QNN(x)] != QNN(E[x])). The resulting
# shift in the QNN's single scalar output flips fc3 = Linear(1,2)'s argmax for
# every sample at once -- producing the observed clean inversion
# (train_acc 0.95 / val_acc 0.032; inverting val gives 0.968).
#
# Mechanism verified numerically against the real circuit in
# scripts/verify_dropout_qnn_shift.py. This run is the confirmation on
# hardware with the real EstimatorQNN.
#
# Expected outcome if the hypothesis holds:
#   drop0.5 arm -> val_acc collapses while train_acc climbs (as before)
#   drop0.0 arm -> val_acc TRACKS train_acc
#
# Usage: ./submit_run5_dropout_ab.sh
set -e
for D in 0.5 0.0; do
    echo "Submitting M2 / mnist / seed=42 / dropout=$D"
    qsub -v MODEL=M2,DATASET=mnist,SEED=42,DROPOUT=$D run5_smoke.pbs
done
echo ""
echo "Compare when done:"
echo "  grep -E 'DROPOUT=|TRAIN/EVAL SHIFT|MARGIN|flipping|epoch|FINAL' v5_smoke.o<jobid>"
