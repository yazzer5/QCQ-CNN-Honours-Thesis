#!/bin/bash
# M1 INVERSION INVESTIGATION.
#
# Smoke batch 2026-07-28: M2=0.97, M3=0.96, M4=0.95, but M1 INVERTED
# (train_acc 0.96 while val_acc fell to 0.04; inverting val gives 0.96;
# reported test_acc=0.5000). Detector caught it. This is the 2nd observed
# inversion -- the 1st was M2 on run 174891473. Both models that invert have
# the nonlinear QNN back-end; M3/M4 (linear head) never do, on the same data
# and the same dropout.
#
# Two arms, both M1/mnist/seed42:
#   dropout=0.5  -> reproduce the inversion, and let the new [MODE CHECK]
#                   diagnostic fire. That prints TRAIN-SET accuracy in EVAL
#                   mode, which separates "mode effect" from "data effect":
#                     ~= 1 - train_acc  -> decision inverts between modes on
#                                          IDENTICAL data => dropout is the
#                                          cause (nonlinear QNN is not
#                                          expectation-preserving under it)
#                     ~= train_acc      -> mode is NOT the cause; investigate
#                                          the val split instead
#   dropout=0.0  -> if M1 trains cleanly here, dropout is implicated.
#
# Usage: ./submit_run5_m1_inversion.sh
set -e
for D in 0.5 0.0; do
    echo "Submitting M1 / mnist / seed=42 / dropout=$D"
    qsub -v MODEL=M1,DATASET=mnist,SEED=42,DROPOUT=$D run5_smoke.pbs
done
echo ""
echo "Read both with:"
echo "  grep -E 'DROPOUT=|MODE CHECK|INVERSION|epoch|FINAL' v5_smoke.o<jobid>"
