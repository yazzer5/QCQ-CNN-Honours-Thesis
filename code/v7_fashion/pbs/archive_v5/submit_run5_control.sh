#!/bin/bash
# STEP 2 OF 3 — batch-size control: M2, MNIST, seed 42, bs=1 vs bs=32.
# This is the evidence that the v5 batch-size change is benign.
# Usage: ./submit_run5_control.sh
set -e
for BS in 1 32; do
    echo "Submitting control arm BS=$BS ..."
    qsub -v BS=$BS,MODEL=M2,DATASET=mnist,SEED=42 run5_batchsize_control.pbs
done
echo ""
echo "Compare when done:"
echo "  cat ../results_v5_control/bs1/*.json  | python -c 'import json,sys;d=json.load(sys.stdin);print(d[\"test_acc\"])'"
echo "  cat ../results_v5_control/bs32/*.json | python -c 'import json,sys;d=json.load(sys.stdin);print(d[\"test_acc\"])'"
