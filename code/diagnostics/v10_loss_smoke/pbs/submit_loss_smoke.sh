#!/bin/bash
# Two jobs. M2, seed 42, one per loss function. Nothing else differs.
#
# M2 is the right model to test with: it is the model the hypothesis is
# about, it has the most parameters, and it exercises both the trainable
# front-end and the 3-observable QNN readout. If the loss matters anywhere,
# it matters here.
set -e
cd "$(dirname "$0")"

echo "Submitting loss-comparison smoke: M2 seed 42, MNIST 10-class"
qsub -v MODEL=M2,SEED=42,LOSS=bce,CONT=0 run_loss_smoke.pbs
qsub -v MODEL=M2,SEED=42,LOSS=ce,CONT=0  run_loss_smoke.pbs
echo
echo "Two jobs submitted. When BOTH finish:"
echo
echo "  # integrity FIRST -- before any accuracy number"
echo "  grep -h 'CODE VERSION'       v10loss.o*"
echo "  grep -h 'PROTOCOL RESOLVED'  v10loss.o*   # loss= should differ, nothing else"
echo "  grep -h 'MULTICLASS'         v10loss.o*   # n_classes=10 n_obs=3"
echo "  grep -h 'LOSS'               v10loss.o*   # one bce, one ce"
echo "  grep -h 'DATA'               v10loss.o*   # 10 classes, test=2000"
echo "  grep -h 'VERIFIED'           v10loss.o*   # total=1646 (expected 1646)"
echo "  grep -h 'DETERMINISM'        v10loss.o*   # ENABLED"
echo "  grep -h 'PREDICTIONS'        v10loss.o*   # recomputed acc matches"
echo "  grep -h 'Job end'            v10loss.o*   # rc=0, both jobs"
echo
echo "  # then, and only then:"
echo "  cd .. && python scripts/compare_loss_smoke.py"
