#!/bin/bash
# ONE job: M2, seed 42, into the grid's own output directory.
#
# This is deliberately a repeat of the CE arm of the loss smoke, at
# byte-identical settings, so it does double duty:
#   1. the mandatory full-scale smoke before a grid, and
#   2. the DETERMINISM CHECK. It must reproduce test_acc = 0.7775 EXACTLY,
#      not approximately. Two runs of the same seed on the same hardware
#      with --deterministic should be bit-identical. If they differ at all,
#      something is injecting nondeterminism and the whole point of
#      --deterministic (tightening SD(diff) so the paired tests have power)
#      is void. Find it before the grid, not after.
set -e
cd "$(dirname "$0")"
qsub -v MODEL=M2,SEED=42,CONT=0 run10_mnist10_grid.pbs
echo
echo "Submitted. When it finishes:"
echo
echo "  # integrity FIRST, before any accuracy number"
echo "  grep -h 'CODE VERSION'       v10_m10.o*"
echo "  grep -h 'PROTOCOL RESOLVED'  v10_m10.o*   # loss=ce, n_classes=10, n_obs=3"
echo "  grep -h 'MULTICLASS'         v10_m10.o*"
echo "  grep -h 'LOSS'               v10_m10.o*   # ce"
echo "  grep -h 'DATA'               v10_m10.o*   # 10 classes, test=2000"
echo "  grep -h 'VERIFIED'           v10_m10.o*   # total=1646 (expected 1646)"
echo "  grep -h 'DETERMINISM'        v10_m10.o*   # ENABLED"
echo "  grep -h 'PREDICTIONS'        v10_m10.o*   # recomputed acc matches"
echo "  grep -h 'Job end'            v10_m10.o*   # rc=0"
echo
echo "  # then the determinism check -- must be EXACTLY 0.7775"
echo "  python -c \"import json;print(json.load(open('../results_v10_mnist10/result_M2_mnist_seed42_alpha0.0.json'))['test_acc'])\""
echo
echo "Only if that matches: ./submit_run10_grid.sh"
