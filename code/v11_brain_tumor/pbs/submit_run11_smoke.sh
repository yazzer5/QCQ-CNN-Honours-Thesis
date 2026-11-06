#!/bin/bash
# ONE job: M2, seed 42, into the grid's own output directory.
# The mandatory full-scale smoke before a grid. Same settings as the grid,
# so the result is kept and seed 42 is skipped when the grid is submitted.
set -e
cd "$(dirname "$0")"
qsub -v MODEL=M2,SEED=42,CONT=0 run11_mri_grid.pbs
echo
echo "Submitted. When it finishes:"
echo
echo "  # integrity FIRST, before any accuracy number"
echo "  grep -h 'CODE VERSION'       v11_mri.o*   # v11"
echo "  grep -h 'PREFLIGHT v11'      v11_mri.o*   # PASS"
echo "  grep -h 'PROTOCOL RESOLVED'  v11_mri.o*   # 2240/560/772"
echo "  grep -h 'DATA'               v11_mri.o*   # Option A, balanced, AS-BUILT"
echo "  grep -h 'ALREADY STANDARDISED' v11_mri.o* # normalisation not re-applied"
echo "  grep -h 'VERIFIED'           v11_mri.o*   # total=1610 (expected 1610)"
echo "  grep -h 'DETERMINISM'        v11_mri.o*   # ENABLED"
echo "  grep -h 'REDUCED-SCALE'      v11_mri.o*   # must print NOTHING"
echo "  grep -h 'Job end'            v11_mri.o*   # rc=0"
echo
echo "  # then the accuracy -- sanity band, not a result"
echo "  python -c \"import json;print(json.load(open('../results_v11_mri/result_M2_mri_seed42_alpha0.0.json'))['test_acc'])\""
echo
echo "  Expect roughly 0.78-0.92. Logistic regression on the same 28x28"
echo "  input gets 0.7824 and majority class is 0.50. Near 0.50 means the"
echo "  loader is broken -- check [DATA] before anything else. Above ~0.95"
echo "  means check for leakage."
echo
echo "Only if all of that is clean: ./submit_run11_grid.sh"
