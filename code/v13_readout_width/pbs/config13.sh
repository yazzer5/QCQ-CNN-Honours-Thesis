#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# v13 readout-width experiment: PRE-REGISTERED design (2026-10-04).
# Do not edit after the first smoke result is read, except as a dated
# amendment logged in POST_CORE_RESULTS.md.
# ═══════════════════════════════════════════════════════════════════════

# Where the width-3 results already live (reused, not re-run).
V12_ROOT=/g/data/wa66/yh2481/qcq_cnn_v12                      # K=5 w=3 (v12)
V10_RES=/g/data/wa66/yh2481/v10_grid/results_v10_mnist10       # K=10 w=3 seeds 42-46 (v10)

# K=5 subsets are v12's, unchanged. K=10 is all digits.
subset_classes() {
  case "$1_$2" in
    5_A)    echo "1,3,5,8,9" ;;
    5_B)    echo "0,2,6,8,9" ;;
    10_all) echo "0,1,2,3,4,5,6,7,8,9" ;;
    *) echo "BAD_SUBSET"; return 1 ;;
  esac
}
subset_tags() { [ "$1" = "10" ] && echo "all" || echo "A B"; }

WIDTHS_NEW="6 15"            # 3 = Z basis (exists), 6 = Z+X, 15 = all Paulis
MODELS_NEW="M2 M3"           # quantum head vs width-matched classical head
SEEDS="42 43 44 45 46 47 48 49 50 51"

# K=10 width-3 top-up: v10 has seeds 42-46; these complete it to 10 seeds.
# FALLBACK (pre-registered): if the K=10 reproduction smoke does NOT match
# v10 exactly, v10 cannot be pooled. Then set TOPUP_SEEDS to all ten seeds
# and log amendment A1-v13. Never mix the two.
TOPUP_MODELS="M1 M2 M3 M4"
TOPUP_SEEDS="47 48 49 50 51"

PER_CLASS_TRAIN=500
PER_CLASS_VAL=125
PER_CLASS_TEST=200

walltime_for() { [ "$1" = "10" ] && echo "04:00:00" || echo "01:00:00"; }
budget_for()   { [ "$1" = "10" ] && echo "3.5"      || echo "0.75"; }

results_dir() { echo "results_v13_k$1_$2_w$3"; }          # K TAG W
repro_dir()   { echo "results_v13_repro_k$1_$2_w3"; }     # K TAG
ref_result()  {                                            # K TAG -> width-3 reference JSON for M2 seed 42
  if [ "$1" = "10" ]; then echo "$V10_RES/result_M2_mnist_seed42_alpha0.0.json"
  else echo "$V12_ROOT/results_v12_k$1_$2/result_M2_mnist_seed42_alpha0.0.json"; fi
}
