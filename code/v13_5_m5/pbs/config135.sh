#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# v13.5: M5 (= M3 with the quantum front-end frozen) at EVERY v12 and v13
# cell. PRE-REGISTERED design (2026-10-12). Do not edit after the first smoke
# result is read, except as a dated amendment in PREREGISTRATION_v13_5.md.
# ═══════════════════════════════════════════════════════════════════════

# Existing M1-M4 results the M5 runs are compared against (read, never re-run).
V12_ROOT=/g/data/wa66/yh2481/qcq_cnn_v12                  # K=3,4,5 width 3
V13_ROOT=/g/data/wa66/yh2481/qcq_cnn_v13                  # K=5,10 width 6/15; K=10 w3 seeds 47-51
V10_RES=/g/data/wa66/yh2481/v10_grid/results_v10_mnist10   # K=10 w3 seeds 42-46

# v12's digit subsets (drawn once with seed 20260928), unchanged. K=10 = all digits.
subset_classes() {
  case "$1_$2" in
    3_A)    echo "1,4,9" ;;
    3_B)    echo "2,4,9" ;;
    4_A)    echo "0,2,3,8" ;;
    4_B)    echo "0,1,2,9" ;;
    5_A)    echo "1,3,5,8,9" ;;
    5_B)    echo "0,2,6,8,9" ;;
    10_all) echo "0,1,2,3,4,5,6,7,8,9" ;;
    *) echo "BAD_SUBSET"; return 1 ;;
  esac
}
subset_tags() { [ "$1" = "10" ] && echo "all" || echo "A B"; }

# Head widths per K: v12 ran every K at width 3; v13 added 6 and 15 at K=5 and 10.
widths_for() { case "$1" in 5|10) echo "3 6 15" ;; *) echo "3" ;; esac; }
KS="3 4 5 10"

GRID_MODEL=M5
SEEDS="42 43 44 45 46 47 48 49 50 51"
SMOKE_SEED=42

PER_CLASS_TRAIN=500
PER_CLASS_VAL=125
PER_CLASS_TEST=200

walltime_for() { echo "01:00:00"; }
budget_for()   { echo "0.75"; }

results_dir() { echo "results_v135_k$1_$2_w$3"; }          # K TAG W
# Reproduction: M3, K=5, subset A, w3, seed 42 in the PATCHED repo must equal v12's run.
REPRO_K=5; REPRO_TAG=A; REPRO_W=3; REPRO_MODEL=M3
repro_dir()  { echo "results_v135_repro_k5_A_w3_M3"; }
ref_result() { echo "$V12_ROOT/results_v12_k5_A/result_M3_mnist_seed42_alpha0.0.json"; }
