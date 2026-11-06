#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# v16b: Design 1 (BFR head) vs M4 and M2-w15 (both REUSED from stored runs). PRE-REGISTERED
# (PREREGISTRATION_v16b.md). Change nothing below after the smoke is submitted except as a dated
# amendment logged in the pre-registration first.
# NAMING RULE: nothing here may share a name with a run16b.pbs input (ARM K TAG N MODEL SEED OUT CONT).
# ═══════════════════════════════════════════════════════════════════════
BASE=/g/data/wa66/yh2481
V13_ROOT=$BASE/qcq_cnn_v13          # stored M2 (w15) at N=500
V14_ROOT=$BASE/qcq_cnn_v14          # stored M4 at N=50 and N=500; stored M2 at N=50

GRID_SEEDS="42 43 44 45 46 47 48 49 50 51"
SMOKE_SEED=42
B_KS="5 10"                        # BFR on K=5 subsets A, B and K=10, N in {50, 500}: 60 jobs
B_NS="50 500"
# reproduction gate: M4, K=5 A, N=500, seed 42 == v14 exactly

W_OBS=15
PER_CLASS_VAL=125
PER_CLASS_TEST=200
PATIENCE=20
WALLTIME=01:00:00                  # every K and N (K=10: user decision 2026-10-21)
BUDGET=0.85                        # --time_budget_hours: the trainer stops cleanly before walltime

max_epochs_for() { echo $(( 200 * 500 / $1 )); }    # v14 N-scaled cap: 50 -> 2000, 500 -> 200
subset_classes() {
  case "$1_$2" in
    5_A)    echo "1,3,5,8,9" ;;
    5_B)    echo "0,2,6,8,9" ;;
    10_all) echo "0,1,2,3,4,5,6,7,8,9" ;;
    *) return 1 ;;
  esac
}
tags_for() { if [ "$1" = "5" ]; then echo "A B"; else echo "all"; fi; }
out_dir() {                         # ARM K TAG N
  case "$1" in
    b) echo "results_v16b_bfr_k$2_$3_n$4" ;;
    r) echo "results_v16b_repro_k$2_$3_n$4" ;;
  esac
}
stored_m4_dir() { echo "$V14_ROOT/results_v14_k$1_$2_n$3"; }                       # K TAG N
stored_m2_dir() {                                                                  # K TAG N
  if [ "$3" = "500" ]; then echo "$V13_ROOT/results_v13_k$1_$2_w15"; else echo "$V14_ROOT/results_v14_k$1_$2_n$3"; fi
}
