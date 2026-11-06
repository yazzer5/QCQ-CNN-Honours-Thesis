#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# v16a: M2c (closed-form replica of the w15 QNN head) + z-dumps. PRE-REGISTERED
# (PREREGISTRATION_v16a.md). Change nothing below after the smoke is submitted except
# as a dated amendment logged in the pre-registration first.
# NAMING RULE: nothing here may share a name with a run16a.pbs input
# (ARM K TAG N MODEL SEED OUT CONT).
# ═══════════════════════════════════════════════════════════════════════
BASE=/g/data/wa66/yh2481
V13_ROOT=$BASE/qcq_cnn_v13          # stored M2, M3 at N=500, w15 (comparators + reproductions)
V14_ROOT=$BASE/qcq_cnn_v14          # stored M2, M3 at N=50, w15 (comparators + reproductions)

GRID_SEEDS="42 43 44 45 46 47 48 49 50 51"
SMOKE_SEED=42
# arm c: M2c on K=5 A, K=5 B, K=10, N in {50, 500}, seeds 42-51 (60 jobs)
C_KS="5 10"
C_NS="50 500"
# arm z: --save_z on M2 and M3, K=10, N in {50, 500}, seeds 42-44 (12 jobs, all reproductions)
Z_MODELS="M2 M3"
Z_K=10
Z_NS="50 500"
Z_SEEDS="42 43 44"
# reproduction gate: M2, K=5 A, N=500, seed 42 == v13 exactly

W_OBS=15
PER_CLASS_VAL=125
PER_CLASS_TEST=200
PATIENCE=20
WALLTIME=01:00:00                  # every K and N (user decision 2026-10-20; revisit only on smoke evidence)
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
    c) echo "results_v16a_c_k$2_$3_n$4" ;;
    z) echo "results_v16a_z_k$2_$3_n$4" ;;
    r) echo "results_v16a_repro_k$2_$3_n$4" ;;
  esac
}
stored_dir() {                      # K TAG N  (the stored M2/M3 runs at w15)
  if [ "$3" = "500" ]; then echo "$V13_ROOT/results_v13_k$1_$2_w15"; else echo "$V14_ROOT/results_v14_k$1_$2_n$3"; fi
}
