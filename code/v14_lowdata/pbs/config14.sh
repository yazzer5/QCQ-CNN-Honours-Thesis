#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# v14: LOW DATA (sample efficiency). M1-M5 at 50 / 100 / 250 train images per
# class, every head at MAX readout width 15 (all 15 Paulis, v13's w15).
# N=500: M2, M3 from v13 w15; M5 from v13.5 w15 (read, never re-run).
#        M1, M4 were never run at w15, so v14 runs them at N=500.
# PRE-REGISTERED design (2026-10-15). Do not edit after the first smoke
# result is read, except as a dated amendment in PREREGISTRATION_v14.md.
#
# NAMING RULE (D1-v13.5): nothing here may share a name with a run14.pbs input
# (K TAG N MODEL SEED OUT CONT). Grid lists are GRID_*.
# ═══════════════════════════════════════════════════════════════════════

V13_ROOT=/g/data/wa66/yh2481/qcq_cnn_v13                  # N=500 w15: M2, M3 (K=5 A/B, K=10)
V135_ROOT=/g/data/wa66/yh2481/qcq_cnn_v135                # N=500 w15: M5     (K=5 A/B, K=10)

subset_classes() {
  case "$1_$2" in
    5_A)    echo "1,3,5,8,9" ;;
    5_B)    echo "0,2,6,8,9" ;;
    10_all) echo "0,1,2,3,4,5,6,7,8,9" ;;
    *) echo "BAD_SUBSET"; return 1 ;;
  esac
}
subset_tags() { [ "$1" = "10" ] && echo "all" || echo "A B"; }

GRID_KS="5 10"                       # K=10 may be dropped ONLY before the smoke is submitted (resource decision)
GRID_NS="50 100 250 500"             # train images per class
GRID_MODELS="M1 M2 M3 M4 M5"
models_for() { [ "$1" = "500" ] && echo "M1 M4" || echo "$GRID_MODELS"; }   # N=500: only the two never run at w15
GRID_SEEDS="42 43 44 45 46 47 48 49 50 51"
SMOKE_SEED=42
W_OBS=15                             # MAX readout width (v13 w15) for every model

PER_CLASS_VAL=125
PER_CLASS_TEST=200
PATIENCE=20                          # epochs, as v12 (warmup 30 epochs is in train_gadi.py, unchanged)
# Max-epoch CAP scaled so the gradient-step budget is never smaller than v12's
# at N=500 (200 epochs): cap = 200 * 500 / N. N=500 -> 200 (reproductions exact).
# Early stopping (patience 20) still decides when a run ends; the cap only stops
# low-N runs from being truncated by a budget 10x tighter than v12's.
max_epochs_for() { echo $(( 200 * 500 / $1 )); }       # 50->2000, 100->1000, 250->400, 500->200
walltime_for()   { case "$1" in 50|100) echo "02:00:00" ;; *) echo "01:00:00" ;; esac; }
budget_for()     { case "$1" in 50|100) echo "1.75" ;;     *) echo "0.75" ;;     esac; }

results_dir() { echo "results_v14_k$1_$2_n$3"; }          # K TAG N
# Reproductions at N=500, w15 (K=5 A, seed 42) in the PATCHED repo, against the stored runs:
repro_dir()   { echo "results_v14_repro_k5_A_n500_$1"; }  # MODEL
repro_ref()   { case "$1" in
                  M3) echo "$V13_ROOT/results_v13_k5_A_w15/result_M3_mnist_seed42_alpha0.0.json" ;;
                  M5) echo "$V135_ROOT/results_v135_k5_A_w15/result_M5_mnist_seed42_alpha0.0.json" ;;
                esac; }
REPRO_MODELS="M3 M5"
