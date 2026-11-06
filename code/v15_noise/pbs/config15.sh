#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# v15: IBM-calibrated FRONT-END noise, M1-M5, w15, N=500/class.
# PRE-REGISTERED design (PREREGISTRATION_v15.md). Change nothing below after the
# smoke is submitted except as a dated amendment.
# ═══════════════════════════════════════════════════════════════════════
BASE=/g/data/wa66/yh2481
V14_ROOT=$BASE/qcq_cnn_v14        # M1, M4 at N=500 w15 (alpha=0 reproductions)
V13_ROOT=$BASE/qcq_cnn_v13        # M2, M3 at w15
V135_ROOT=$BASE/qcq_cnn_v135      # M5 at w15

GRID_KS="5 10"                     # K=10 may be dropped ONLY before the smoke (SU), as an amendment
GRID_ALPHAS="0 0.5 1 2 4"          # training (and matched test) noise levels
EVAL_ALPHAS="0 0.5 1 2 4 8 16"     # cross-evaluation levels for every trained model
NOISE_MODELS="M1 M2 M3 M5"         # models with a quantum front-end: run at every alpha
CTRL_MODELS="M4"                   # no quantum front-end: alpha=0 only (identical at every alpha)
GRID_SEEDS="42 43 44 45 46 47 48 49 50 51"
SMOKE_SEED=42

N_PER_CLASS=500
PER_CLASS_VAL=125
PER_CLASS_TEST=200
W_OBS=15
MAX_EPOCHS=200
PATIENCE=20
WALLTIME=01:00:00
BUDGET=0.85                        # --time_budget_hours (trainer stops cleanly before walltime)

CALIB_REL=calib/ibm_brisbane_calib.json
CALIB_SHA=da9374fa89e6a0656e9e18f1295aa84aab1be8d754ad8b8143199a2c700cc3fc

subset_classes() {
  case "$1_$2" in
    5_A)    echo "1,3,5,8,9" ;;
    5_B)    echo "0,2,6,8,9" ;;
    10_all) echo "0,1,2,3,4,5,6,7,8,9" ;;
    *) return 1 ;;
  esac
}
tags_for() { if [ "$1" = "5" ]; then echo "A B"; else echo "all"; fi; }
models_for_alpha() { if [ "$1" = "0" ]; then echo "$NOISE_MODELS $CTRL_MODELS"; else echo "$NOISE_MODELS"; fi; }
out_dir() { echo "results_v15_k$1_$2_a$3"; }
