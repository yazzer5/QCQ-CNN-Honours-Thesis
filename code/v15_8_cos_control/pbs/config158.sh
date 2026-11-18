#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# v15.8: M6 (fixed cos(pi x) front-end) vs the stored v15.5 M3/M4/M5. PREREGISTRATION_v15_8.md.
# Change nothing below after the smoke is submitted except as a dated amendment.
# ═══════════════════════════════════════════════════════════════════════
BASE=${BASE:-/g/data/wa66/yh2481}
V155_ROOT=${V155_ROOT:-$BASE/qcq_cnn_v155}   # stored M3/M4/M5 runs (reused, and the reproduction reference)

GRID_KS="5 10"
GRID_SIGMAS="0 0.3"                      # sigma_train: 0 = deployed arm, 0.3 = matched arm
EVAL_SIGMAS="0 0.1 0.2 0.3 0.5 0.7 1.0"  # identical to v15.5 (same fixed noise fields)
EVAL_SP="0.05 0.1 0.2"
NEW_MODELS="M6"
REUSED_MODELS="M3 M4 M5"                 # from v15.5, same seeds
REPRO_K=5; REPRO_TAG=A; REPRO_SIGMA=0.3  # smoke re-runs M3/M4/M5 seed 42 here; must equal v15.5 bit-exactly
GRID_SEEDS="${GRID_SEEDS:-42 43 44 45 46}"
SMOKE_SEED=42

N_PER_CLASS=500
PER_CLASS_VAL=125
PER_CLASS_TEST=200
W_OBS=15
MAX_EPOCHS=200
PATIENCE=20
WALLTIME=01:00:00
BUDGET=0.85

FILES_PER_JOB=3
INODE_MARGIN=1500

subset_classes() {
  case "$1_$2" in
    5_A)    echo "1,3,5,8,9" ;;
    5_B)    echo "0,2,6,8,9" ;;
    10_all) echo "0,1,2,3,4,5,6,7,8,9" ;;
    *) return 1 ;;
  esac
}
tags_for() { if [ "$1" = "5" ]; then echo "A B"; else echo "all"; fi; }
out_dir() { echo "results_v158_k$1_$2_s$3"; }
v155_dir() { echo "$V155_ROOT/results_v155_k$1_$2_s$3"; }
