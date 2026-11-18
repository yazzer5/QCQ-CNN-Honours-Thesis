#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# v15.5: Gaussian PIXEL noise on the input images, M1-M5, w15, N=500/class.
# PRE-REGISTERED design (PREREGISTRATION_v15_5.md). Change nothing below after the
# smoke is submitted except as a dated amendment.
# ═══════════════════════════════════════════════════════════════════════
BASE=${BASE:-/g/data/wa66/yh2481}
V14_ROOT=$BASE/qcq_cnn_v14        # M1, M4 at N=500 w15 (sigma=0 reproductions)
V13_ROOT=$BASE/qcq_cnn_v13        # M2, M3 at w15
V135_ROOT=$BASE/qcq_cnn_v135      # M5 at w15

GRID_KS="5 10"                       # K=10 may be dropped ONLY before the smoke (resources), as an amendment
GRID_SIGMAS="0 0.1 0.2 0.3 0.5"      # training (and matched test) pixel-noise sigma
EVAL_SIGMAS="0 0.1 0.2 0.3 0.5 0.7 1.0"   # cross-evaluation sigma for every trained model
EVAL_SP="0.05 0.1 0.2"               # salt-and-pepper cross-evaluation (exploratory)
MODELS="M1 M2 M3 M4 M5"              # every model at every sigma (the noise is on the input)
GRID_SEEDS="42 43 44 45 46 47 48 49 50 51"
SMOKE_SEED=42

N_PER_CLASS=500
PER_CLASS_VAL=125
PER_CLASS_TEST=200
W_OBS=15
MAX_EPOCHS=200
PATIENCE=20
WALLTIME=01:00:00
BUDGET=0.85                          # --time_budget_hours (trainer stops cleanly before walltime)

FILES_PER_JOB=3                      # log, result JSON, cross-eval sidecar (checkpoint dir removed after MATCH)
INODE_MARGIN=1500                    # submit refuses unless free gdata inodes >= jobs*FILES_PER_JOB + margin

subset_classes() {
  case "$1_$2" in
    5_A)    echo "1,3,5,8,9" ;;
    5_B)    echo "0,2,6,8,9" ;;
    10_all) echo "0,1,2,3,4,5,6,7,8,9" ;;
    *) return 1 ;;
  esac
}
tags_for() { if [ "$1" = "5" ]; then echo "A B"; else echo "all"; fi; }
out_dir() { echo "results_v156_k$1_$2_s$3"; }
