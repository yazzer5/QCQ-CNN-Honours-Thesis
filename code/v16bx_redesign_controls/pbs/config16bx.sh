#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# v16b-ext: controls M4-trim and BFR-ZZ at K=10, N=50 (PREREGISTRATION_v16b.md amendment A3).
# Change nothing below after the smoke is submitted except as a dated amendment logged first.
# NAMING RULE: nothing here may share a name with a run16bx.pbs input (ARM K TAG N MODEL SEED OUT CONT).
# ═══════════════════════════════════════════════════════════════════════
BASE=/g/data/wa66/yh2481
V14_ROOT=$BASE/qcq_cnn_v14          # stored M4 and M2-w15 at K=10, N=50
V16B_ROOT=$BASE/qcq_cnn_v16b        # stored BFR at K=10, N=50 (comparator + reproduction reference)

GRID_SEEDS="42 43 44 45 46 47 48 49 50 51"
SMOKE_SEED=42
X_MODELS="M4trim BFRZZ"            # 2 x 10 = 20 jobs; reproduction: BFR K=10 N=50 seed 42 == v16b
XK=10
XTAG=all
XN=50

W_OBS=15
PER_CLASS_VAL=125
PER_CLASS_TEST=200
PATIENCE=20
WALLTIME=01:00:00
BUDGET=0.85

max_epochs_for() { echo $(( 200 * 500 / $1 )); }    # v14 N-scaled cap: 50 -> 2000
subset_classes() { case "$1_$2" in 10_all) echo "0,1,2,3,4,5,6,7,8,9" ;; *) return 1 ;; esac; }
out_dir() {                         # ARM MODEL
  case "$1" in
    x) echo "results_v16bx_$(echo $2 | tr 'A-Z' 'a-z')_k${XK}_${XTAG}_n${XN}" ;;
    r) echo "results_v16bx_repro_bfr_k${XK}_${XTAG}_n${XN}" ;;
  esac
}
