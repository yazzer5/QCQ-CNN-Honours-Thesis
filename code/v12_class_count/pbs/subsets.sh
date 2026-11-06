#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# v12 class-count sweep: PRE-REGISTERED digit subsets.
#
# Drawn ONCE on 2026-09-28, before any v12 job ran, with
#   numpy.random.default_rng(20260928)
#   for k in (3,4,5): draw sorted k-of-10 without replacement until two
#   distinct subsets exist.
# Do NOT redraw, reorder, or swap a subset after seeing results. That would
# be choosing the answer. If a subset turns out to be degenerate, log it in
# MNIST_LOG.md as an amendment and keep it in the tables.
#
# Two subsets per k so "harder because more classes" can be separated from
# "harder because these particular digits look alike" (the FMNIST
# difficulty-gradient retraction was exactly that confound).
# ═══════════════════════════════════════════════════════════════════════

# usage: subset_classes K TAG   -> prints the comma list
subset_classes() {
  case "$1_$2" in
    3_A) echo "1,4,9"      ;;
    3_B) echo "2,4,9"      ;;
    4_A) echo "0,2,3,8"    ;;
    4_B) echo "0,1,2,9"    ;;
    5_A) echo "1,3,5,8,9"  ;;
    5_B) echo "0,2,6,8,9"  ;;
    *) echo "BAD_SUBSET"; return 1 ;;
  esac
}

SUBSET_TAGS="A B"
MODELS="M1 M2 M3 M4"
SEEDS="42 43 44 45 46 47 48 49 50 51"

# Per-class split is FIXED across k (identical to v10: 5000/1250/2000 over
# 10 classes = 500/125/200 per class). So k=10 in this sweep IS the v10
# protocol, and v10 results slot in as the k=10 point without re-running.
PER_CLASS_TRAIN=500
PER_CLASS_VAL=125
PER_CLASS_TEST=200

results_dir() { echo "results_v12_k$1_$2"; }   # relative to repo root
