#!/bin/bash
# The 20-job grid: 4 models x seeds 42-46.
# M2/seed42 is SKIPPED -- the smoke run already produced it at identical
# settings, and resubmitting would overwrite the determinism reference.
set -e
cd "$(dirname "$0")"

if [ ! -f ../results_v11_mri/result_M2_mri_seed42_alpha0.0.json ]; then
    echo "FATAL: no smoke result found."
    echo "Run ./submit_run11_smoke.sh first and check its integrity output."
    exit 1
fi

N=0
for MODEL in M1 M2 M3 M4; do
    for SEED in 42 43 44 45 46; do
        if [ "$MODEL" = "M2" ] && [ "$SEED" = "42" ]; then
            echo "skip M2/seed42 (smoke result, already present)"
            continue
        fi
        qsub -v MODEL=$MODEL,SEED=$SEED,CONT=0 run11_mri_grid.pbs
        N=$((N+1))
    done
done
echo
echo "Submitted $N jobs (19 + the existing smoke = 20)."
echo
echo "When they finish, integrity gate BEFORE any accuracy number:"
echo "  grep -c 'rc=0'              v11_mri.o*  | wc -l   # expect 19"
echo "  grep -h 'VERIFIED'          v11_mri.o*  # 1604/1610/1600/1594"
echo "  grep -h 'PROTOCOL RESOLVED' v11_mri.o*  # all 2240/560/772"
echo "  grep -h 'REDUCED-SCALE'     v11_mri.o*  # must print NOTHING"
echo "  grep -h 'DATA'              v11_mri.o*  # all Option A, all balanced"
echo
echo "Then: python scripts/collect_results.py --results_dir ../results_v11_mri"
echo "Report MDE (expected 3-4 pp) with every null."
