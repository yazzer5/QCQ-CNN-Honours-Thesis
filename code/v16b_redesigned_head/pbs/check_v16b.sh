#!/bin/bash
# Integrity gate. Must print ALL CHECKS PASSED before any accuracy is read.
cd "$(dirname "$0")"
[ -n "$CONDA_DEFAULT_ENV" ] || { source /g/data/wa66/yh2481/miniconda3/etc/profile.d/conda.sh 2>/dev/null && conda activate qcqcnn; }
python3 ../scripts/gate_v16b.py "$@"
