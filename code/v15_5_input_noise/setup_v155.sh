#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# v15.5 setup: build qcq_cnn_v155 from qcq_cnn_v14, patch, verify.
#   cd /g/data/wa66/yh2481/v155_pkg && bash setup_v155.sh
# Must end in ALL CHECKS PASSED. Paste the WHOLE output.
# ═══════════════════════════════════════════════════════════════════════
set -e
set -o pipefail
PKG=$(cd "$(dirname "$0")" && pwd)
BASE=${BASE:-/g/data/wa66/yh2481}
SRC=$BASE/qcq_cnn_v14
DST=$BASE/qcq_cnn_v155
CONDA_SH=${CONDA_SH:-/g/data/wa66/yh2481/miniconda3/etc/profile.d/conda.sh}

[ -f "$SRC/train_gadi.py" ] || { echo "FATAL: $SRC/train_gadi.py not found"; exit 1; }
[ -e "$DST" ] && { echo "REFUSED: $DST already exists. Move it aside (mv $DST ${DST}_old) and re-run."; exit 1; }
if grep -q "V15_FE_NOISE" "$SRC/train_gadi.py"; then echo "FATAL: $SRC/train_gadi.py carries the v15 patch; need the clean v14 file"; exit 1; fi
echo "source train_gadi.py: $(md5sum "$SRC/train_gadi.py" | cut -d' ' -f1)  ($(wc -l < "$SRC/train_gadi.py") lines)"

source "$CONDA_SH"
conda activate qcqcnn; export PYTHONPATH=""
echo "python: $(python --version 2>&1)  torch: $(python -c 'import torch; print(torch.__version__)')"

echo "== 1. Build $DST from $SRC (code only: no results, checkpoints, logs; data linked)"
mkdir -p "$DST"
tar -C "$SRC" --exclude='./results_*' --exclude='./ckpt_*' --exclude='./figures_*' --exclude='./data' \
    --exclude='./pbs' --exclude='*.tgz' --exclude='__pycache__' --exclude='*.log' --exclude='*.csv' \
    --exclude='*.v1*bak' --exclude='./logs' -cf - . | tar -C "$DST" -xf -
if [ -e "$SRC/data" ]; then ln -s "$(readlink -f "$SRC/data")" "$DST/data"; else echo "WARN: $SRC/data not found"; fi
mkdir -p "$DST/pbs" "$DST/scripts"
cp "$PKG/patch_v155_input_noise.py" "$PKG/PREREGISTRATION_v15_5.md" "$PKG/README_v155.md" "$DST/"
cp "$PKG"/scripts/*.py "$DST/scripts/"
cp "$PKG"/pbs/* "$DST/pbs/"
chmod +x "$DST"/pbs/*.sh "$DST"/pbs/*.pbs
echo "   files in the new repo: $(find "$DST" -type f | wc -l)"

echo; echo "== 2. Patch train_gadi.py"
cd "$DST"
python patch_v155_input_noise.py . 2>&1 | tee patch_v155.log

echo; echo "== 3. Verify (qcqcnn env, CPU, ~5-15 min)"
python -u scripts/verify_v155.py 2>&1 | tee verify_v155.log
echo
echo "Repo: $DST    files: $(find "$DST" -type f | wc -l)"
echo "Next: README_v155.md step 2 (cheap checks), then the smoke."
