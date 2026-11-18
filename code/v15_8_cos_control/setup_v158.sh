#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# v15.8 setup: build qcq_cnn_v158 from qcq_cnn_v155 (code only, data linked), add M6, verify.
#   cd /g/data/wa66/yh2481/v158_pkg && bash setup_v158.sh 2>&1 | tee ~/setup_v158.out
# Must end in ALL CHECKS PASSED.
# ═══════════════════════════════════════════════════════════════════════
set -e
set -o pipefail
PKG=$(cd "$(dirname "$0")" && pwd)
BASE=${BASE:-/g/data/wa66/yh2481}
SRC=$BASE/qcq_cnn_v155
DST=$BASE/qcq_cnn_v158
CONDA_SH=${CONDA_SH:-/g/data/wa66/yh2481/miniconda3/etc/profile.d/conda.sh}

[ -f "$SRC/train_gadi.py" ] || { echo "FATAL: $SRC/train_gadi.py not found"; exit 1; }
[ -e "$DST" ] && { echo "REFUSED: $DST already exists. Move it aside (mv $DST ${DST}_old) and re-run."; exit 1; }
grep -q "V155_INPUT_NOISE" "$SRC/train_gadi.py" || { echo "FATAL: $SRC/train_gadi.py is not the v15.5 trainer"; exit 1; }
for g in v156 v157 V156 V157 V158; do
  if grep -q "$g" "$SRC/train_gadi.py"; then echo "FATAL: $SRC/train_gadi.py mentions $g; need the clean v15.5 file"; exit 1; fi
done
echo "source train_gadi.py: $(md5sum "$SRC/train_gadi.py" | cut -d' ' -f1)  ($(wc -l < "$SRC/train_gadi.py") lines)"
n_ref=$( (ls "$SRC"/results_v155_k*_s0/result_M4_*seed4[2-6]_*.json 2>/dev/null || true) | wc -l)
echo "stored v15.5 M4 sigma=0 runs for seeds 42-46: $n_ref (expect 15)"

source "$CONDA_SH"
conda activate qcqcnn; export PYTHONPATH=""
echo "python: $(python --version 2>&1)  torch: $(python -c 'import torch; print(torch.__version__)')"

echo "== 1. Build $DST from $SRC (code only; data linked)"
mkdir -p "$DST"
tar -C "$SRC" --exclude='./results_*' --exclude='./ckpt_*' --exclude='./figures_*' --exclude='./data' \
    --exclude='./pbs' --exclude='*.tgz' --exclude='__pycache__' --exclude='*.log' --exclude='*.csv' \
    --exclude='*.npz' --exclude='./logs' --exclude='./v15*_out*' -cf - . | tar -C "$DST" -xf -
if [ -e "$SRC/data" ]; then ln -s "$(readlink -f "$SRC/data")" "$DST/data"; else echo "WARN: $SRC/data not found"; fi
mkdir -p "$DST/pbs" "$DST/scripts"
cp "$PKG/patch_v158_m6.py" "$PKG/README_v158.md" "$DST/"
cp "$PKG"/scripts/*.py "$DST/scripts/"
cp "$PKG"/pbs/* "$DST/pbs/"
chmod +x "$DST"/pbs/*.sh "$DST"/pbs/*.pbs
[ -f "$DST/scripts/preflight_data.py" ] || { echo "FATAL: scripts/preflight_data.py missing in the copy"; exit 1; }
echo "   files in the new repo: $(find "$DST" -type f | wc -l)"

echo; echo "== 2. Patch train_gadi.py (add M6)"
cd "$DST"
python patch_v158_m6.py . 2>&1 | tee patch_v158.log

echo; echo "== 3. Verify (CPU, ~5-10 min)"
python -u scripts/verify_v158.py "$SRC" 2>&1 | tee verify_v158.log
echo
echo "Repo: $DST"
