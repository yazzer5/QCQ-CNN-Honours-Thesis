#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# v15 setup: build qcq_cnn_v15 from qcq_cnn_v14, add the noise module, patch, verify.
#   cd /g/data/wa66/yh2481/v15_pkg && bash setup_v15.sh
# Must end in ALL CHECKS PASSED. Paste the WHOLE output.
# ═══════════════════════════════════════════════════════════════════════
set -e
set -o pipefail
PKG=$(cd "$(dirname "$0")" && pwd)
BASE=${BASE:-/g/data/wa66/yh2481}
SRC=$BASE/qcq_cnn_v14
DST=$BASE/qcq_cnn_v15
CONDA_SH=${CONDA_SH:-/g/data/wa66/yh2481/miniconda3/etc/profile.d/conda.sh}

[ -f "$SRC/train_gadi.py" ] || { echo "FATAL: $SRC/train_gadi.py not found"; exit 1; }
[ -e "$DST" ] && { echo "REFUSED: $DST already exists. Move it aside (mv $DST ${DST}_old) and re-run."; exit 1; }

echo "== 1. Build $DST from $SRC (code only: no results, checkpoints, logs; data linked)"
mkdir -p "$DST"
tar -C "$SRC" --exclude='./results_*' --exclude='./ckpt_*' --exclude='./figures_*' --exclude='./data' \
    --exclude='./pbs' --exclude='*.tgz' --exclude='__pycache__' --exclude='*.log' --exclude='*.csv' \
    --exclude='*.v1*bak' -cf - . | tar -C "$DST" -xf -
if [ -e "$SRC/data" ]; then ln -s "$(readlink -f "$SRC/data")" "$DST/data"; else echo "WARN: $SRC/data not found"; fi
mkdir -p "$DST/pbs" "$DST/calib" "$DST/scripts"
cp "$PKG/quanv_noise_v15.py" "$PKG/patch_v15_noise.py" "$PKG/PREREGISTRATION_v15.md" "$PKG/README_v15.md" "$DST/"
cp "$PKG"/calib/* "$DST/calib/"
cp "$PKG"/scripts/*.py "$DST/scripts/"
cp "$PKG"/pbs/* "$DST/pbs/"
chmod +x "$DST"/pbs/*.sh "$DST"/pbs/*.pbs
echo "   files in the new repo: $(find "$DST" -type f | wc -l)"

echo; echo "== 2. Patch train_gadi.py"
cd "$DST"
python3 patch_v15_noise.py . 2>&1 | tee patch_v15.log

source "$CONDA_SH"
echo; echo "== 3. Calibration and Aer reference on this machine (qcqcnn_noise env; informational)"
if conda env list | grep -q "qcqcnn_noise"; then
  conda activate qcqcnn_noise; export PYTHONPATH=""
  python scripts/extract_ibm_calibration.py --check calib/ibm_brisbane_calib.json 2>&1 | grep "V15-CALIB" || echo "   calibration re-extraction SKIPPED (see above)"
  if python -c "import qiskit_aer" 2>/dev/null; then
    python scripts/make_aer_reference_v15.py --calib calib/ibm_brisbane_calib.json --out calib/aer_reference_v15_gadi.npz \
      || { echo "   Aer reference on Gadi FAILED -- verify uses the packaged one"; rm -f calib/aer_reference_v15_gadi.npz; }
  else
    echo "   qiskit_aer not importable in qcqcnn_noise -- verify uses the packaged Aer reference"
  fi
  conda deactivate
else
  echo "   no qcqcnn_noise env -- verify uses the packaged calibration check and Aer reference"
fi

echo; echo "== 4. Verify (qcqcnn env)"
conda activate qcqcnn; export PYTHONPATH=""
python -u scripts/verify_v15.py 2>&1 | tee verify_v15.log
echo
echo "Repo: $DST    inodes now: $(find "$DST" -type f | wc -l) files"
echo "Next: README_v15.md step 2 (cheap checks), then the smoke."
