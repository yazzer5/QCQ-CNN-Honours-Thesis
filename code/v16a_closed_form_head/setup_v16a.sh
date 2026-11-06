#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# v16a setup: build qcq_cnn_v16a from qcq_cnn_v15 (code only), add the v16a files, patch, verify.
#   cd /g/data/wa66/yh2481/v16a_pkg && bash setup_v16a.sh
# Must end in ALL CHECKS PASSED. Paste the WHOLE output.
# ═══════════════════════════════════════════════════════════════════════
set -e
set -o pipefail
PKG=$(cd "$(dirname "$0")" && pwd)
BASE=${BASE:-/g/data/wa66/yh2481}
SRC=$BASE/qcq_cnn_v15
DST=$BASE/qcq_cnn_v16a
CONDA_SH=${CONDA_SH:-/g/data/wa66/yh2481/miniconda3/etc/profile.d/conda.sh}

[ -f "$SRC/train_gadi.py" ] || { echo "FATAL: $SRC/train_gadi.py not found"; exit 1; }
[ -e "$DST" ] && { echo "REFUSED: $DST already exists. Move it aside (mv $DST ${DST}_old) and re-run."; exit 1; }
command -v lquota >/dev/null && lquota 2>/dev/null | head -6 || true

echo "== 1. Build $DST from $SRC (code only: no results, checkpoints, logs, figures; data linked)"
mkdir -p "$DST"
EXCL=(--exclude 'results_*' --exclude 'ckpt_*' --exclude 'figures_*' --exclude '*.tgz' --exclude '__pycache__'
      --exclude '*.log' --exclude '*.csv' --exclude '*.v1*bak' --exclude '*.o[0-9]*')
if command -v rsync >/dev/null; then
  rsync -a "${EXCL[@]}" --exclude '/data' --exclude '/pbs' "$SRC/" "$DST/"
else
  (cd "$SRC" && tar cf - "${EXCL[@]}" --exclude './data' --exclude './pbs' .) | (cd "$DST" && tar xf -)
fi
if [ -e "$SRC/data" ]; then ln -s "$(readlink -f "$SRC/data")" "$DST/data"; else echo "WARN: $SRC/data not found"; fi
mkdir -p "$DST/pbs" "$DST/scripts"
cp "$PKG/closed_form_zz_v16.py" "$PKG/v16_tools.py" "$PKG/patch_v16a.py" "$PKG/PREREGISTRATION_v16a.md" "$PKG/README_v16a.md" "$DST/"
cp "$PKG"/scripts/*.py "$DST/scripts/"
cp "$PKG"/pbs/* "$DST/pbs/"
chmod +x "$DST"/pbs/*.sh "$DST"/pbs/*.pbs
echo "   files in the new repo: $(find "$DST" -type f | wc -l)"
echo "   train_gadi.py sha256 (source v15): $(sha256sum "$SRC/train_gadi.py" | cut -c1-16)"

echo; echo "== 2. Patch train_gadi.py"
cd "$DST"
[ -n "$CONDA_DEFAULT_ENV" ] || { source "$CONDA_SH"; conda activate qcqcnn; }
export PYTHONPATH=""
python3 patch_v16a.py . 2>&1 | tee patch_v16a.log

echo; echo "== 3. Verify (qcqcnn env, CPU)"
unset V15_ALPHA V15_CALIB V15_EVAL_ALPHAS
python3 -u scripts/verify_v16a.py 2>&1 | tee verify_v16a.log
echo
echo "Repo: $DST    files now: $(find "$DST" -type f | wc -l)"
echo "Next: README_v16a.md step 3 (cheap checks), then the smoke."
