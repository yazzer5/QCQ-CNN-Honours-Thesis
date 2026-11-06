#!/bin/bash
# Builds the v14 repo on Gadi from the v13.5 repo (v12 seed fix + v13 readout
# patch + v13.5 M5 patch with the A1 warmup fix), applies the v14 split-audit
# patch, drops in v14 pbs/ + scripts/, and runs the offline verification.
# Stops at the first failure.
#   bash setup_v14.sh                      (defaults to /g/data/wa66/yh2481/qcq_cnn_v135)
set -eo pipefail
SRC="${1:-/g/data/wa66/yh2481/qcq_cnn_v135}"
DST=/g/data/wa66/yh2481/qcq_cnn_v14
HERE="$(cd "$(dirname "$0")" && pwd)"
[ -f "$SRC/train_gadi.py" ] || { echo "No train_gadi.py in $SRC"; exit 1; }
grep -q 'V135_M5_PATCH' "$SRC/train_gadi.py" || { echo "STOP: $SRC/train_gadi.py has no v13.5 M5 patch"; exit 1; }
grep -q 'model_name in ("M2", "M3", "M5")' "$SRC/train_gadi.py" || { echo "STOP: A1-v13.5 warmup fix missing in $SRC"; exit 1; }
grep -q 'V13_READOUT_PATCH' "$SRC/qnn_torch.py" || { echo "STOP: $SRC/qnn_torch.py is not v13-patched"; exit 1; }
[ -e "$DST" ] && { echo "STOP: $DST already exists. Remove or rename it first."; exit 1; }
mkdir -p "$DST"
rsync -a --exclude 'results*' --exclude 'ckpt*' --exclude '*.o[0-9]*' --exclude 'gpu_util*' \
      --exclude 'figures*' --exclude '__pycache__' --exclude 'pbs/' --exclude 'v1[0-9]*_*.csv' \
      --exclude '*.tgz' --exclude 'verify_v1*.log' --exclude 'patch_v1*.log' --exclude 'stray_*' \
      --exclude '*.v13bak' "$SRC/" "$DST/"
mkdir -p "$DST/pbs" "$DST/scripts"
cp "$HERE"/pbs/* "$DST/pbs/"
cp "$HERE"/scripts/*.py "$DST/scripts/"
cp "$HERE"/PREREGISTRATION_v14.md "$HERE"/README_v14.md "$DST/"
chmod +x "$DST"/pbs/*.sh
cd "$DST"
python3 "$HERE/patch_v14_lowdata.py" "$DST" | tee patch_v14.log
python3 -u scripts/verify_v14.py 2>&1 | tee verify_v14.log
echo
echo "v14 repo ready at $DST.  Outputs saved to patch_v14.log and verify_v14.log"
