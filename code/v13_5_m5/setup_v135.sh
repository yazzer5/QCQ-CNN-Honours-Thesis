#!/bin/bash
# Builds the v13.5 repo on Gadi from the v13 repo (which already has the v12 D1
# seed fix and the v13 readout patch), applies the M5 patch, drops in v13.5
# pbs/ + scripts/, and runs the offline verification. Stops at the first failure.
#   bash setup_v135.sh                      (defaults to /g/data/wa66/yh2481/qcq_cnn_v13)
set -eo pipefail
SRC="${1:-/g/data/wa66/yh2481/qcq_cnn_v13}"
DST=/g/data/wa66/yh2481/qcq_cnn_v135
HERE="$(cd "$(dirname "$0")" && pwd)"
[ -f "$SRC/train_gadi.py" ] || { echo "No train_gadi.py in $SRC"; exit 1; }
grep -Eq 'SEEDS[[:space:]]*=[[:space:]]*list[(]range[(]42,[[:space:]]*52[)][)]' "$SRC/train_gadi.py" || { echo "STOP: no SEEDS = list(range(42, 52)) in $SRC/train_gadi.py"; exit 1; }
grep -q 'V13_READOUT_PATCH' "$SRC/qnn_torch.py" || { echo "STOP: $SRC/qnn_torch.py is not v13-patched"; exit 1; }
[ -e "$DST" ] && { echo "STOP: $DST already exists. Remove or rename it first."; exit 1; }
mkdir -p "$DST"
rsync -a --exclude 'results*' --exclude 'ckpt*' --exclude '*.o[0-9]*' --exclude 'gpu_util*' \
      --exclude 'figures*' --exclude '__pycache__' --exclude 'pbs/' --exclude 'v1[0-9]_*.csv' \
      --exclude '*.tgz' --exclude 'verify_v13*.log' "$SRC/" "$DST/"
mkdir -p "$DST/pbs" "$DST/scripts"
cp "$HERE"/pbs/* "$DST/pbs/"
cp "$HERE"/scripts/*.py "$DST/scripts/"
cp "$HERE"/PREREGISTRATION_v13_5.md "$HERE"/README_v13_5.md "$DST/"
chmod +x "$DST"/pbs/*.sh
cd "$DST"
python3 "$HERE/patch_v135_m5.py" "$DST" | tee patch_v135.log
python3 scripts/verify_v135_m5.py | tee verify_v135.log
echo
echo "v13.5 repo ready at $DST.  Outputs saved to patch_v135.log and verify_v135.log"
