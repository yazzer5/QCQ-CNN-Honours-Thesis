#!/bin/bash
# Builds the v12 repo on Gadi: copies the CODE (not results/checkpoints/logs)
# from the last clean grid, then drops the v12 pbs/ and scripts/ on top.
#   bash setup_v12.sh /g/data/wa66/yh2481/v10_grid
# Use whichever folder holds the newest train_gadi.py that ran clean
# (v10_grid, or your v11 folder if that code is newer).
set -e
SRC="${1:?usage: bash setup_v12.sh /path/to/last/clean/repo}"
DST=/g/data/wa66/yh2481/qcq_cnn_v12
HERE="$(cd "$(dirname "$0")" && pwd)"
[ -f "$SRC/train_gadi.py" ] || { echo "No train_gadi.py in $SRC"; exit 1; }
mkdir -p "$DST"
rsync -a --exclude 'results*' --exclude 'ckpt*' --exclude '*.o[0-9]*' \
      --exclude 'gpu_util*' --exclude 'figures*' --exclude '__pycache__' \
      --exclude 'pbs/' "$SRC/" "$DST/"
mkdir -p "$DST/pbs" "$DST/scripts"
cp "$HERE"/pbs/* "$DST/pbs/"
cp "$HERE"/scripts/*.py "$DST/scripts/"
cp "$HERE"/PREREGISTRATION_v12.md "$HERE"/README_v12.md "$DST/"
chmod +x "$DST"/pbs/*.sh
echo "v12 repo ready at $DST (code copied from $SRC)"
ls "$DST"
