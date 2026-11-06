#!/bin/bash
# Builds the v13 repo on Gadi from the v12 repo (which has the D1 seed fix),
# applies the readout patch, drops in v13 pbs/ + scripts/, and runs the
# offline verification. Stops at the first failure.
#   bash setup_v13.sh                      (defaults to /g/data/wa66/yh2481/qcq_cnn_v12)
set -eo pipefail
SRC="${1:-/g/data/wa66/yh2481/qcq_cnn_v12}"
DST=/g/data/wa66/yh2481/qcq_cnn_v13
HERE="$(cd "$(dirname "$0")" && pwd)"
[ -f "$SRC/train_gadi.py" ] || { echo "No train_gadi.py in $SRC"; exit 1; }
grep -Eq '^SEEDS[[:space:]]*=[[:space:]]*list\(range\(42, *52\)\)' "$SRC/train_gadi.py" || { echo "STOP: no SEEDS = list(range(42, 52)) in $SRC/train_gadi.py (v12 D1 fix). Paste: grep -n SEEDS $SRC/train_gadi.py"; exit 1; }
[ -e "$DST" ] && { echo "STOP: $DST already exists. Remove or rename it first."; exit 1; }
mkdir -p "$DST"
rsync -a --exclude 'results*' --exclude 'ckpt*' --exclude '*.o[0-9]*' --exclude 'gpu_util*' \
      --exclude 'figures*' --exclude '__pycache__' --exclude 'pbs/' --exclude 'v12_*.csv' \
      --exclude '*.tgz' "$SRC/" "$DST/"
mkdir -p "$DST/pbs" "$DST/scripts"
cp "$HERE"/pbs/* "$DST/pbs/"
cp "$HERE"/scripts/*.py "$DST/scripts/"
cp "$HERE"/PREREGISTRATION_v13.md "$HERE"/README_v13.md "$HERE"/V14_SCOPE.md "$DST/"
chmod +x "$DST"/pbs/*.sh
cd "$DST"
python3 "$HERE/patch_v13_readout.py" "$DST"
python3 scripts/verify_v13_readout.py | tee verify_v13.log
echo
echo "v13 repo ready at $DST.  verify output saved to verify_v13.log"
