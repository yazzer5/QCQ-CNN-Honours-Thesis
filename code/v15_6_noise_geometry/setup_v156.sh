#!/bin/bash
# setup_v156.sh : build /g/data/wa66/yh2481/qcq_cnn_v156 from qcq_cnn_v155 (code only), patch, verify.
# Run from the unpacked package dir, with the qcqcnn env active.
set -euo pipefail
BASE=/g/data/wa66/yh2481
SRC=$BASE/qcq_cnn_v155
DST=$BASE/qcq_cnn_v156
PKG=$(cd "$(dirname "$0")" && pwd)
[ -e "$DST" ] && { echo "REFUSED: $DST exists"; exit 1; }
mkdir -p "$DST/pbs"
cp "$SRC"/*.py "$DST"/
cp -r "$SRC/scripts" "$DST"/ 2>/dev/null || true
ln -s "$(readlink -f "$SRC/data")" "$DST/data"
for f in run155.pbs config155.sh names155.sh; do
  sed -e 's#qcq_cnn_v155#qcq_cnn_v156#g' -e 's#results_v155#results_v156#g' "$SRC/pbs/$f" > "$DST/pbs/$f"
done
cp "$PKG"/geom_v156.py "$PKG"/patch_v156_geom.py "$PKG"/verify_v156.py "$PKG"/gate_v156.py "$PKG"/analyse_v156.py "$DST"/
cp "$PKG"/PREREGISTRATION_v15_6.md "$DST"/ 2>/dev/null || true
cd "$DST"
python3 patch_v156_geom.py train_gadi.py
rm -f train_gadi.py.v155bak
touch V156_GEOM_ON
echo "---- verify (CPU, ~2 min) ----"
python3 verify_v156.py "$DST" "$SRC" 2>&1 | tee verify_v156.log
echo
echo "---- inodes ----"; lquota | grep gdata; du --inodes -s "$DST"
echo
echo "==== v15.5 submit layer (template for the v15.6 submit scripts) ===="
for f in run155.pbs config155.sh names155.sh submit_v155_smoke.sh submit_v155_grid.sh; do
  echo "######## $f"; cat "$SRC/pbs/$f"
done
