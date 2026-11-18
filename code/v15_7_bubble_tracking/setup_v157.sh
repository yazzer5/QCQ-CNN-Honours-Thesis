#!/bin/bash
# setup_v157.sh: build qcq_cnn_v157 from qcq_cnn_v156 (code only), patch, verify. Run with qcqcnn active.
set -euo pipefail
BASE=/g/data/wa66/yh2481; SRC=$BASE/qcq_cnn_v156; DST=$BASE/qcq_cnn_v157
PKG=$(cd "$(dirname "$0")" && pwd)
[ -e "$DST" ] && { echo "REFUSED: $DST exists"; exit 1; }
grep -q "V156_GEOM begin" "$SRC/train_gadi.py" || { echo "REFUSED: $SRC/train_gadi.py is not v156-patched"; exit 1; }
mkdir -p "$DST/pbs"
cp "$SRC"/*.py "$DST"/
cp -r "$SRC/scripts" "$DST"/
ln -s "$(readlink -f "$SRC/data")" "$DST/data"
for f in run155.pbs config155.sh names155.sh; do
  sed -e 's#qcq_cnn_v156#qcq_cnn_v157#g' -e 's#results_v156#results_v157#g' "$SRC/pbs/$f" > "$DST/pbs/$f"
done
grep -q 'results_v157_' "$DST/pbs/config155.sh" || { echo "REFUSED: out_dir rename failed"; exit 1; }
cp "$PKG"/track_v157.py "$PKG"/patch_v157.py "$PKG"/verify_v157.py "$PKG"/gate_v157.py "$PKG"/analyse_v157.py "$DST"/
cp "$PKG"/submit_v157_smoke.sh "$PKG"/submit_v157_grid.sh "$DST/pbs/" && chmod +x "$DST"/pbs/*.sh
cd "$DST"
python3 patch_v157.py train_gadi.py && rm -f train_gadi.py.v156bak
touch V156_GEOM_ON V157_TRACK_ON
echo "---- verify ----"
python3 verify_v157.py "$DST" "$SRC" 2>&1 | tee verify_v157.log
echo "---- inodes ----"; lquota | grep gdata
