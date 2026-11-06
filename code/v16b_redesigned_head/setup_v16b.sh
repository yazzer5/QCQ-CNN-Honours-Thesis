#!/bin/bash
# v16b setup: build qcq_cnn_v16b from qcq_cnn_v16a (code only, data linked), patch, verify.
#   cd /g/data/wa66/yh2481/v16b_pkg && bash setup_v16b.sh
# Must end in ALL CHECKS PASSED. Refuses if qcq_cnn_v16b already exists (move it aside to start over).
set -eo pipefail
BASE=/g/data/wa66/yh2481
SRC=$BASE/qcq_cnn_v16a
DST=$BASE/qcq_cnn_v16b
PKG=$(cd "$(dirname "$0")" && pwd)

[ -e "$DST" ] && { echo "REFUSED: $DST exists. mv it aside first (only before the smoke is submitted)."; exit 1; }
grep -q "V16A_HEAD_SAVEZ" "$SRC/train_gadi.py" || { echo "REFUSED: $SRC/train_gadi.py is not v16a-patched"; exit 1; }
grep -q "V16B_BFR_HEAD" "$SRC/train_gadi.py" && { echo "REFUSED: $SRC/train_gadi.py already has the v16b block"; exit 1; }
[ -e "$SRC/data" ] || { echo "REFUSED: $SRC/data not found"; exit 1; }

echo "== build $DST from $SRC (code only)"
mkdir -p "$DST/scripts" "$DST/pbs"
cp -p "$SRC"/*.py "$DST"/
cp -p "$SRC"/scripts/*.py "$DST"/scripts/
ln -s "$(readlink -f "$SRC/data")" "$DST/data"
cp -p "$PKG"/bfr_head_v16b.py "$PKG"/patch_v16b.py "$DST"/
cp -p "$PKG"/scripts/*.py "$DST"/scripts/
cp -p "$PKG"/pbs/* "$DST"/pbs/
chmod +x "$DST"/pbs/*.sh
cp -p "$PKG"/PREREGISTRATION_v16b.md "$PKG"/README_v16b.md "$DST"/
ls -la "$DST" | head -40

echo "== patch"
cd "$DST"
python3 patch_v16b.py "$DST" 2>&1 | tee patch_v16b.log
sha256sum train_gadi.py train_gadi.py.v16abak bfr_head_v16b.py v16_tools.py closed_form_zz_v16.py qnn_torch.py quanv_torch.py \
    | tee -a patch_v16b.log
diff <(sha256sum < "$SRC/train_gadi.py") <(sha256sum < train_gadi.py.v16abak) >/dev/null \
    && echo "backup train_gadi.py.v16abak == $SRC/train_gadi.py (byte-identical)" | tee -a patch_v16b.log \
    || { echo "FATAL: backup differs from the v16a source"; exit 1; }

echo "== verify (a few minutes)"
python3 -u scripts/verify_v16b.py 2>&1 | tee verify_v16b.log || true
grep -q "^ALL CHECKS PASSED" verify_v16b.log && ! grep -q "^VERIFY FAILED" verify_v16b.log || { echo "VERIFY DID NOT PASS -- paste verify_v16b.log"; exit 1; }
echo "== setup done: $DST"
