#!/bin/bash
# v16b-ext setup: build qcq_cnn_v16bx from qcq_cnn_v16b (code only, data linked), add A3 to the pre-registration copy,
# patch, verify.     cd /g/data/wa66/yh2481/v16bx_pkg && bash setup_v16bx.sh       (must end in ALL CHECKS PASSED)
set -eo pipefail
BASE=/g/data/wa66/yh2481
SRC=$BASE/qcq_cnn_v16b
DST=$BASE/qcq_cnn_v16bx
PKG=$(cd "$(dirname "$0")" && pwd)
[ -e "$DST" ] && { echo "REFUSED: $DST exists. mv it aside first (only before the smoke is submitted)."; exit 1; }
grep -q "V16B_BFR_HEAD" "$SRC/train_gadi.py" || { echo "REFUSED: $SRC/train_gadi.py is not v16b-patched"; exit 1; }
grep -q "V16BX_EXT_HEAD" "$SRC/train_gadi.py" && { echo "REFUSED: $SRC already has the v16bx block"; exit 1; }
echo "== build $DST from $SRC (code only; v16b repo and results untouched)"
mkdir -p "$DST/scripts" "$DST/pbs"
cp -p "$SRC"/*.py "$DST"/
rm -f "$DST"/train_gadi.py.v15bak "$DST"/train_gadi.py.v16abak
cp -p "$SRC"/scripts/*.py "$DST"/scripts/
ln -s "$(readlink -f "$SRC/data")" "$DST/data"
cp -p "$PKG"/bfr_ext_v16b.py "$PKG"/patch_v16bx.py "$DST"/
cp -p "$PKG"/scripts/*.py "$DST"/scripts/
cp -p "$PKG"/pbs/* "$DST"/pbs/ && chmod +x "$DST"/pbs/*.sh
cp -p "$PKG"/PREREGISTRATION_v16b.md "$PKG"/README_v16bx.md "$DST"/
grep -q "A3 (2026-10-22" "$DST/PREREGISTRATION_v16b.md" || { echo "FATAL: A3 missing from the pre-registration"; exit 1; }
echo "== patch"
cd "$DST"
python3 patch_v16bx.py "$DST" 2>&1 | tee patch_v16bx.log
sha256sum train_gadi.py train_gadi.py.v16bbak bfr_head_v16b.py bfr_ext_v16b.py | tee -a patch_v16bx.log
cmp -s "$SRC/train_gadi.py" train_gadi.py.v16bbak && echo "backup train_gadi.py.v16bbak == $SRC/train_gadi.py (byte-identical)" | tee -a patch_v16bx.log \
    || { echo "FATAL: backup differs from the v16b source"; exit 1; }
cmp -s "$SRC/bfr_head_v16b.py" bfr_head_v16b.py && echo "bfr_head_v16b.py == v16b's (byte-identical)" | tee -a patch_v16bx.log \
    || { echo "FATAL: bfr_head_v16b.py differs from v16b's"; exit 1; }
echo "== verify (a few minutes)"
python3 -u scripts/verify_v16bx.py 2>&1 | tee verify_v16bx.log || true
grep -q "^ALL CHECKS PASSED" verify_v16bx.log && ! grep -q "^VERIFY FAILED" verify_v16bx.log || { echo "VERIFY DID NOT PASS -- paste verify_v16bx.log"; exit 1; }
echo "== setup done: $DST"
