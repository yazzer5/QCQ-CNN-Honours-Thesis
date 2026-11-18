#!/usr/bin/env python3
"""Insert the v15.6 geometry hook into train_gadi.py (v15.5 code), inside _v155_xeval,
after the cross-eval and before _v155_rng_restore(st). Nothing else changes.

usage: python3 patch_v156_geom.py <train_gadi.py>      (patches in place, writes .v155bak)
Refuses if the anchor is not found exactly once, is not inside _v155_xeval, or the file is already patched.
"""
import shutil
import sys

ANCHOR = ('        _v155_rng_restore(st)\n'
          '        out = _v155_os.path.join(a.output_dir, "v155_xeval_%s_seed%s.json" % (M, S))\n')
BEGIN = "        # === V156_GEOM begin"
END = "        # === V156_GEOM end ===\n"
BLOCK = (
    BEGIN + " (v15.6 geometry pass; private RNGs only; runs after training, inert unless enabled) ===\n"
    '        if _v155_os.environ.get("V156_GEOM", "0") == "1" or _v155_os.path.exists(_v155_os.path.join(\n'
    '                _v155_os.path.dirname(_v155_os.path.abspath(__file__)), "V156_GEOM_ON")):\n'
    "            try:\n"
    "                import geom_v156 as _g156\n"
    "                _g156.run(model=model, Xte=Xte, res=res, sigma_train=sigma, M=M, S=S, out_dir=a.output_dir,\n"
    '                          fixed_fn=lambda X_, lv_: _v155_fixed(X_, "gauss", lv_, V155_TEST_GAUSS_BASE),\n'
    '                          bs=globals().get("BATCH_SIZE", 32))\n'
    "            except Exception as _e156:  # noqa: BLE001\n"
    "                import traceback as _tb156\n"
    "                _tb156.print_exc()\n"
    '                print("[V156] GEOM FAILED: %s: %s" % (type(_e156).__name__, _e156), flush=True)\n'
    + END)


def main():
    path = sys.argv[1]
    src = open(path).read()
    if "V156_GEOM begin" in src:
        sys.exit("REFUSED: already patched")
    n = src.count(ANCHOR)
    if n != 1:
        sys.exit("REFUSED: anchor found %d times (need 1)" % n)
    pos = src.index(ANCHOR)
    fdef = src.rfind("\ndef ", 0, pos)
    if not src[fdef:].startswith("\ndef _v155_xeval("):
        sys.exit("REFUSED: anchor is not inside _v155_xeval")
    shutil.copy2(path, path + ".v155bak")
    open(path, "w").write(src[:pos] + BLOCK + src[pos:])
    print("PATCHED %s: inserted %d lines at line %d (inside _v155_xeval); backup %s.v155bak"
          % (path, BLOCK.count("\n"), src[:pos].count("\n") + 1, path))


if __name__ == "__main__":
    main()
