#!/usr/bin/env python3
"""patch_v157.py <train_gadi.py (v156-patched)>: two inert blocks.
 (1) before `if __name__ == "__main__":` wrap build_model so every built model gets the v15.7 tracker
 (2) after the V156_GEOM block in _v155_xeval, save the tracker records of the MATCH model
Both run only when V157_TRACK_ON sits beside train_gadi.py."""
import shutil
import sys

MAIN = '\n\nif __name__ == "__main__":\n    main()'
B1 = (
    "\n\n# === V157_TRACK begin (wrap build_model: attach the v15.7 bubble tracker; inert unless enabled) ===\n"
    'if _v155_os.path.exists(_v155_os.path.join(_v155_os.path.dirname(_v155_os.path.abspath(__file__)), "V157_TRACK_ON")):\n'
    "    _v157_build_model_orig = build_model\n\n"
    "    def build_model(*_a, **_k):\n"
    "        _m = _v157_build_model_orig(*_a, **_k)\n"
    "        try:\n"
    "            import track_v157 as _t157\n"
    '            _t157.install(_m, dict(get_test=lambda: _V155.get("test"), stack=_v155_stack, bs=globals().get("BATCH_SIZE", 32),\n'
    '                                   fixed=lambda X_, lv_: _v155_fixed(X_, "gauss", lv_, V155_TEST_GAUSS_BASE)))\n'
    "        except Exception as _e157:  # noqa: BLE001\n"
    '            print("[V157] TRACK FAILED (install): %s: %s" % (type(_e157).__name__, _e157), flush=True)\n'
    "        return _m\n"
    "# === V157_TRACK end ===")
ANCH2 = "        # === V156_GEOM end ===\n"
B2 = (
    "        # === V157_SAVE begin ===\n"
    '        if _v155_os.path.exists(_v155_os.path.join(_v155_os.path.dirname(_v155_os.path.abspath(__file__)), "V157_TRACK_ON")):\n'
    "            try:\n"
    "                import track_v157 as _t157s\n"
    "                _t157s.save(model, a.output_dir, M, S, sigma)\n"
    "            except Exception as _e157s:  # noqa: BLE001\n"
    '                print("[V157] TRACK FAILED (save): %s: %s" % (type(_e157s).__name__, _e157s), flush=True)\n'
    "        # === V157_SAVE end ===\n")


def main():
    p = sys.argv[1]
    s = open(p).read()
    if "V157_TRACK begin" in s:
        sys.exit("REFUSED: already patched")
    if s.count(MAIN) != 1 or not s.rstrip().endswith('main()'):
        sys.exit("REFUSED: __main__ anchor not found exactly once at file end")
    if s.count(ANCH2) != 1:
        sys.exit("REFUSED: V156_GEOM end anchor not found exactly once (is this the v156 trainer?)")
    shutil.copy2(p, p + ".v156bak")
    s = s.replace(ANCH2, ANCH2 + B2)
    s = s.replace(MAIN, B1 + MAIN)
    open(p, "w").write(s)
    print("PATCHED %s: build_model wrapper (%d lines) + save block (%d lines)" % (p, B1.count("\n"), B2.count("\n")))


if __name__ == "__main__":
    main()
