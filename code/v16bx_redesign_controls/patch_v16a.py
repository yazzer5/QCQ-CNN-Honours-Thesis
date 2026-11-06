#!/usr/bin/env python3
"""
v16a patch: the M2c head (closed-form replica of the w15 QNN head) and --save_z.

    python3 patch_v16a.py /g/data/wa66/yh2481/qcq_cnn_v16a        (setup_v16a.sh does this)

Inserts ONE block into train_gadi.py, just above `if __name__ == "__main__":`.
No existing line is edited. The block reads three new command-line flags and removes
them from sys.argv before train_gadi.py's own argparse runs:

  --v16_head m2c   only with --model M2. After M2 is built, the QNN module and the
                   Linear(15 -> K) reading it are replaced by ZZTrig12 (12 closed-form
                   features, 0 parameters) and a fresh Linear(12 -> K) (v16_tools.install_m2c).
                   Every RNG state is restored, so data order is M2's. expected_params(M2, ...)
                   returns the M2c count while the head is active (and M2's count inside
                   build_model, where M2 is what exists).
  --save_z         at exit: fc2's test-set outputs z -> <output_dir>/v16_z_<MODEL>_seed<S>.npz
  --v16_sidecar    at exit: <output_dir>/v16_run_<MODEL>_seed<S>.json (labels, self-check).
                   Implied by the two flags above.

At exit (any flag): the trained model in memory (or, failing that, a checkpoint) is run on
the test set and must reproduce the per-image predictions train_gadi.py saved: selfcheck=MATCH.
It runs after the result JSON is written and restores every RNG state, so it cannot change it.

No flag -> the block is a pass-through (verify_v16a.py proves M1-M5 are bit-identical).
Backup: train_gadi.py.v15bak.
"""
import re
import shutil
import sys
from pathlib import Path

MARK = "V16A_HEAD_SAVEZ"

BLOCK = r'''
# ═══════════════════════════════════════════════════════════════════════════════
# V16A_HEAD_SAVEZ  (inserted by patch_v16a.py -- v16a M2c head + --save_z)
# Active only with --v16_head / --save_z / --v16_sidecar (removed from sys.argv here).
# Wraps build_model / expected_params / load_dataset by name; edits no existing code.
# ═══════════════════════════════════════════════════════════════════════════════
import atexit as _v16_atexit
import glob as _v16_glob
import json as _v16_json
import os as _v16_os
import sys as _v16_sys
import time as _v16_time

_V16 = dict(head=None, save_z=False, sidecar=False, in_build=False, model=None, name=None, seed=None,
            test=None, info=None, registered=False, t0=_v16_time.time(), argv=list(_v16_sys.argv))


def _v16_strip_argv():
    out, i, av = [], 0, _v16_sys.argv
    while i < len(av):
        t = av[i]
        if t == "--save_z":
            _V16["save_z"] = True
        elif t == "--v16_sidecar":
            _V16["sidecar"] = True
        elif t == "--v16_head":
            _V16["head"] = av[i + 1] if i + 1 < len(av) else ""
            i += 1
        elif t.startswith("--v16_head="):
            _V16["head"] = t.split("=", 1)[1]
        else:
            out.append(t)
        i += 1
    _v16_sys.argv[:] = out
    if _V16["head"] not in (None, "m2c"):
        _v16_sys.exit(f"[V16] FATAL: --v16_head {_V16['head']!r} (only 'm2c' exists)")


_v16_strip_argv()


def _v16_active():
    return bool(_V16["head"] or _V16["save_z"] or _V16["sidecar"])


def _v16_label():
    return "M2c" if _V16["head"] == "m2c" else _V16["name"]


def _v16_args():
    import argparse as _ap
    p = _ap.ArgumentParser(add_help=False)
    for a in ("--model", "--output_dir", "--checkpoint_dir", "--classes"):
        p.add_argument(a, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_known_args(_V16["argv"][1:])[0]


_v16_build_model_orig = build_model  # noqa: F821


def build_model(*args, **kwargs):  # noqa: F811
    _V16["in_build"] = True
    try:
        model = _v16_build_model_orig(*args, **kwargs)
    finally:
        _V16["in_build"] = False
    if not _v16_active():
        return model
    name = args[0] if args else kwargs.get("model_name", kwargs.get("name", kwargs.get("model")))
    seed = kwargs.get("seed", args[1] if len(args) > 1 else None)
    _V16.update(model=model, name=name, seed=seed)
    if _V16["head"] == "m2c":
        if name != "M2":
            _v16_sys.exit(f"[V16] FATAL: --v16_head m2c needs --model M2 (got {name})")
        import v16_tools as _vt
        try:
            info = _vt.install_m2c(model, seed if seed is not None else 0)
        except _vt.V16Error as e:
            _v16_sys.exit(f"[V16] FATAL: M2c surgery refused: {e}")
        table = {(5, 15): 1665, (10, 15): 1730}.get((info["K"], info["n_obs"]))
        ok = info["delta_ok"] and (table is None or info["params_m2c"] == table)
        info["table"] = table
        _V16["info"] = info
        print(f"[V16] HEAD m2c installed model=M2 seed={seed} qnn={info['qnn']} readout={info['readout']} "
              f"z={info['z_layer']} wiring=identity params_M2={info['params_m2']} params_M2c={info['params_m2c']} "
              f"expected_M2c={table} {'VERIFIED' if ok else 'MISMATCH'}", flush=True)
        if not ok:
            _v16_sys.exit("[V16] FATAL: M2c parameter count is not the pre-registered one")
    if not _V16["registered"]:
        _v16_atexit.register(_v16_finish)
        _V16["registered"] = True
    return model


if "expected_params" in globals():
    _v16_ep_orig = expected_params  # noqa: F821

    def expected_params(*args, **kwargs):  # noqa: F811
        n = _v16_ep_orig(*args, **kwargs)
        if _V16["head"] == "m2c" and not _V16["in_build"]:
            name = args[0] if args else kwargs.get("model_name", kwargs.get("name", kwargs.get("model")))
            if name == "M2":
                ints = [v for v in list(args[1:]) + list(kwargs.values())
                        if isinstance(v, int) and not isinstance(v, bool)]
                K, W = (ints[0], ints[1]) if len(ints) >= 2 else (N_CLASSES, N_OBS)  # noqa: F821
                n = n - 6 - (W + 1) * K + 13 * K
        return n


_v16_load_dataset_orig = load_dataset  # noqa: F821


def load_dataset(*args, **kwargs):  # noqa: F811
    res = _v16_load_dataset_orig(*args, **kwargs)
    if _v16_active() and isinstance(res, (tuple, list)) and len(res) >= 3:
        _V16["test"] = res[2]
    return res


def _v16_saved_preds(j, n):
    for k in ("predictions", "test_predictions", "preds", "y_pred", "test_preds"):
        v = j.get(k)
        if isinstance(v, list) and len(v) == n:
            if v and isinstance(v[0], list):
                return k, [max(range(len(r)), key=r.__getitem__) for r in v]
            return k, [int(p) for p in v]
    return None, None


def _v16_finish():
    import numpy as _np
    import torch as _t
    import v16_tools as _vt
    try:
        a = _v16_args()
        model, ds = _V16["model"], _V16["test"]
        M, S = _v16_label(), (a.seed if a.seed is not None else _V16["seed"])
        if model is None or ds is None or a.output_dir is None:
            print("[V16] RUN skipped: model, test split or --output_dir not captured", flush=True)
            return
        arch = a.model or _V16["name"]
        cands = [f for f in _v16_glob.glob(_v16_os.path.join(a.output_dir, f"result_{arch}_*seed{S}_*.json"))
                 if _v16_os.path.getmtime(f) >= _V16["t0"] - 1]
        if not cands:
            print(f"[V16] RUN skipped: no result JSON written by this run in {a.output_dir}", flush=True)
            return
        jf = max(cands, key=_v16_os.path.getmtime)
        j = _v16_json.load(open(jf))
        st = _vt.rng_save()
        classes = [int(c) for c in (a.classes or "").replace(" ", "").split(",") if c != ""]
        preds, labels, z = _vt.predict_with_z(model, ds)
        key, saved = _v16_saved_preds(j, len(preds))

        def enc(p):
            if saved is None:
                return None
            if p == saved:
                return "index"
            if classes and len(classes) > max(p) and [classes[i] for i in p] == saved:
                return "class"
            return None

        source, encoding = "memory", enc(preds)
        if encoding is None and a.checkpoint_dir and _v16_os.path.isdir(a.checkpoint_dir):
            for ck in sorted(_v16_glob.glob(_v16_os.path.join(a.checkpoint_dir, "*.pt*"))):
                try:
                    obj = _t.load(ck, map_location=next(model.parameters()).device)
                except Exception:  # noqa: BLE001
                    continue
                sd = None
                if isinstance(obj, dict):
                    for k in ("model_state_dict", "state_dict", "model", "best_state", "best_model_state"):
                        if isinstance(obj.get(k), dict):
                            sd = obj[k]; break
                    if sd is None and obj and all(hasattr(v, "shape") for v in obj.values()):
                        sd = obj
                if sd is None:
                    continue
                try:
                    model.load_state_dict(sd, strict=True)
                except Exception:  # noqa: BLE001
                    continue
                preds, labels, z = _vt.predict_with_z(model, ds)
                encoding = enc(preds)
                if encoding is not None:
                    source = "ckpt:" + _v16_os.path.basename(ck)
                    break
        _vt.rng_restore(st)
        acc = sum(int(p == y) for p, y in zip(preds, labels)) / len(labels)
        jacc = float(j.get("test_acc", float("nan")))
        acc_ok = abs(acc - jacc) < 1e-6 or abs(100 * acc - jacc) < 1e-4
        match = encoding is not None and acc_ok
        from closed_form_zz_v16 import FEATURE_NAMES as _FN
        info = _V16["info"] or {}
        res = dict(model=M, arch=arch, head=_V16["head"] or "native", seed=S, result_json=_v16_os.path.basename(jf),
                   predictions_key=key, pred_encoding=encoding, source=source,
                   selfcheck="MATCH" if match else "MISMATCH", n_test=len(labels), acc=acc, test_acc_json=jacc,
                   params=_vt.n_params(model), m2c=info, features=_FN if _V16["head"] == "m2c" else None,
                   labels=labels, classes=classes, z_saved=False)
        if _V16["save_z"]:
            zf = _v16_os.path.join(a.output_dir, f"v16_z_{M}_seed{S}.npz")
            _np.savez(zf, z=z, labels=_np.asarray(labels), preds=_np.asarray(preds))
            res.update(z_saved=True, z_file=_v16_os.path.basename(zf), z_shape=list(z.shape), z_dtype=str(z.dtype),
                       z_sha256=_vt.sha(z), z_min=z.min(0).tolist(), z_max=z.max(0).tolist(),
                       z_mean=z.mean(0).tolist())
            print(f"[V16] SAVEZ model={M} seed={S} n={z.shape[0]} dim={z.shape[1]} file={_v16_os.path.basename(zf)} "
                  f"sha={res['z_sha256'][:12]} selfcheck={res['selfcheck']}", flush=True)
        with open(_v16_os.path.join(a.output_dir, f"v16_run_{M}_seed{S}.json"), "w") as f:
            _v16_json.dump(res, f)
        print(f"[V16] RUN model={M} arch={arch} head={res['head']} seed={S} params={res['params']} source={source} "
              f"encoding={encoding} selfcheck={res['selfcheck']} (acc {acc:.4f} vs result JSON {jacc})", flush=True)
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        print(f"[V16] RUN FAILED: {type(e).__name__}: {e}", flush=True)
# ═══ end V16A_HEAD_SAVEZ ════════════════════════════════════════════════════════

'''


def die(msg):
    sys.exit(f"PATCH REFUSED (nothing written): {msg}")


def main():
    if len(sys.argv) != 2:
        die("usage: patch_v16a.py <repo root>")
    root = Path(sys.argv[1]).resolve()
    tp = root / "train_gadi.py"
    for p in (tp, root / "qnn_torch.py", root / "quanv_torch.py", root / "closed_form_zz_v16.py", root / "v16_tools.py"):
        if not p.exists():
            die(f"{p} not found")
    t = tp.read_text()
    if MARK in t:
        die("train_gadi.py is already v16a-patched (marker found). Start from a clean copy.")
    for m, what in (("V135_M5_PATCH", "v13.5 M5 patch"), ("V14_SPLIT_AUDIT", "v14 split audit"),
                    ("V15_FE_NOISE", "v15 noise block")):
        if m not in t:
            die(f"train_gadi.py lacks the {what} ({m}). Build from qcq_cnn_v15.")
    q = (root / "qnn_torch.py").read_text()
    if "V13_READOUT_PATCH" not in q or "class QNNBackendTorch" not in q:
        die("qnn_torch.py is not the v13-patched file with class QNNBackendTorch.")
    mains = [m.start() for m in re.finditer(r"^if __name__ == [\"']__main__[\"']:", t, re.M)]
    if len(mains) != 1:
        die(f"expected exactly one top-level 'if __name__ == \"__main__\":', found {len(mains)}.")
    main_line = t[:mains[0]].count("\n") + 1
    lines = t.split("\n")
    for fn, required in (("build_model", True), ("load_dataset", True), ("expected_params", False)):
        defs = [i + 1 for i, l in enumerate(lines) if re.match(rf"def {fn}\(", l)]
        if not defs:
            if required:
                die(f"no top-level 'def {fn}('.")
            print(f"  {fn}: not defined at top level -- not wrapped (the verify dry run decides if that matters)")
            continue
        last = defs[-1]
        if last > main_line:
            die(f"'def {fn}(' at line {last} is after __main__ (line {main_line}).")
        bypass = []
        for i in range(last, main_line - 1):
            l = lines[i]
            if l[:1] in (" ", "\t", "#") or not l.strip():
                continue
            if re.search(rf"(?<![\w.]){fn}\b(?!\s*\()", l) and not re.match(rf"_v1\d\w*_{fn}_orig\s*=\s*{fn}\s*(#.*)?$", l) \
                    and not re.match(rf"_v1\d\w*_orig\s*=\s*{fn}\s*(#.*)?$", l):
                bypass.append(f"{i + 1}: {l.strip()}")
        if bypass:
            die(f"'{fn}' is captured by reference after its last definition (would bypass the v16a wrap): {bypass}")
        calls = len(re.findall(rf"(?<![\w.]){fn}\(", t)) - len(defs)
        print(f"  {fn}: {len(defs)} top-level definition(s) at line(s) {defs} (effective = line {last}); "
              f"{calls} call(s) by name")
    if re.search(r"from\s+train_gadi\s+import|import\s+train_gadi", t):
        die("train_gadi.py imports itself; the by-name wrap would not cover that path.")
    for flag in ("--save_z", "--v16_head", "--v16_sidecar"):
        if f'"{flag}"' in t or f"'{flag}'" in t:
            die(f"train_gadi.py already defines {flag}.")
    t2 = t[:mains[0]] + BLOCK.lstrip("\n") + "\n" + t[mains[0]:]
    bak = tp.with_name(tp.name + ".v15bak")
    if not bak.exists():
        shutil.copy2(tp, bak)
    tp.write_text(t2)
    print(f"Patched {tp.name} in {root}  (backup {bak.name}; block inserted above line {main_line})")
    print("Next: python3 -u scripts/verify_v16a.py   (must end in ALL CHECKS PASSED)")


if __name__ == "__main__":
    main()
