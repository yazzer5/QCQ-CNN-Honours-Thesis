#!/usr/bin/env python3
"""
v16b-ext patch (PREREGISTRATION_v16b.md amendment A3): M4-trim and BFR-ZZ, on top of the v16b-patched train_gadi.py.

    python3 patch_v16bx.py /g/data/wa66/yh2481/qcq_cnn_v16bx      (setup_v16bx.sh does this)

Inserts ONE block above `if __name__ == "__main__":` (after the V16B block). No existing line is edited.
New flag, removed from sys.argv before argparse runs:

  --v16bx_head m4trim   only with --model M4: fc_mid trimmed 15 -> w(K) exactly as BFR, no quantum branch.
  --v16bx_head bfrzz    only with --model M4: BFR with the ZZ product phase restored in the encoding.

Both: built on M4 in a forked RNG (M4's split and batch order); expected_params(M4) returns the variant count while
active; the BFR-ZZ gradient probe runs at g = 1 (D1) and g is restored bit-exact; at exit a self-check against the
result JSON and a sidecar v16bx_run_<MODEL>_seed<S>.json (BFR-ZZ: g, s, theta at best and stop).
No flag -> pass-through (verify_v16bx.py: M1-M5, M2c and BFR bit-identical). Backup: train_gadi.py.v16bbak.
"""
import re
import shutil
import sys
from pathlib import Path

MARK = "V16BX_EXT_HEAD"

BLOCK = r'''
# ═══════════════════════════════════════════════════════════════════════════════
# V16BX_EXT_HEAD  (inserted by patch_v16bx.py -- v16b-ext controls M4-trim and BFR-ZZ, amendment A3)
# Active only with --v16bx_head m4trim|bfrzz (removed from sys.argv here). Wraps build_model / expected_params /
# load_dataset / assert_gradient_flow by name; edits no existing code.
# ═══════════════════════════════════════════════════════════════════════════════
import atexit as _v16bx_atexit
import glob as _v16bx_glob
import json as _v16bx_json
import os as _v16bx_os
import sys as _v16bx_sys
import time as _v16bx_time

_V16BX = dict(head=None, in_build=False, model=None, seed=None, test=None, info=None, registered=False,
              stop=None, n_loads=0, t0=_v16bx_time.time(), argv=list(_v16bx_sys.argv))
_V16BX_LABEL = {"m4trim": "M4trim", "bfrzz": "BFRZZ"}


def _v16bx_strip_argv():
    out, i, av = [], 0, _v16bx_sys.argv
    while i < len(av):
        t = av[i]
        if t == "--v16bx_head":
            _V16BX["head"] = av[i + 1] if i + 1 < len(av) else ""
            i += 1
        elif t.startswith("--v16bx_head="):
            _V16BX["head"] = t.split("=", 1)[1]
        else:
            out.append(t)
        i += 1
    _v16bx_sys.argv[:] = out
    if _V16BX["head"] not in (None, "m4trim", "bfrzz"):
        _v16bx_sys.exit(f"[V16BX] FATAL: --v16bx_head {_V16BX['head']!r} (only m4trim, bfrzz)")
    if _V16BX["head"] is not None and _V16B["head"] is not None:  # noqa: F821
        _v16bx_sys.exit("[V16BX] FATAL: --v16bx_head and --v16b_head together")


_v16bx_strip_argv()


def _v16bx_args():
    import argparse as _ap
    p = _ap.ArgumentParser(add_help=False)
    for a in ("--model", "--output_dir", "--checkpoint_dir", "--classes"):
        p.add_argument(a, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_known_args(_V16BX["argv"][1:])[0]


_v16bx_build_model_orig = build_model  # noqa: F821


def build_model(*args, **kwargs):  # noqa: F811
    _V16BX["in_build"] = True
    try:
        model = _v16bx_build_model_orig(*args, **kwargs)
    finally:
        _V16BX["in_build"] = False
    kind = _V16BX["head"]
    if kind is None:
        return model
    name = args[0] if args else kwargs.get("model_name", kwargs.get("name", kwargs.get("model")))
    seed = kwargs.get("seed", args[1] if len(args) > 1 else None)
    if name != "M4":
        _v16bx_sys.exit(f"[V16BX] FATAL: --v16bx_head {kind} needs --model M4 (got {name})")
    import bfr_ext_v16b as _bx
    import bfr_head_v16b as _bh
    try:
        info = _bx.install(model, seed if seed is not None else 0, kind, HEAD_VARIANT)  # noqa: F821
    except _bx.ExtError as e:
        _v16bx_sys.exit(f"[V16BX] FATAL: surgery refused: {e}")
    _V16BX.update(model=model, seed=seed, info=info)
    q = _bh.bfr_module(model)
    qs = "" if q is None else (f" s0={float(q.s):.6f} g0={float(q.g):.1f} theta0={q.theta.detach().flatten().tolist()} "
                               f"wq_std={float(q.W_q.detach().std()):.4f} zeta_scale={q.zeta_scale:g}")
    print(f"[V16BX] HEAD {kind} installed model=M4 seed={seed} K={info['K']} fc_mid_width={info['width']} "
          f"params_M4={info['params_m4']} params={info['params']} expected={info['expected']}{qs} "
          f"{'VERIFIED' if info['ok'] else 'MISMATCH'}", flush=True)
    if not info["ok"]:
        _v16bx_sys.exit("[V16BX] FATAL: parameter count is not the pre-registered one")
    if q is not None:
        _orig_lsd = model.load_state_dict

        def _lsd(sd, *a, **k):
            st = _bh.bfr_state(model)
            if st is not None:
                _V16BX["stop"] = st
                _V16BX["n_loads"] += 1
            return _orig_lsd(sd, *a, **k)

        model.load_state_dict = _lsd
    if not _V16BX["registered"]:
        _v16bx_atexit.register(_v16bx_finish)
        _V16BX["registered"] = True
    return model


_v16bx_ep_orig = expected_params  # noqa: F821


def expected_params(*args, **kwargs):  # noqa: F811
    n = _v16bx_ep_orig(*args, **kwargs)
    if _V16BX["head"] is not None and not _V16BX["in_build"]:
        name = args[0] if args else kwargs.get("model_name", kwargs.get("name", kwargs.get("model")))
        if name == "M4":
            import bfr_ext_v16b as _bx
            ints = [v for v in list(args[1:]) + list(kwargs.values()) if isinstance(v, int) and not isinstance(v, bool)]
            n = _bx.expected_count(_V16BX["head"], ints[0] if ints else N_CLASSES)  # noqa: F821
    return n


_v16bx_agf_orig = assert_gradient_flow  # noqa: F821


def assert_gradient_flow(model, *args, **kwargs):  # noqa: F811
    import bfr_head_v16b as _bh
    q = _bh.bfr_module(model) if _V16BX["head"] == "bfrzz" else None
    if q is None:
        return _v16bx_agf_orig(model, *args, **kwargs)
    import torch as _t
    g0 = q.g.detach().clone()
    with _t.no_grad():
        q.g.fill_(1.0)
    print("[V16BX] GRADCHECK probe with g=1 (D1). Same batch draw as M4's probe.", flush=True)
    try:
        _v16bx_agf_orig(model, *args, **kwargs)
    finally:
        with _t.no_grad():
            q.g.copy_(g0)
    print(f"[V16BX] GRADCHECK g restored to {float(q.g):.1f} (bit-exact: {bool(_t.equal(q.g.detach(), g0))})", flush=True)


_v16bx_load_dataset_orig = load_dataset  # noqa: F821


def load_dataset(*args, **kwargs):  # noqa: F811
    res = _v16bx_load_dataset_orig(*args, **kwargs)
    if _V16BX["head"] is not None and isinstance(res, (tuple, list)) and len(res) >= 3:
        _V16BX["test"] = res[2]
    return res


def _v16bx_finish():
    import bfr_head_v16b as _bh
    import v16_tools as _vt
    try:
        a = _v16bx_args()
        model, ds, kind = _V16BX["model"], _V16BX["test"], _V16BX["head"]
        M = _V16BX_LABEL[kind]
        S = a.seed if a.seed is not None else _V16BX["seed"]
        if model is None or ds is None or a.output_dir is None:
            print("[V16BX] RUN skipped: model, test split or --output_dir not captured", flush=True)
            return
        cands = [f for f in _v16bx_glob.glob(_v16bx_os.path.join(a.output_dir, f"result_M4_*seed{S}_*.json"))
                 if _v16bx_os.path.getmtime(f) >= _V16BX["t0"] - 1]
        if not cands:
            print(f"[V16BX] RUN skipped: no result JSON written by this run in {a.output_dir}", flush=True)
            return
        jf = max(cands, key=_v16bx_os.path.getmtime)
        j = _v16bx_json.load(open(jf))
        st = _vt.rng_save()
        preds, labels, _ = _vt.predict_with_z(model, ds, want_z=False)
        _vt.rng_restore(st)
        saved = j.get("test_predictions")
        jacc = float(j.get("test_acc", float("nan")))
        acc = sum(int(p == y) for p, y in zip(preds, labels)) / len(labels)
        match = saved is not None and preds == [int(p) for p in saved] and abs(acc - jacc) < 1e-6
        best = _bh.bfr_state(model)
        res = dict(model=M, arch="M4", head=kind, seed=S, result_json=_v16bx_os.path.basename(jf),
                   selfcheck="MATCH" if match else "MISMATCH", n_test=len(labels), acc=acc, test_acc_json=jacc,
                   params=_vt.n_params(model), info=_V16BX["info"], best=best, stop=_V16BX["stop"],
                   n_state_loads=_V16BX["n_loads"], best_epoch=j.get("best_epoch"), stop_epoch=j.get("stop_epoch"),
                   labels=labels)
        with open(_v16bx_os.path.join(a.output_dir, f"v16bx_run_{M}_seed{S}.json"), "w") as f:
            _v16bx_json.dump(res, f)
        if best is not None:
            fmt = lambda d: "none" if d is None else f"g={d['g']:+.6f} s={d['s']:.6f} theta={[round(t, 6) for t in d['theta']]}"  # noqa: E731
            print(f"[V16BX] GS best(epoch {j.get('best_epoch')}): {fmt(best)} | stop(epoch {j.get('stop_epoch')}): "
                  f"{fmt(_V16BX['stop'])}", flush=True)
        print(f"[V16BX] RUN model={M} arch=M4 head={kind} seed={S} params={res['params']} selfcheck={res['selfcheck']} "
              f"(acc {acc:.4f} vs result JSON {jacc})", flush=True)
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        print(f"[V16BX] RUN FAILED: {type(e).__name__}: {e}", flush=True)
# ═══ end V16BX_EXT_HEAD ═════════════════════════════════════════════════════════

'''


def die(msg):
    sys.exit(f"PATCH REFUSED (nothing written): {msg}")


def main():
    if len(sys.argv) != 2:
        die("usage: patch_v16bx.py <repo root>")
    root = Path(sys.argv[1]).resolve()
    tp = root / "train_gadi.py"
    for p in (tp, root / "bfr_head_v16b.py", root / "bfr_ext_v16b.py", root / "v16_tools.py", root / "closed_form_zz_v16.py"):
        if not p.exists():
            die(f"{p} not found")
    t = tp.read_text()
    if MARK in t:
        die("train_gadi.py is already v16bx-patched. Start from a clean copy.")
    for m in ("V135_M5_PATCH", "V14_SPLIT_AUDIT", "V15_FE_NOISE", "V16A_HEAD_SAVEZ", "V16B_BFR_HEAD"):
        if m not in t:
            die(f"train_gadi.py lacks {m}. Build from qcq_cnn_v16b.")
    mains = [m.start() for m in re.finditer(r"^if __name__ == [\"']__main__[\"']:", t, re.M)]
    if len(mains) != 1 or t.find("V16B_BFR_HEAD") > mains[0]:
        die("expected one __main__ guard below the V16B block")
    main_line = t[:mains[0]].count("\n") + 1
    lines = t.split("\n")
    for fn in ("build_model", "load_dataset", "expected_params", "assert_gradient_flow"):
        defs = [i + 1 for i, l in enumerate(lines) if re.match(rf"def {fn}\(", l)]
        if not defs or defs[-1] > main_line:
            die(f"no top-level 'def {fn}(' above __main__")
        bypass = []
        for i in range(defs[-1], main_line - 1):
            l = lines[i]
            if l[:1] in (" ", "\t", "#") or not l.strip():
                continue
            code = re.sub(r"\"[^\"]*\"|'[^']*'", "", l)
            if re.search(rf"(?<![\w.]){fn}\b(?!\s*\()", code) and not re.match(rf"_v1\d\w*_orig\s*=\s*{fn}\s*(#.*)?$", l):
                bypass.append(f"{i + 1}: {l.strip()}")
        if bypass:
            die(f"'{fn}' captured by reference after its last definition: {bypass}")
        print(f"  {fn}: {len(defs)} top-level definition(s), effective = line {defs[-1]}")
    if '"--v16bx_head"' in t:
        die("train_gadi.py already defines --v16bx_head")
    t2 = t[:mains[0]] + BLOCK.lstrip("\n") + "\n" + t[mains[0]:]
    bak = tp.with_name(tp.name + ".v16bbak")
    if not bak.exists():
        shutil.copy2(tp, bak)
    tp.write_text(t2)
    print(f"Patched {tp.name} in {root}  (backup {bak.name}; block inserted above line {main_line})")


if __name__ == "__main__":
    main()
