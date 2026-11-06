#!/usr/bin/env python3
"""
v16b patch: the BFR head (Design 1, V16_PLAN.md A1), on top of the v16a-patched train_gadi.py.

    python3 patch_v16b.py /g/data/wa66/yh2481/qcq_cnn_v16b        (setup_v16b.sh does this)

Inserts ONE block into train_gadi.py, just above `if __name__ == "__main__":` (i.e. after the V16A block).
No existing line is edited. The block reads one new flag and removes it from sys.argv before
train_gadi.py's own argparse runs:

  --v16b_head bfr   only with --model M4. After M4 is built, its ClassicalHead is replaced by BFRHead
                    (bfr_head_v16b.install_bfr): M4's backbone and fc2 untouched, fc_mid trimmed 15 -> w(K),
                    gated quantum branch added. Every RNG state is restored, so the data split AND the
                    batch order are M4's at the same seed. expected_params(M4, ...) returns the BFR count
                    while the head is active (M4's count inside build_model, where M4 is what exists).
                    The trainer's gradient tripwire (assert_gradient_flow) is run with g temporarily = 1,
                    because at g = 0 the gated tensors (s, theta, W_q) get exactly zero gradient by design
                    (pre-registration D1); g is restored to exactly 0 and the probe draws the same batch as M4's.
                    At exit: per-image self-check against the result JSON (selfcheck=MATCH), g, s, theta at the
                    best checkpoint (the reported model) and at the stop epoch, sidecar v16b_run_BFR_seed<S>.json.

No flag -> the block is a pass-through (verify_v16b.py proves M1-M5 and M2c are bit-identical).
Backup: train_gadi.py.v16abak.
"""
import re
import shutil
import sys
from pathlib import Path

MARK = "V16B_BFR_HEAD"

BLOCK = r'''
# ═══════════════════════════════════════════════════════════════════════════════
# V16B_BFR_HEAD  (inserted by patch_v16b.py -- v16b Design 1, BFR head)
# Active only with --v16b_head bfr (removed from sys.argv here). Wraps build_model / expected_params /
# load_dataset / assert_gradient_flow by name; edits no existing code.
# ═══════════════════════════════════════════════════════════════════════════════
import atexit as _v16b_atexit
import glob as _v16b_glob
import json as _v16b_json
import os as _v16b_os
import sys as _v16b_sys
import time as _v16b_time

_V16B = dict(head=None, in_build=False, model=None, seed=None, test=None, info=None, registered=False,
             stop=None, n_loads=0, t0=_v16b_time.time(), argv=list(_v16b_sys.argv))


def _v16b_strip_argv():
    out, i, av = [], 0, _v16b_sys.argv
    while i < len(av):
        t = av[i]
        if t == "--v16b_head":
            _V16B["head"] = av[i + 1] if i + 1 < len(av) else ""
            i += 1
        elif t.startswith("--v16b_head="):
            _V16B["head"] = t.split("=", 1)[1]
        else:
            out.append(t)
        i += 1
    _v16b_sys.argv[:] = out
    if _V16B["head"] not in (None, "bfr"):
        _v16b_sys.exit(f"[V16B] FATAL: --v16b_head {_V16B['head']!r} (only 'bfr' exists)")


_v16b_strip_argv()


def _v16b_args():
    import argparse as _ap
    p = _ap.ArgumentParser(add_help=False)
    for a in ("--model", "--output_dir", "--checkpoint_dir", "--classes"):
        p.add_argument(a, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_known_args(_V16B["argv"][1:])[0]


_v16b_build_model_orig = build_model  # noqa: F821


def build_model(*args, **kwargs):  # noqa: F811
    _V16B["in_build"] = True
    try:
        model = _v16b_build_model_orig(*args, **kwargs)
    finally:
        _V16B["in_build"] = False
    if _V16B["head"] != "bfr":
        return model
    name = args[0] if args else kwargs.get("model_name", kwargs.get("name", kwargs.get("model")))
    seed = kwargs.get("seed", args[1] if len(args) > 1 else None)
    if name != "M4":
        _v16b_sys.exit(f"[V16B] FATAL: --v16b_head bfr needs --model M4 (got {name})")
    import bfr_head_v16b as _bh
    try:
        info = _bh.install_bfr(model, seed if seed is not None else 0, HEAD_VARIANT)  # noqa: F821
    except _bh.BFRError as e:
        _v16b_sys.exit(f"[V16B] FATAL: BFR surgery refused: {e}")
    _V16B.update(model=model, seed=seed, info=info)
    q = _bh.bfr_module(model)
    print(f"[V16B] HEAD bfr installed model=M4 seed={seed} K={info['K']} fc_mid_width={info['width']} "
          f"params_M4={info['params_m4']} params_BFR={info['params_bfr']} expected_BFR={info['expected_bfr']} "
          f"delta={info['delta']:+d} ({100 * info['rel']:+.2f}%) s0={float(q.s):.6f} g0={float(q.g):.1f} "
          f"theta0={q.theta.detach().flatten().tolist()} wq_std={float(q.W_q.detach().std()):.4f} "
          f"{'VERIFIED' if info['ok'] else 'MISMATCH'}", flush=True)
    if not info["ok"]:
        _v16b_sys.exit("[V16B] FATAL: BFR parameter count is not the pre-registered one")
    _orig_lsd = model.load_state_dict

    def _lsd(sd, *a, **k):   # record the state the run stopped in, before train_model restores the best one
        st = _bh.bfr_state(model)
        if st is not None:
            _V16B["stop"] = st
            _V16B["n_loads"] += 1
        return _orig_lsd(sd, *a, **k)

    model.load_state_dict = _lsd
    if not _V16B["registered"]:
        _v16b_atexit.register(_v16b_finish)
        _V16B["registered"] = True
    return model


if "expected_params" in globals():
    _v16b_ep_orig = expected_params  # noqa: F821

    def expected_params(*args, **kwargs):  # noqa: F811
        n = _v16b_ep_orig(*args, **kwargs)
        if _V16B["head"] == "bfr" and not _V16B["in_build"]:
            name = args[0] if args else kwargs.get("model_name", kwargs.get("name", kwargs.get("model")))
            if name == "M4":
                import bfr_head_v16b as _bh
                ints = [v for v in list(args[1:]) + list(kwargs.values())
                        if isinstance(v, int) and not isinstance(v, bool)]
                K = ints[0] if ints else N_CLASSES  # noqa: F821
                n = _bh.bfr_params(K, _bh.choose_width(K))
        return n


_v16b_agf_orig = assert_gradient_flow  # noqa: F821


def assert_gradient_flow(model, *args, **kwargs):  # noqa: F811
    import bfr_head_v16b as _bh
    q = _bh.bfr_module(model) if _V16B["head"] == "bfr" else None
    if q is None:
        return _v16b_agf_orig(model, *args, **kwargs)
    import torch as _t
    g0 = q.g.detach().clone()
    with _t.no_grad():
        q.g.fill_(1.0)
    print("[V16B] GRADCHECK probe with g=1 (at g=0 the gated tensors s, theta, W_q get zero gradient by "
          "design; D1). Same batch draw as M4's probe.", flush=True)
    try:
        _v16b_agf_orig(model, *args, **kwargs)
    finally:
        with _t.no_grad():
            q.g.copy_(g0)
    print(f"[V16B] GRADCHECK g restored to {float(q.g):.1f} (bit-exact: {bool(_t.equal(q.g.detach(), g0))})",
          flush=True)


_v16b_load_dataset_orig = load_dataset  # noqa: F821


def load_dataset(*args, **kwargs):  # noqa: F811
    res = _v16b_load_dataset_orig(*args, **kwargs)
    if _V16B["head"] == "bfr" and isinstance(res, (tuple, list)) and len(res) >= 3:
        _V16B["test"] = res[2]
    return res


def _v16b_saved_preds(j, n):
    for k in ("predictions", "test_predictions", "preds", "y_pred", "test_preds"):
        v = j.get(k)
        if isinstance(v, list) and len(v) == n:
            return k, [int(p) for p in v]
    return None, None


def _v16b_finish():
    import bfr_head_v16b as _bh
    import v16_tools as _vt
    try:
        a = _v16b_args()
        model, ds = _V16B["model"], _V16B["test"]
        S = a.seed if a.seed is not None else _V16B["seed"]
        stop = _V16B["stop"]
        if model is None or ds is None or a.output_dir is None:
            print("[V16B] RUN skipped: model, test split or --output_dir not captured", flush=True)
            return
        cands = [f for f in _v16b_glob.glob(_v16b_os.path.join(a.output_dir, f"result_M4_*seed{S}_*.json"))
                 if _v16b_os.path.getmtime(f) >= _V16B["t0"] - 1]
        if not cands:
            print(f"[V16B] RUN skipped: no result JSON written by this run in {a.output_dir}", flush=True)
            return
        jf = max(cands, key=_v16b_os.path.getmtime)
        j = _v16b_json.load(open(jf))
        st = _vt.rng_save()
        preds, labels, _ = _vt.predict_with_z(model, ds, want_z=False)
        _vt.rng_restore(st)
        key, saved = _v16b_saved_preds(j, len(preds))
        jacc = float(j.get("test_acc", float("nan")))
        acc = sum(int(p == y) for p, y in zip(preds, labels)) / len(labels)
        match = saved is not None and preds == saved and (abs(acc - jacc) < 1e-6 or abs(100 * acc - jacc) < 1e-4)
        best = _bh.bfr_state(model)
        res = dict(model="BFR", arch="M4", head="bfr", seed=S, result_json=_v16b_os.path.basename(jf),
                   predictions_key=key, selfcheck="MATCH" if match else "MISMATCH", n_test=len(labels), acc=acc,
                   test_acc_json=jacc, params=_vt.n_params(model), bfr=_V16B["info"], best=best, stop=stop,
                   n_state_loads=_V16B["n_loads"], best_epoch=j.get("best_epoch"), stop_epoch=j.get("stop_epoch"),
                   labels=labels)
        with open(_v16b_os.path.join(a.output_dir, f"v16b_run_BFR_seed{S}.json"), "w") as f:
            _v16b_json.dump(res, f)
        fmt = lambda d: "none" if d is None else f"g={d['g']:+.6f} s={d['s']:.6f} theta={[round(t, 6) for t in d['theta']]}"  # noqa: E731
        print(f"[V16B] GS best(epoch {j.get('best_epoch')}): {fmt(best)} | stop(epoch {j.get('stop_epoch')}): {fmt(stop)}",
              flush=True)
        print(f"[V16B] RUN model=BFR arch=M4 head=bfr seed={S} params={res['params']} selfcheck={res['selfcheck']} "
              f"(acc {acc:.4f} vs result JSON {jacc})", flush=True)
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        print(f"[V16B] RUN FAILED: {type(e).__name__}: {e}", flush=True)
# ═══ end V16B_BFR_HEAD ══════════════════════════════════════════════════════════

'''


def die(msg):
    sys.exit(f"PATCH REFUSED (nothing written): {msg}")


def main():
    if len(sys.argv) != 2:
        die("usage: patch_v16b.py <repo root>")
    root = Path(sys.argv[1]).resolve()
    tp = root / "train_gadi.py"
    for p in (tp, root / "qnn_torch.py", root / "quanv_torch.py", root / "closed_form_zz_v16.py",
              root / "v16_tools.py", root / "bfr_head_v16b.py"):
        if not p.exists():
            die(f"{p} not found")
    t = tp.read_text()
    if MARK in t:
        die("train_gadi.py is already v16b-patched (marker found). Start from a clean copy.")
    for m, what in (("V135_M5_PATCH", "v13.5 M5 patch"), ("V14_SPLIT_AUDIT", "v14 split audit"),
                    ("V15_FE_NOISE", "v15 noise block"), ("V16A_HEAD_SAVEZ", "v16a block")):
        if m not in t:
            die(f"train_gadi.py lacks the {what} ({m}). Build from qcq_cnn_v16a.")
    mains = [m.start() for m in re.finditer(r"^if __name__ == [\"']__main__[\"']:", t, re.M)]
    if len(mains) != 1:
        die(f"expected exactly one top-level 'if __name__ == \"__main__\":', found {len(mains)}.")
    if t.find("V16A_HEAD_SAVEZ") > mains[0]:
        die("the V16A block is not above __main__")
    main_line = t[:mains[0]].count("\n") + 1
    lines = t.split("\n")
    for fn in ("build_model", "load_dataset", "expected_params", "assert_gradient_flow"):
        defs = [i + 1 for i, l in enumerate(lines) if re.match(rf"def {fn}\(", l)]
        if not defs:
            die(f"no top-level 'def {fn}('.")
        if defs[-1] > main_line:
            die(f"'def {fn}(' at line {defs[-1]} is after __main__ (line {main_line}).")
        bypass = []
        for i in range(defs[-1], main_line - 1):
            l = lines[i]
            if l[:1] in (" ", "\t", "#") or not l.strip():
                continue
            code = re.sub(r"\"[^\"]*\"|'[^']*'", "", l)          # names inside string literals are not references
            if re.search(rf"(?<![\w.]){fn}\b(?!\s*\()", code) and not re.match(rf"_v1\d\w*_orig\s*=\s*{fn}\s*(#.*)?$", l):
                bypass.append(f"{i + 1}: {l.strip()}")
        if bypass:
            die(f"'{fn}' is captured by reference after its last definition (would bypass the wrap): {bypass}")
        print(f"  {fn}: {len(defs)} top-level definition(s), effective = line {defs[-1]}")
    # every call site of assert_gradient_flow must be a by-name call inside a function (looked up at call time)
    calls = [i + 1 for i, l in enumerate(lines) if re.search(r"(?<![\w.])assert_gradient_flow\(", l) and not l.startswith("def ")]
    if not calls or any(not lines[c - 1].startswith((" ", "\t")) for c in calls):
        die(f"assert_gradient_flow call sites {calls} are not all inside functions")
    print(f"  assert_gradient_flow: called by name at line(s) {calls}")
    if '"--v16b_head"' in t or "'--v16b_head'" in t:
        die("train_gadi.py already defines --v16b_head.")
    t2 = t[:mains[0]] + BLOCK.lstrip("\n") + "\n" + t[mains[0]:]
    bak = tp.with_name(tp.name + ".v16abak")
    if not bak.exists():
        shutil.copy2(tp, bak)
    tp.write_text(t2)
    print(f"Patched {tp.name} in {root}  (backup {bak.name}; block inserted above line {main_line})")
    print("Next: python3 -u scripts/verify_v16b.py   (must end in ALL CHECKS PASSED)")


if __name__ == "__main__":
    main()
