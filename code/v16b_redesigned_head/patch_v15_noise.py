#!/usr/bin/env python3
"""
v15 patch: IBM-calibrated noise on the quantum FRONT-END, set by environment.

    python3 patch_v15_noise.py /g/data/wa66/yh2481/qcq_cnn_v15     (setup_v15.sh does this)

Inserts one block into train_gadi.py, just above `if __name__ == "__main__":`.
No model, data or training line is edited. The block wraps two module-level
functions by name:

  build_model(...)  -> when V15_ALPHA is set, attaches quanv_noise_v15.FrontEndNoise
                       to the model's quantum front-end (M1: RandomLayers, M2/M3/M5:
                       Circuit 4; M4 has none) and sets it to alpha = V15_ALPHA.
                       At alpha = 0 the ORIGINAL forward runs (bit-exact).
                       Self-checks printed at install: the PTM path at alpha = 0 equals
                       the exact forward; M1's gate decomposition equals the stored
                       unitary. Every RNG state is saved and restored around it.
  load_dataset(...) -> returned untouched; the test split is kept for the
                       cross-evaluation below.

At process exit (V15_ALPHA set): CROSS-EVALUATION. The trained model is evaluated
on the test set at every alpha in V15_EVAL_ALPHAS. First it is evaluated at its own
training alpha and must reproduce the per-image predictions train_gadi.py saved in
its result JSON exactly (if the weights in memory are not the tested ones, the
checkpoints in --checkpoint_dir are tried). Only a MATCH is usable; the gate checks.
Writes <output_dir>/v15_xeval_<MODEL>_seed<SEED>.json.

Unset V15_ALPHA -> the block does nothing (verify_v15.py proves bit-identity).
Backup: train_gadi.py.v14bak.
"""
import re
import shutil
import sys
from pathlib import Path

MARK = "V15_FE_NOISE"

BLOCK = r'''
# ═══════════════════════════════════════════════════════════════════════════════
# V15_FE_NOISE  (inserted by patch_v15_noise.py -- v15 IBM-calibrated front-end noise)
# Active only when the environment variable V15_ALPHA is set (run15.pbs sets it).
# Wraps build_model / load_dataset by name; changes no model, data or training code.
# ═══════════════════════════════════════════════════════════════════════════════
import atexit as _v15_atexit
import glob as _v15_glob
import json as _v15_json
import os as _v15_os
import random as _v15_random
import sys as _v15_sys
import time as _v15_time

_V15 = dict(active=_v15_os.environ.get("V15_ALPHA") is not None, t0=_v15_time.time(),
            model=None, noise=None, test=None, name=None, seed=None, registered=False)


def _v15_rng_save():
    import numpy as _np
    import torch as _t
    st = dict(py=_v15_random.getstate(), np=_np.random.get_state(), t=_t.get_rng_state())
    if _t.cuda.is_available():
        st["cuda"] = _t.cuda.get_rng_state_all()
    return st


def _v15_rng_restore(st):
    import numpy as _np
    import torch as _t
    _v15_random.setstate(st["py"]); _np.random.set_state(st["np"]); _t.set_rng_state(st["t"])
    if "cuda" in st:
        _t.cuda.set_rng_state_all(st["cuda"])


def _v15_args():
    import argparse as _ap
    p = _ap.ArgumentParser(add_help=False)
    for a in ("--model", "--output_dir", "--checkpoint_dir", "--classes"):
        p.add_argument(a, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_known_args(_v15_sys.argv[1:])[0]


def _v15_install(model, name, seed):
    import torch as _t
    import quanv_noise_v15 as _qn
    from quanv_torch import TrainableQuanvFrontEndTorch as _C4, FrozenHendersonFrontEndTorch as _HE
    alpha = float(_v15_os.environ["V15_ALPHA"])
    calib = _qn.load_calib(_v15_os.environ["V15_CALIB"])
    fes = [m for m in model.modules() if isinstance(m, (_C4, _HE))]
    if name == "M4":
        if fes:
            _v15_sys.exit(f"[V15] FATAL: M4 has {len(fes)} quantum front-end(s); expected none")
        print(f"[V15] NOISE installed model=M4 kind=none alpha={alpha:g} calib={calib['sha256'][:12]} "
              f"(no quantum front-end: noise has nothing to act on)", flush=True)
        return None
    if len(fes) != 1:
        _v15_sys.exit(f"[V15] FATAL: model {name} has {len(fes)} quantum front-ends; expected exactly 1")
    fe = fes[0]
    kind = "henderson" if isinstance(fe, _HE) else "circuit4"
    st = _v15_rng_save()
    try:
        nz = _qn.FrontEndNoise(fe, kind, calib, seed=seed)
        if kind == "henderson":
            ok = nz.decomp_err < 1e-5
            print(f"[V15] NOISE selfcheck M1 RandomLayers decomposition == stored unitary: "
                  f"max|d|={nz.decomp_err:.2e} {'PASS' if ok else 'FAIL'}", flush=True)
            if not ok:
                _v15_sys.exit("[V15] FATAL: RandomLayers decomposition does not reproduce the front-end unitary")
        dev = next(iter(list(fe.parameters()) + list(fe.buffers()))).device
        g = _t.Generator().manual_seed(1234)
        x = _t.rand(4, 1, 28, 28, generator=g).to(dev)
        with _t.no_grad():
            d = (nz.ptm_path_at_zero(x) - nz._orig(x)).abs().max().item()
        ok = d < 1e-5
        print(f"[V15] NOISE selfcheck PTM path at alpha=0 vs exact forward: max|d|={d:.2e} "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
        if not ok:
            _v15_sys.exit("[V15] FATAL: noise path at alpha=0 does not reproduce the exact front-end")
        nz.set_alpha(alpha)
    finally:
        _v15_rng_restore(st)
    summ = nz.NP.summary() if nz.NP is not None else "noiseless (original forward)"
    print(f"[V15] NOISE installed model={name} kind={kind} alpha={alpha:g} calib={calib['sha256'][:12]} "
          f"snapshot={calib['last_update_date'][:10]} {summ}", flush=True)
    return nz


_v15_build_model_orig = build_model  # noqa: F821


def build_model(*args, **kwargs):  # noqa: F811
    model = _v15_build_model_orig(*args, **kwargs)
    if _V15["active"]:
        name = args[0] if args else kwargs.get("model_name", kwargs.get("name"))
        seed = kwargs.get("seed", args[1] if len(args) > 1 else None)
        if seed is None:
            seed = _v15_args().seed
        nz = _v15_install(model, name, seed)
        _V15.update(model=model, name=name, seed=seed, noise=nz)
        _V15.setdefault("built", []).append((model, nz))
        if not _V15["registered"]:
            _v15_atexit.register(_v15_xeval)
            _V15["registered"] = True
    return model


_v15_load_dataset_orig = load_dataset  # noqa: F821


def load_dataset(*args, **kwargs):  # noqa: F811
    res = _v15_load_dataset_orig(*args, **kwargs)
    if _V15["active"] and isinstance(res, (tuple, list)) and len(res) >= 3:
        _V15["test"] = res[2]
    return res


def _v15_predict(model, ds, bs=250):
    import torch as _t
    dev = next(model.parameters()).device
    was_training = model.training
    model.eval()
    preds, labels = [], []
    loader = _t.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
    with _t.no_grad():
        for batch in loader:
            x, y = batch[0], batch[1]
            out = model(x.to(dev))
            preds += out.argmax(1).cpu().tolist()
            labels += [int(v) for v in (y.tolist() if hasattr(y, "tolist") else y)]
    model.train(was_training)
    return preds, labels


def _v15_saved_preds(j, n):
    for k in ("predictions", "test_predictions", "preds", "y_pred", "test_preds"):
        v = j.get(k)
        if isinstance(v, list) and len(v) == n:
            if v and isinstance(v[0], list):
                return k, [max(range(len(r)), key=r.__getitem__) for r in v]
            return k, [int(p) for p in v]
    return None, None


def _v15_xeval():
    import torch as _t
    try:
        a = _v15_args()
        model, nz, ds = _V15["model"], _V15["noise"], _V15["test"]
        M, S = a.model or _V15["name"], a.seed if a.seed is not None else _V15["seed"]
        alpha_tr = float(_v15_os.environ["V15_ALPHA"])
        evals = [float(x) for x in _v15_os.environ.get("V15_EVAL_ALPHAS", "").split()]
        if model is None or ds is None or a.output_dir is None:
            print("[V15] XEVAL skipped: model, test split or --output_dir not captured", flush=True)
            return
        cands = [f for f in _v15_glob.glob(_v15_os.path.join(a.output_dir, f"result_{M}_*seed{S}_*.json"))
                 if _v15_os.path.getmtime(f) >= _V15["t0"] - 1]
        if not cands:
            print(f"[V15] XEVAL skipped: no result JSON written by this run in {a.output_dir}", flush=True)
            return
        jf = max(cands, key=_v15_os.path.getmtime)
        j = _v15_json.load(open(jf))
        st = _v15_rng_save()
        if nz is not None:
            nz.set_alpha(alpha_tr)
        preds, labels = _v15_predict(model, ds)
        key, saved = _v15_saved_preds(j, len(preds))
        classes = [int(c) for c in (a.classes or "").replace(" ", "").split(",") if c != ""]

        def same(p):
            if saved is None:
                return False
            if p == saved:
                return True
            return len(classes) > max(p) and [classes[i] for i in p] == saved

        source = "memory"
        match = same(preds)
        if not match:                       # train_gadi built more than one model: try each, latest first
            for k_, (m_, nz_) in enumerate(reversed(_V15.get("built", [])[:-1])):
                if nz_ is not None:
                    nz_.set_alpha(alpha_tr)
                p_, l_ = _v15_predict(m_, ds)
                if same(p_):
                    model, nz, preds, labels = m_, nz_, p_, l_
                    source, match = "memory(model %d)" % k_, True
                    break
        if not match and a.checkpoint_dir and _v15_os.path.isdir(a.checkpoint_dir):
            for ck in sorted(_v15_glob.glob(_v15_os.path.join(a.checkpoint_dir, "*.pt*"))):
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
                preds, labels = _v15_predict(model, ds)
                if same(preds):
                    source, match = "ckpt:" + _v15_os.path.basename(ck), True
                    break
        acc_tr = sum(int(p == y) for p, y in zip(preds, labels)) / len(labels)
        jacc = float(j.get("test_acc", float("nan")))
        acc_ok = abs(acc_tr - jacc) < 1e-6 or abs(100 * acc_tr - jacc) < 1e-4   # result JSON may hold float32
        res = dict(model=M, seed=S, alpha_train=alpha_tr, eval_alphas=evals,
                   calib_sha=_v15_json.load(open(_v15_os.environ["V15_CALIB"]))["sha256"],
                   result_json=_v15_os.path.basename(jf), predictions_key=key, source=source,
                   selfcheck="MATCH" if (match and acc_ok) else "MISMATCH", n_test=len(labels),
                   labels=labels, acc={}, predictions={})
        for al in evals:
            if nz is not None:
                nz.set_alpha(al)
            p, _ = _v15_predict(model, ds) if (nz is not None or al == evals[0]) else (res["predictions"][f"{evals[0]:g}"], None)
            res["predictions"][f"{al:g}"] = p
            res["acc"][f"{al:g}"] = sum(int(q == y) for q, y in zip(p, labels)) / len(labels)
        if nz is not None:
            nz.set_alpha(alpha_tr)
        _v15_rng_restore(st)
        out = _v15_os.path.join(a.output_dir, f"v15_xeval_{M}_seed{S}.json")
        with open(out, "w") as f:
            _v15_json.dump(res, f)
        accs = " ".join(f"{k}={v:.4f}" for k, v in res["acc"].items())
        print(f"[V15] XEVAL model={M} seed={S} alpha_train={alpha_tr:g} source={source} "
              f"selfcheck={res['selfcheck']} (acc at alpha_train {acc_tr:.4f} vs result JSON {jacc}) "
              f"acc_by_eval_alpha: {accs}", flush=True)
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        print(f"[V15] XEVAL FAILED: {type(e).__name__}: {e}", flush=True)
# ═══ end V15_FE_NOISE ═══════════════════════════════════════════════════════════

'''


def die(msg):
    sys.exit(f"PATCH REFUSED: {msg}")


def main():
    if len(sys.argv) != 2:
        die("usage: patch_v15_noise.py <repo root>")
    root = Path(sys.argv[1]).resolve()
    tp, qt = root / "train_gadi.py", root / "quanv_torch.py"
    for p in (tp, qt, root / "qnn_torch.py", root / "quanv_noise_v15.py"):
        if not p.exists():
            die(f"{p} not found")
    t = tp.read_text()
    if MARK in t:
        die("train_gadi.py is already v15-patched (marker found). Start from a clean copy.")
    for m, what in (("V14_SPLIT_AUDIT", "v14 split audit"), ("V135_M5_PATCH", "v13.5 M5 patch")):
        if m not in t:
            die(f"train_gadi.py lacks the {what} ({m}). Build from qcq_cnn_v14.")
    if "V13_READOUT_PATCH" not in (root / "qnn_torch.py").read_text():
        die("qnn_torch.py is not the v13-patched file.")
    q = qt.read_text()
    for cls in ("class TrainableQuanvFrontEndTorch", "class FrozenHendersonFrontEndTorch", "def _block_unitary"):
        if cls not in q:
            die(f"quanv_torch.py: '{cls}' not found")
    mains = [m.start() for m in re.finditer(r"^if __name__ == [\"']__main__[\"']:", t, re.M)]
    if len(mains) != 1:
        die(f"train_gadi.py: expected exactly one top-level 'if __name__ == \"__main__\":', found {len(mains)}.")
    main_line = t[:mains[0]].count("\n") + 1
    lines = t.split("\n")
    for fn in ("build_model", "load_dataset"):
        # Earlier patches (v13.5, v14) may have re-defined these names as wrappers of the
        # previous definition. Python binds the module-global name to the LAST top-level
        # definition, so the v15 block (inserted after all of them, just above __main__)
        # wraps the effective function. What must not exist: a top-level reference that
        # captures the name AFTER the last definition (it would bypass the v15 wrap).
        defs = [i + 1 for i, l in enumerate(lines) if re.match(rf"def {fn}\(", l)]
        if not defs:
            die(f"train_gadi.py: no top-level 'def {fn}('.")
        last = defs[-1]
        if last > main_line:
            die(f"train_gadi.py: 'def {fn}(' at line {last} is after __main__ (line {main_line}).")
        bypass = []
        for i in range(last, main_line - 1):          # top-level lines after the last def, before __main__
            l = lines[i]
            if l[:1] in (" ", "\t", "#") or not l.strip():
                continue
            if re.search(rf"(?<![\w.]){fn}\b(?!\s*\()", l):
                bypass.append(f"{i + 1}: {l.strip()}")
        if bypass:
            die(f"train_gadi.py: '{fn}' is captured by reference after its last definition "
                f"(would bypass the v15 wrap): {bypass}")
        calls = len(re.findall(rf"(?<![\w.]){fn}\(", t)) - len(defs)
        if calls < 1:
            die(f"train_gadi.py: {fn} is defined but never called by its module-global name.")
        print(f"  {fn}: {len(defs)} top-level definition(s) at line(s) {defs} "
              f"(effective = line {last}); {calls} call site(s) by name; no capture after line {last}")
        for d in defs[:-1]:
            print(f"    line {d} is wrapped by a later definition (an earlier patch); the v15 wrap goes on top")
    if re.search(r"from\s+train_gadi\s+import|import\s+train_gadi", t):
        die("train_gadi.py imports itself; the by-name wrap would not cover that path.")

    g = re.search(r"^if __name__ == [\"']__main__[\"']:", t, re.M)
    t = t[:g.start()] + BLOCK.lstrip("\n") + "\n" + t[g.start():]
    bak = tp.with_name(tp.name + ".v14bak")
    if not bak.exists():
        shutil.copy2(tp, bak)
    tp.write_text(t)
    print(f"Patched {tp.name} in {root}  (backup {bak.name})")
    print("Next: python3 scripts/verify_v15.py   (must end in ALL CHECKS PASSED)")


if __name__ == "__main__":
    main()
