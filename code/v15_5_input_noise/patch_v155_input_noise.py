#!/usr/bin/env python3
"""
v15.5 patch: Gaussian PIXEL noise on the input images, set by environment.

    python3 patch_v155_input_noise.py /g/data/wa66/yh2481/qcq_cnn_v155     (setup_v155.sh does this)

Inserts one block into train_gadi.py, just above `if __name__ == "__main__":`.
No model, data, split or training line is edited. The block wraps two module-level
functions by name (their LAST top-level definitions, i.e. the v13.5 / v14 wrappers):

  load_dataset(...) -> calls the effective load_dataset (the v14 split audit runs inside
                       it, on the CLEAN splits), then, when V155_SIGMA > 0, wraps the three
                       datasets it returns:
                         train: x -> clip(x + sigma*eps, 0, 1), eps fresh on every fetch,
                                from a private torch.Generator seeded by (seed, sigma);
                                the global RNGs (python, numpy, torch, cuda) are never used.
                         val:   fixed noise field per sigma (fixed seed, same for every
                                model and seed).
                         test:  fixed noise field per sigma (fixed seed, same for every
                                model and seed) -> identical noisy test images everywhere.
                       V155_SIGMA = 0 -> the datasets are returned untouched (bit-exact v14).
  build_model(...)  -> returned untouched; a reference to the model is kept for the
                       cross-evaluation.

At process exit (V155_SIGMA set): CROSS-EVALUATION. The trained model is evaluated on
the test set at its own sigma and must reproduce the per-image predictions
train_gadi.py saved in its result JSON exactly (MATCH). It is then evaluated at every
sigma in V155_EVAL_SIGMAS (Gaussian) and every p in V155_EVAL_SP (salt-and-pepper,
exploratory). Writes <output_dir>/v155_xeval_<MODEL>_seed<SEED>.json.

Unset V155_SIGMA -> the block does nothing (verify_v155.py proves bit-identity).
Backup: train_gadi.py.v14bak.
"""
import re
import shutil
import sys
from pathlib import Path

MARK = "V155_INPUT_NOISE"

BLOCK = r'''
# ═══════════════════════════════════════════════════════════════════════════════
# V155_INPUT_NOISE  (inserted by patch_v155_input_noise.py -- v15.5 pixel noise on the input)
# Active only when the environment variable V155_SIGMA is set (run155.pbs sets it).
# Wraps load_dataset / build_model by name; changes no model, data or training code.
# ═══════════════════════════════════════════════════════════════════════════════
import atexit as _v155_atexit
import glob as _v155_glob
import hashlib as _v155_hashlib
import json as _v155_json
import os as _v155_os
import random as _v155_random
import sys as _v155_sys
import time as _v155_time

V155_TEST_GAUSS_BASE = 15500000     # test noise seed  = base + round(1000*sigma)
V155_VAL_GAUSS_BASE = 15600000      # val noise seed   = base + round(1000*sigma)
V155_TEST_SP_BASE = 15700000        # salt-and-pepper  = base + round(1000*p)
V155_TRAIN_BASE = 155000000         # train generator  = base + 10000*seed + round(1000*sigma)

_V155 = dict(t0=_v155_time.time(), model=None, models=[], test=None, train=None, seed=None,
             registered=False)


def _v155_code(level):
    return int(round(1000.0 * float(level)))


def _v155_sigma():
    v = _v155_os.environ.get("V155_SIGMA")
    return None if v is None or v.strip() == "" else float(v)


def _v155_train_seed(seed, sigma):
    return V155_TRAIN_BASE + 10000 * int(seed) + _v155_code(sigma)


def _v155_rng_save():
    import numpy as _np
    import torch as _t
    st = dict(py=_v155_random.getstate(), np=_np.random.get_state(), t=_t.get_rng_state())
    if _t.cuda.is_available():
        st["cuda"] = _t.cuda.get_rng_state_all()
    return st


def _v155_rng_restore(st):
    import numpy as _np
    import torch as _t
    _v155_random.setstate(st["py"]); _np.random.set_state(st["np"]); _t.set_rng_state(st["t"])
    if "cuda" in st:
        _t.cuda.set_rng_state_all(st["cuda"])


def _v155_rng_equal(a, b):
    import numpy as _np
    import torch as _t
    ok = a["py"] == b["py"] and _t.equal(a["t"], b["t"])
    ok = ok and all(_np.array_equal(x, y) if hasattr(x, "shape") else x == y for x, y in zip(a["np"], b["np"]))
    if "cuda" in a or "cuda" in b:
        ok = ok and len(a.get("cuda", [])) == len(b.get("cuda", [])) and \
            all(_t.equal(x, y) for x, y in zip(a.get("cuda", []), b.get("cuda", [])))
    return ok


def _v155_args():
    import argparse as _ap
    p = _ap.ArgumentParser(add_help=False)
    for a in ("--model", "--output_dir", "--checkpoint_dir", "--classes"):
        p.add_argument(a, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_known_args(_v155_sys.argv[1:])[0]


def _v155_stack(ds):
    """All images of a dataset as one float tensor (n,1,28,28), plus the label objects as returned."""
    import torch as _t
    xs, ys = [], []
    for i in range(len(ds)):
        x, y = ds[i]
        xs.append(x)
        ys.append(y)
    return _t.stack(xs), ys


def _v155_hash(X, ys):
    import numpy as _np
    h = _v155_hashlib.sha256()
    h.update(X.detach().cpu().contiguous().numpy().astype("<f4").tobytes())
    h.update(_np.asarray([int(y) for y in ys], dtype="<i8").tobytes())
    return h.hexdigest()


def _v155_fixed(X, kind, level, base):
    """Deterministic corruption of a whole image tensor. Never touches the global RNGs."""
    import torch as _t
    level = float(level)
    if level == 0:
        return X.clone()
    g = _t.Generator().manual_seed(base + _v155_code(level))
    if kind == "gauss":
        return (X + level * _t.randn(X.shape, generator=g, dtype=X.dtype)).clamp(0.0, 1.0)
    if kind == "sp":
        u = _t.rand(X.shape, generator=g, dtype=X.dtype)
        Y = X.clone()
        Y[u < level / 2.0] = 0.0
        Y[u > 1.0 - level / 2.0] = 1.0
        return Y
    raise ValueError("unknown corruption " + kind)


class _V155FixedDS(Dataset):  # noqa: F821  (train_gadi imports torch Dataset)
    """Fixed (pre-computed) noisy images; labels exactly as the clean dataset returns them."""

    def __init__(self, X, ys, base):
        self.X, self.ys, self._base = X, ys, base

    def __len__(self):
        return len(self.ys)

    def __getitem__(self, i):
        return self.X[i], self.ys[i]

    def __getattr__(self, k):
        if k.startswith("__") or k in ("X", "ys", "_base"):
            raise AttributeError(k)
        return getattr(self._base, k)


class _V155TrainDS(Dataset):  # noqa: F821
    """Fresh Gaussian noise on every fetch, from a private generator (global RNGs untouched)."""

    def __init__(self, base, sigma, gen_seed):
        import torch as _t
        self._base, self.sigma, self.gen_seed = base, float(sigma), int(gen_seed)
        self.g = _t.Generator().manual_seed(self.gen_seed)
        self.n_fetch = 0

    def __len__(self):
        return len(self._base)

    def __getitem__(self, i):
        import torch as _t
        if _t.utils.data.get_worker_info() is not None:
            raise RuntimeError("[V155] FATAL: training noise drawn inside a DataLoader worker "
                               "(num_workers > 0 would duplicate the generator)")
        x, y = self._base[i]
        self.n_fetch += 1
        n = _t.randn(x.shape, generator=self.g, dtype=x.dtype)
        return (x + self.sigma * n).clamp(0.0, 1.0), y

    def __getattr__(self, k):
        if k.startswith("__") or k in ("_base", "sigma", "gen_seed", "g", "n_fetch"):
            raise AttributeError(k)
        return getattr(self._base, k)


_v155_build_model_orig = build_model  # noqa: F821


def build_model(*args, **kwargs):  # noqa: F811
    model = _v155_build_model_orig(*args, **kwargs)
    if _v155_sigma() is not None:
        _V155["model"] = model
        _V155["models"].append(model)
    return model


_v155_load_dataset_orig = load_dataset  # noqa: F821


def load_dataset(*args, **kwargs):  # noqa: F811
    res = _v155_load_dataset_orig(*args, **kwargs)
    sigma = _v155_sigma()
    if sigma is None:
        return res
    if not (isinstance(res, (tuple, list)) and len(res) == 3):
        _v155_sys.exit("[V155] FATAL: load_dataset did not return (train, val, test)")
    if sigma < 0 or sigma > 1:
        _v155_sys.exit("[V155] FATAL: sigma=%g outside [0, 1]" % sigma)
    a = _v155_args()
    seed = kwargs.get("seed", args[2] if len(args) > 2 else a.seed)
    if seed is None:
        _v155_sys.exit("[V155] FATAL: could not determine the seed")
    tr, va, te = res
    st0 = _v155_rng_save()
    Xte, yte = _v155_stack(te)
    name = a.model or "?"
    if sigma == 0:
        out = res
        print("[V155] NOISE sigma=0 model=%s seed=%d datasets untouched (clean, bit-exact v14)" % (name, seed),
              flush=True)
        print("[V155] TESTSET sigma=0 n=%d sha=%s" % (len(yte), _v155_hash(Xte, yte)), flush=True)
    else:
        Xte_n = _v155_fixed(Xte, "gauss", sigma, V155_TEST_GAUSS_BASE)
        Xva, yva = _v155_stack(va)
        Xva_n = _v155_fixed(Xva, "gauss", sigma, V155_VAL_GAUSS_BASE)
        gseed = _v155_train_seed(seed, sigma)
        tr_n = _V155TrainDS(tr, sigma, gseed)
        out = (tr_n, _V155FixedDS(Xva_n, yva, va), _V155FixedDS(Xte_n, yte, te))
        _V155["train"] = tr_n
        print("[V155] NOISE sigma=%g model=%s seed=%d kind=gaussian clip=[0,1] train=fresh-per-fetch "
              "train_gen_seed=%d val_seed=%d test_seed=%d" % (
                  sigma, name, seed, gseed, V155_VAL_GAUSS_BASE + _v155_code(sigma),
                  V155_TEST_GAUSS_BASE + _v155_code(sigma)), flush=True)
        print("[V155] TESTSET sigma=%g n=%d sha=%s" % (sigma, len(yte), _v155_hash(Xte_n, yte)), flush=True)
        print("[V155] VALSET sigma=%g n=%d sha=%s" % (sigma, len(yva), _v155_hash(Xva_n, yva)), flush=True)
        import torch as _t
        ok = bool(((Xte_n >= 0) & (Xte_n <= 1)).all()) and bool(_t.isfinite(Xte_n).all()) and \
            Xte_n.shape == Xte.shape and bool(((Xva_n >= 0) & (Xva_n <= 1)).all())
        d = (Xte_n - Xte).abs()
        print("[V155] selfcheck noisy images in [0,1], finite, same shape: %s  mean|noisy-clean|=%.4f "
              "changed=%.3f" % ("PASS" if ok else "FAIL", d.mean().item(), (d > 0).float().mean().item()),
              flush=True)
        if not ok:
            _v155_sys.exit("[V155] FATAL: noisy images out of range")
    st1 = _v155_rng_save()
    same = _v155_rng_equal(st0, st1)
    print("[V155] selfcheck global RNG (python, numpy, torch%s) unchanged by noise setup: %s" % (
        ", cuda" if "cuda" in st0 else "", "PASS" if same else "FAIL"), flush=True)
    if not same:
        _v155_sys.exit("[V155] FATAL: the noise setup consumed a global RNG")
    _V155.update(test=te, seed=int(seed))
    if not _V155["registered"]:
        _v155_atexit.register(_v155_xeval)
        _V155["registered"] = True
    return out


def _v155_predict(model, ds):
    import torch as _t
    dev = next(model.parameters()).device
    was_training = model.training
    model.eval()
    preds, labels = [], []
    bs = globals().get("BATCH_SIZE", 32)
    loader = _t.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
    with _t.no_grad():
        for x, y in loader:
            logits = model(x.to(dev))
            preds += logits.argmax(dim=1).cpu().tolist()
            labels += [int(v) for v in (y.tolist() if hasattr(y, "tolist") else y)]
    model.train(was_training)
    return preds, labels


def _v155_xeval():
    import torch as _t
    try:
        a = _v155_args()
        model, te, sigma = _V155["model"], _V155["test"], _v155_sigma()
        M = a.model or "?"
        S = a.seed if a.seed is not None else _V155["seed"]
        gauss = [float(x) for x in _v155_os.environ.get("V155_EVAL_SIGMAS", "").split()]
        sps = [float(x) for x in _v155_os.environ.get("V155_EVAL_SP", "").split()]
        if model is None or te is None or a.output_dir is None:
            print("[V155] XEVAL skipped: model, test split or --output_dir not captured", flush=True)
            return
        cands = [f for f in _v155_glob.glob(_v155_os.path.join(a.output_dir, "result_%s_*seed%s_*.json" % (M, S)))
                 if _v155_os.path.getmtime(f) >= _V155["t0"] - 1]
        if not cands:
            print("[V155] XEVAL skipped: no result JSON written by this run in %s" % a.output_dir, flush=True)
            return
        jf = max(cands, key=_v155_os.path.getmtime)
        j = _v155_json.load(open(jf))
        st = _v155_rng_save()
        Xte, yte = _v155_stack(te)

        def ds_at(kind, level):
            if kind == "gauss" and float(level) == 0:
                return te                              # exactly what train_gadi tested at sigma=0
            base = V155_TEST_GAUSS_BASE if kind == "gauss" else V155_TEST_SP_BASE
            return _V155FixedDS(_v155_fixed(Xte, kind, level, base), yte, te)

        def hash_at(kind, level):
            base = V155_TEST_GAUSS_BASE if kind == "gauss" else V155_TEST_SP_BASE
            return _v155_hash(_v155_fixed(Xte, kind, level, base), yte)

        saved = j.get("test_predictions", j.get("predictions"))
        saved = [int(p) for p in saved] if isinstance(saved, list) else None
        own = ds_at("gauss", sigma)
        preds, labels = _v155_predict(model, own)
        source, match = "memory", (saved is not None and preds == saved)
        if not match:                                   # more than one model built: try each, latest first
            for k_, m_ in enumerate(reversed(_V155["models"][:-1])):
                p_, l_ = _v155_predict(m_, own)
                if p_ == saved:
                    model, preds, labels, source, match = m_, p_, l_, "memory(model %d)" % k_, True
                    break
        if not match and a.checkpoint_dir and _v155_os.path.isdir(a.checkpoint_dir):
            for ck in sorted(_v155_glob.glob(_v155_os.path.join(a.checkpoint_dir, "*.pt*"))):
                try:
                    obj = _t.load(ck, map_location=next(model.parameters()).device)
                except Exception:  # noqa: BLE001
                    continue
                for key in ("best_state", "model_state", "model_state_dict", "state_dict"):
                    sd = obj.get(key) if isinstance(obj, dict) else None
                    if not isinstance(sd, dict):
                        continue
                    try:
                        model.load_state_dict(sd, strict=True)
                    except Exception:  # noqa: BLE001
                        continue
                    p_, l_ = _v155_predict(model, own)
                    if p_ == saved:
                        preds, labels, source, match = p_, l_, "ckpt:%s[%s]" % (_v155_os.path.basename(ck), key), True
                        break
                if match:
                    break
        acc_own = sum(int(p == y) for p, y in zip(preds, labels)) / float(len(labels))
        jacc = float(j.get("test_acc", float("nan")))
        acc_ok = abs(acc_own - jacc) < 1e-6 or abs(100 * acc_own - jacc) < 1e-4
        res = dict(model=M, seed=S, sigma_train=sigma, eval_sigmas=gauss, eval_sp=sps,
                   result_json=_v155_os.path.basename(jf), source=source,
                   selfcheck="MATCH" if (match and acc_ok) else "MISMATCH", n_test=len(labels),
                   train_gen_seed=(_v155_train_seed(S, sigma) if sigma > 0 else None),
                   train_fetches=(_V155["train"].n_fetch if _V155["train"] is not None else 0),
                   labels=labels, acc={}, acc_sp={}, test_hash={}, predictions={})
        for lv in gauss:
            p = preds if float(lv) == float(sigma) else _v155_predict(model, ds_at("gauss", lv))[0]
            res["predictions"]["g%g" % lv] = p
            res["acc"]["%g" % lv] = sum(int(q == y) for q, y in zip(p, labels)) / float(len(labels))
            res["test_hash"]["g%g" % lv] = hash_at("gauss", lv)
        for lv in sps:
            p = _v155_predict(model, ds_at("sp", lv))[0]
            res["predictions"]["sp%g" % lv] = p
            res["acc_sp"]["%g" % lv] = sum(int(q == y) for q, y in zip(p, labels)) / float(len(labels))
            res["test_hash"]["sp%g" % lv] = hash_at("sp", lv)
        _v155_rng_restore(st)
        out = _v155_os.path.join(a.output_dir, "v155_xeval_%s_seed%s.json" % (M, S))
        with open(out, "w") as f:
            _v155_json.dump(res, f)
        print("[V155] TRAIN noise draws this run: n=%d (gen_seed=%s)" % (res["train_fetches"], res["train_gen_seed"]),
              flush=True)
        print("[V155] XEVAL model=%s seed=%s sigma_train=%g source=%s selfcheck=%s (acc at sigma_train %.4f vs "
              "result JSON %s) gauss: %s | sp: %s" % (
                  M, S, sigma, source, res["selfcheck"], acc_own, jacc,
                  " ".join("%s=%.4f" % (k, v) for k, v in res["acc"].items()),
                  " ".join("%s=%.4f" % (k, v) for k, v in res["acc_sp"].items())), flush=True)
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        print("[V155] XEVAL FAILED: %s: %s" % (type(e).__name__, e), flush=True)
# ═══ end V155_INPUT_NOISE ════════════════════════════════════════════════════════

'''


def die(msg):
    sys.exit("PATCH REFUSED: %s" % msg)


def main():
    if len(sys.argv) != 2:
        die("usage: patch_v155_input_noise.py <repo root>")
    root = Path(sys.argv[1]).resolve()
    tp = root / "train_gadi.py"
    for p in (tp, root / "qnn_torch.py", root / "quanv_torch.py"):
        if not p.exists():
            die("%s not found" % p)
    t = tp.read_text()
    if MARK in t:
        die("train_gadi.py is already v15.5-patched (marker found). Start from a clean copy.")
    if "V15_FE_NOISE" in t:
        die("train_gadi.py carries the v15 circuit-noise patch. Build from qcq_cnn_v14, not qcq_cnn_v15.")
    for m, what in (("V14_SPLIT_AUDIT", "v14 split audit"), ("V135_M5_PATCH", "v13.5 M5 patch")):
        if m not in t:
            die("train_gadi.py lacks the %s (%s). Build from qcq_cnn_v14." % (what, m))
    if "V13_READOUT_PATCH" not in (root / "qnn_torch.py").read_text():
        die("qnn_torch.py is not the v13-patched file.")
    if "class _TensorImgDS(Dataset)" not in t:
        die("train_gadi.py: 'class _TensorImgDS(Dataset)' not found (the noise is applied to what it returns).")
    if not re.search(r"^\s*from\s+torch\.utils\.data\s+import\s+.*\bDataset\b", t, re.M):
        die("train_gadi.py: no module-level 'from torch.utils.data import ... Dataset' (the wrappers subclass it).")
    if re.search(r"num_workers\s*=\s*(?!0\b)", t):
        die("train_gadi.py sets num_workers to something other than 0; the per-fetch training noise "
            "generator must live in the main process.")
    mains = [m.start() for m in re.finditer(r"^if __name__ == [\"']__main__[\"']:", t, re.M)]
    if len(mains) != 1:
        die("train_gadi.py: expected exactly one top-level 'if __name__ == \"__main__\":', found %d." % len(mains))
    main_line = t[:mains[0]].count("\n") + 1
    lines = t.split("\n")
    for fn in ("build_model", "load_dataset"):
        # Earlier patches (v13.5, v14) re-define these names as wrappers of the previous
        # definition. Python binds the module-global name to the LAST top-level definition,
        # so the v15.5 block (inserted after all of them, just above __main__) wraps the
        # effective function. What must not exist: a top-level reference that captures
        # the name AFTER its last definition (it would bypass the v15.5 wrap).
        defs = [i + 1 for i, l in enumerate(lines) if re.match(r"def %s\(" % fn, l)]
        if not defs:
            die("train_gadi.py: no top-level 'def %s('." % fn)
        last = defs[-1]
        if last > main_line:
            die("train_gadi.py: 'def %s(' at line %d is after __main__ (line %d)." % (fn, last, main_line))
        bypass = []
        for i in range(last, main_line - 1):          # top-level lines after the last def, before __main__
            l = lines[i]
            if l[:1] in (" ", "\t", "#") or not l.strip():
                continue
            if re.search(r"(?<![\w.])%s\b(?!\s*\()" % fn, l):
                bypass.append("%d: %s" % (i + 1, l.strip()))
        if bypass:
            die("train_gadi.py: '%s' is captured by reference after its last definition "
                "(would bypass the v15.5 wrap): %s" % (fn, bypass))
        calls = len(re.findall(r"(?<![\w.])%s\(" % fn, t)) - len(defs)
        if calls < 1:
            die("train_gadi.py: %s is defined but never called by its module-global name." % fn)
        print("  %s: %d top-level definition(s) at line(s) %s (effective = line %d); %d call site(s) by name; "
              "no capture after line %d" % (fn, len(defs), defs, last, calls, last))
        for d in defs[:-1]:
            print("    line %d is wrapped by a later definition (an earlier patch); the v15.5 wrap goes on top" % d)
    if re.search(r"from\s+train_gadi\s+import|import\s+train_gadi", t):
        die("train_gadi.py imports itself; the by-name wrap would not cover that path.")

    g = re.search(r"^if __name__ == [\"']__main__[\"']:", t, re.M)
    t = t[:g.start()] + BLOCK.lstrip("\n") + "\n" + t[g.start():]
    bak = tp.with_name(tp.name + ".v14bak")
    if not bak.exists():
        shutil.copy2(str(tp), str(bak))
    tp.write_text(t)
    print("Patched %s in %s  (backup %s)" % (tp.name, root, bak.name))
    print("Next: python3 -u scripts/verify_v155.py   (must end in ALL CHECKS PASSED)")


if __name__ == "__main__":
    main()
