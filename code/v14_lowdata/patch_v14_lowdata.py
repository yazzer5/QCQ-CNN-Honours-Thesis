#!/usr/bin/env python3
"""
v14 code patch: data-split AUDIT for the low-data runs. No model, loss, optimiser
or data-selection code changes.

    python3 patch_v14_lowdata.py /path/to/repo        (run by setup_v14.sh)

WHY
  v14 only changes --n_train (50 / 100 / 250 per class). The split itself comes
  from the unchanged load_dataset(): a balanced pool of (n_train+n_val)/K images
  per class, permuted by the seed, then cut into train / val. So at low N the
  train set is a seed-dependent random draw, and val depends on N. That is the
  v10-v13.5 protocol and it is kept so N=500 reproduces v12 / v13.5 exactly.
  What the patch adds is PROOF of what each job trained and tested on.

WHAT IT CHANGES (train_gadi.py only)
  A wrapper block, inserted just above `if __name__ == "__main__":`, rebinds
  load_dataset. It calls the original with the original arguments, returns the
  original object untouched, and prints:
      [V14] SPLIT train n=... per_class=a,b,c,... min=.. max=.. hash=...
      [V14] SPLIT val   n=... per_class=...                    hash=...
      [V14] SPLIT test  n=... per_class=...                    hash=...
      [V14] SPLIT overlap train&val=0 train&test=0 val&test=0
  hash = order-independent hash of the image bytes in that split. The gate uses
  it to prove (a) every model at a (K, subset, N, seed) cell saw the SAME data,
  (b) the test set never changes, (c) no image is in two splits.
  The audit saves and restores every RNG state (python, numpy, torch, CUDA), so
  training is unaffected. verify_v14.py proves this with a dry run, audit on vs off.

  Env switches (used by verify_v14.py only, never by PBS jobs):
      V14_NO_AUDIT=1           skip the audit (bit-identity test)
      V14_SPLIT_DUMP=path      also write per-image hashes + labels to JSON
      V14_SPLIT_ONLY=1         exit right after the audit (split checks, no training)

SAFETY
  Anchored on exact text. If an anchor is missing, NOTHING is written and the
  script exits 1 with what to paste. Backup: train_gadi.py.v135bak.
  Refuses to patch twice, and refuses a repo that is not the v13.5 repo.
"""
import re
import shutil
import sys
from pathlib import Path

MARK = "# V14_SPLIT_AUDIT"

BLOCK = '''

''' + MARK + ''' -------------------------------------------------------
# v14: audit what load_dataset() returned. Calls the original with the original
# arguments and returns its result untouched. RNG states are saved and restored.
import hashlib as _v14_hashlib
import json as _v14_json
import os as _v14_os
import random as _v14_random
import numpy as _v14_np
import torch as _v14_torch

_v14_orig_load_dataset = load_dataset


def _v14_as_dataset(o):
    d = getattr(o, "dataset", None)            # DataLoader -> its dataset
    if d is not None and hasattr(d, "__getitem__") and hasattr(d, "__len__"):
        return d
    if hasattr(o, "__getitem__") and hasattr(o, "__len__") and not isinstance(o, (str, bytes, dict)):
        return o
    return None


def _v14_items(ds):
    hs, ys = [], []
    for i in range(len(ds)):
        it = ds[i]
        x, y = it[0], it[1]
        xb = _v14_torch.as_tensor(x).detach().cpu().contiguous().to(_v14_torch.float32).numpy().tobytes()
        hs.append(_v14_hashlib.blake2b(xb, digest_size=10).hexdigest())
        yt = _v14_torch.as_tensor(y)
        ys.append(int(yt.argmax()) if yt.numel() > 1 else int(yt))
    return hs, ys


def _v14_set_hash(hs):
    return _v14_hashlib.blake2b("".join(sorted(hs)).encode(), digest_size=8).hexdigest()


def _v14_audit(out):
    if _v14_os.environ.get("V14_NO_AUDIT") == "1":
        print("[V14] SPLIT audit disabled (V14_NO_AUDIT=1)", flush=True)
        return
    st = (_v14_random.getstate(), _v14_np.random.get_state(), _v14_torch.get_rng_state(),
          _v14_torch.cuda.get_rng_state_all() if _v14_torch.cuda.is_available() else None)
    try:
        parts = list(out) if isinstance(out, (tuple, list)) else [out]
        dss = [d for d in (_v14_as_dataset(p) for p in parts) if d is not None][:3]
        if len(dss) != 3:
            print(f"[V14] SPLIT-AUDIT FAILED: expected 3 splits from load_dataset, found {len(dss)} "
                  f"(return type {type(out).__name__})", flush=True)
            return
        dump = {}
        sets = {}
        for name, ds in zip(("train", "val", "test"), dss):
            hs, ys = _v14_items(ds)
            labs = sorted(set(ys))
            pc = [ys.count(c) for c in labs]
            sets[name] = set(hs)
            print(f"[V14] SPLIT {name:<5} n={len(ys)} labels={','.join(map(str, labs))} "
                  f"per_class={','.join(map(str, pc))} min={min(pc) if pc else 0} max={max(pc) if pc else 0} "
                  f"hash={_v14_set_hash(hs)}", flush=True)
            dump[name] = {"hashes": hs, "labels": ys}
        print(f"[V14] SPLIT overlap train&val={len(sets['train'] & sets['val'])} "
              f"train&test={len(sets['train'] & sets['test'])} val&test={len(sets['val'] & sets['test'])}",
              flush=True)
        p = _v14_os.environ.get("V14_SPLIT_DUMP")
        if p:
            with open(p, "w") as fh:
                _v14_json.dump(dump, fh)
    finally:
        _v14_random.setstate(st[0]); _v14_np.random.set_state(st[1]); _v14_torch.set_rng_state(st[2])
        if st[3] is not None:
            _v14_torch.cuda.set_rng_state_all(st[3])
    if _v14_os.environ.get("V14_SPLIT_ONLY") == "1":
        print("[V14] V14_SPLIT_ONLY=1: exiting after the split audit", flush=True)
        import sys as _s
        _s.stdout.flush(); _s.stderr.flush()
        _v14_os._exit(0)


def load_dataset(*args, **kwargs):
    out = _v14_orig_load_dataset(*args, **kwargs)
    _v14_audit(out)
    return out
# end ''' + MARK + '''

'''


def die(msg):
    print("PATCH ABORTED, nothing written.\n" + msg)
    print("\nPaste the output of:\n"
          "  grep -n 'def load_dataset\\|load_dataset(\\|__main__\\|V135_M5_PATCH\\|needs_warmup' train_gadi.py")
    sys.exit(1)


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    tp, qp = root / "train_gadi.py", root / "qnn_torch.py"
    for p in (tp, qp):
        if not p.exists():
            die(f"{p} not found")
    t = tp.read_text()
    if MARK in t:
        die("train_gadi.py is already v14-patched (marker found). Start from a clean copy.")
    if "V135_M5_PATCH" not in t:
        die("train_gadi.py is not the v13.5-patched file (no V135_M5_PATCH). Build from qcq_cnn_v135.")
    if "V13_READOUT_PATCH" not in qp.read_text():
        die("qnn_torch.py is not the v13-patched file.")
    if not re.search(r'needs_warmup = model_name in \("M2", "M3", "M5"\)', t):
        die("train_gadi.py: the A1-v13.5 warmup fix (\"M2\", \"M3\", \"M5\") is not present.")
    if "[V135] warmup: model=" not in t:
        die("train_gadi.py: the A1-v13.5 warmup log line is not present.")

    for pat, what in ((r"^def load_dataset\(", "def load_dataset("),
                      (r"^if __name__ == [\"']__main__[\"']:", "if __name__ == '__main__':")):
        n = len(re.findall(pat, t, re.M))
        if n != 1:
            die(f"train_gadi.py: expected exactly one top-level '{what}', found {n}.")
    calls = len(re.findall(r"(?<![\w.])load_dataset\(", t)) - 1     # minus the def
    if calls < 1:
        die("train_gadi.py: load_dataset is defined but never called by name.")
    print(f"  load_dataset: 1 definition, {calls} call site(s) by module-global name -> all go through the audit")

    g = re.search(r"^if __name__ == [\"']__main__[\"']:", t, re.M)
    t = t[:g.start()] + BLOCK.lstrip("\n") + "\n" + t[g.start():]

    bak = tp.with_name(tp.name + ".v135bak")
    if not bak.exists():
        shutil.copy2(tp, bak)
    tp.write_text(t)
    print(f"Patched {tp.name} in {root}  (backup {bak.name})")
    print("Next: python3 scripts/verify_v14.py   (must end in ALL CHECKS PASSED)")


if __name__ == "__main__":
    main()
