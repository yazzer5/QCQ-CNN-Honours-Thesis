#!/usr/bin/env python3
"""
preflight_mri.py -- v11 integrity gate, run before every MRI job.

Called from the PBS with `|| exit 1`. Mirrors the role of preflight_data.py's
mnist / fashion_mnist branches. Fails loudly rather than letting a job train on
the wrong data.

NOTE: this is a standalone gate for v11. It can be folded into
preflight_data.py as an `mri` branch, but keeping it separate means the
existing MNIST/FMNIST gates cannot be broken by this change.
"""

import argparse, hashlib, json, sys
import numpy as np

# CONTENT hash, not file hash. .npz is a zip container and np.savez_compressed
# runs arrays through zlib, so the FILE bytes differ across numpy/zlib builds
# even when the arrays are byte-identical. This digest covers the array
# contents only, and is portable across machines.
EXPECTED_CONTENT_HASH = "1ea96a91e59b2b6b14bc22d366b053dfeef47ed682277d57e39d4ee8e9287e60"
_HASH_KEYS = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]


def content_hash(d):
    import numpy as _np
    combined = hashlib.sha256()
    for k in _HASH_KEYS:
        a = d[k]
        h = hashlib.sha256()
        h.update(str(a.dtype.str).encode())
        h.update(str(a.shape).encode())
        h.update(_np.ascontiguousarray(a).astype(a.dtype.newbyteorder("<")).tobytes())
        combined.update(k.encode())
        combined.update(h.hexdigest().encode())
    return combined.hexdigest()
EXPECTED = {
    "n_train": 2240, "n_val": 560, "n_test": 772,
    "img_size": 28, "n_classes": 2,
    "mapping": "OptionA_benign=meningioma+pituitary_malignant=glioma_notumor=DROPPED",
}

def fail(msg):
    print(f"[PREFLIGHT FAIL] {msg}")
    sys.exit(1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--expect_hash", default=EXPECTED_CONTENT_HASH)
    ap.add_argument("--skip_hash", action="store_true",
                    help="only for a deliberate re-pin; never in a grid run")
    a = ap.parse_args()

    print("=" * 60)
    print("PREFLIGHT v11 -- BRAIN TUMOUR MRI")
    print("=" * 60)

    # 1. file exists + CONTENT checksum
    try:
        d = np.load(a.cache, allow_pickle=True)
    except OSError as e:
        fail(f"cannot read cache: {e}")
    meta = json.loads(str(d["meta"]))

    ch = content_hash(d)
    file_sha = hashlib.sha256(open(a.cache, "rb").read()).hexdigest()
    print(f"[1] content hash = {ch}")
    print(f"    file sha256  = {file_sha}  (informational only -- not portable)")
    if not a.skip_hash:
        if ch != a.expect_hash:
            fail(f"CONTENT hash mismatch -- the pixel data differs\n"
                 f"     expected {a.expect_hash}\n"
                 f"     got      {ch}\n"
                 f"     run cache_fingerprint.py on both machines to localise")
        print("    content hash OK")
    else:
        print("    !! CONTENT HASH CHECK SKIPPED -- not valid for a grid run")

    # 2. mapping is the locked one
    print(f"[2] mapping = {meta['mapping']}")
    if meta["mapping"] != EXPECTED["mapping"]:
        fail("class mapping is not the locked Option A mapping")

    # 3. shapes and split sizes
    for split, key in [("train", "n_train"), ("val", "n_val"), ("test", "n_test")]:
        X, y = d[f"X_{split}"], d[f"y_{split}"]
        if len(X) != len(y):
            fail(f"{split}: X/y length mismatch")
        if len(y) != EXPECTED[key]:
            fail(f"{split}: expected {EXPECTED[key]} samples, got {len(y)}")
        if X.shape[1:] != (EXPECTED["img_size"], EXPECTED["img_size"]):
            fail(f"{split}: expected {EXPECTED['img_size']}x{EXPECTED['img_size']}, got {X.shape[1:]}")
        print(f"[3] {split:5s} n={len(y):5d} shape={tuple(X.shape[1:])} OK")

    # 4. exact class balance
    for split in ["train", "val", "test"]:
        y = d[f"y_{split}"]
        counts = np.bincount(y, minlength=2)
        if len(np.unique(y)) != EXPECTED["n_classes"]:
            fail(f"{split}: expected {EXPECTED['n_classes']} classes, got {len(np.unique(y))}")
        if counts[0] != counts[1]:
            fail(f"{split}: not balanced -- benign={counts[0]} malignant={counts[1]}")
        print(f"[4] {split:5s} balance benign={counts[0]} malignant={counts[1]} OK")

    # 5. normalisation sanity (train should be ~0 mean, ~1 std)
    m, s = float(d["X_train"].mean()), float(d["X_train"].std())
    print(f"[5] train normalised mean={m:+.4f} std={s:.4f}")
    if abs(m) > 1e-3 or abs(s - 1.0) > 1e-3:
        fail("training split is not correctly normalised")

    # 6. no NaN / Inf
    for split in ["train", "val", "test"]:
        X = d[f"X_{split}"]
        if not np.isfinite(X).all():
            fail(f"{split}: contains NaN or Inf")
    print("[6] all splits finite OK")

    print("=" * 60)
    print("PREFLIGHT v11 PASS")
    print(f"  dataset : {meta['dataset']}")
    print(f"  mapping : Option A (0=benign, 1=malignant)")
    print(f"  sizes   : {meta['n_train']}/{meta['n_val']}/{meta['n_test']}")
    print("=" * 60)

if __name__ == "__main__":
    main()
