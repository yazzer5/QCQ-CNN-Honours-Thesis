#!/usr/bin/env python3
"""
cache_fingerprint.py -- portable content hash for the v11 MRI cache.

WHY THIS EXISTS
---------------
The original preflight hashed the .npz FILE. That was wrong. .npz is a zip
container and np.savez_compressed runs arrays through zlib, so two machines with
different numpy/zlib builds produce different file bytes from byte-identical
arrays. A file hash on an .npz is not portable.

This hashes the ARRAY CONTENTS instead: the raw little-endian bytes of each
array, in a fixed order, with dtype and shape included in the digest. That is
stable across platforms, numpy versions and compression settings.

It also prints per-split digests and pixel-level diagnostics, so that if two
machines DO disagree, the disagreement can be localised to a split and
characterised (container difference vs genuine resample difference).
"""

import argparse, hashlib, json
import numpy as np

KEYS = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test"]


def arr_digest(a):
    h = hashlib.sha256()
    h.update(str(a.dtype.str).encode())
    h.update(str(a.shape).encode())
    # force little-endian, C-contiguous, exact bytes
    h.update(np.ascontiguousarray(a).astype(a.dtype.newbyteorder("<")).tobytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--json", action="store_true", help="emit machine-readable output")
    a = ap.parse_args()

    d = np.load(a.cache, allow_pickle=True)
    meta = json.loads(str(d["meta"]))

    per = {k: arr_digest(d[k]) for k in KEYS}

    combined = hashlib.sha256()
    for k in KEYS:
        combined.update(k.encode())
        combined.update(per[k].encode())
    content_hash = combined.hexdigest()

    if a.json:
        print(json.dumps({"content_hash": content_hash, "per_array": per}, indent=2))
        return

    print("=" * 70)
    print("v11 CACHE CONTENT FINGERPRINT")
    print("=" * 70)
    print(f"file    : {a.cache}")
    print(f"mapping : {meta['mapping']}")
    print()
    print("per-array digests (first 16 chars):")
    for k in KEYS:
        arr = d[k]
        print(f"  {k:8s} {str(arr.dtype):8s} {str(arr.shape):18s} {per[k][:16]}")
    print()
    print(f"CONTENT HASH : {content_hash}")
    print()

    # diagnostics that distinguish "container difference" from "pixel difference"
    print("pixel diagnostics (for cross-machine comparison):")
    for split in ["train", "val", "test"]:
        X = d[f"X_{split}"]
        # float64 accumulation -- much more sensitive than the float32 mean
        s64 = np.float64(X).sum()
        print(f"  X_{split:5s} float64 sum = {s64!r}")
        print(f"          min={np.float64(X).min()!r}  max={np.float64(X).max()!r}")
    print()
    print("label digests:")
    for split in ["train", "val", "test"]:
        y = d[f"y_{split}"]
        print(f"  y_{split:5s} counts={np.bincount(y).tolist()} "
              f"sha={hashlib.sha256(y.tobytes()).hexdigest()[:16]}")
    print("=" * 70)


if __name__ == "__main__":
    main()
