#!/usr/bin/env python
"""
Offline data preflight for Gadi compute nodes (NO INTERNET).

Why this exists
---------------
`train_gadi.py` calls `datasets.FashionMNIST(..., download=True)`. On a modern
torchvision that is still safe offline: `_check_exists()` looks for the
EXTRACTED idx files, fails when only `.gz` is staged, enters `download()`, and
`download_url()` then finds the `.gz` already present with a matching MD5 and
skips the network. It works -- but it works by relying on a fallback inside the
download path, and the exact behaviour has shifted across torchvision versions.

This script removes that dependency. It extracts the `.gz` archives into
`raw/` so `_check_exists()` returns True on the very first call and the
download path is never entered. It then re-opens both datasets with
`download=False`, which raises if anything is missing -- proving the compute
node needs no network.

Run it ONCE on the Gadi login node after uploading v7. It is also invoked
automatically at the top of run7_fashion_grid.pbs (idempotent, ~2 s once
extracted).

Usage:
    python scripts/preflight_data.py --data_root ./data --classes 0,6
"""
import argparse
import gzip
import hashlib
import shutil
import sys
from pathlib import Path

# torchvision's published MD5s. If one of these fails the file is corrupt or
# truncated -- almost always a bad scp -- and must be re-uploaded.
EXPECTED_MD5 = {
    "MNIST/raw/train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "MNIST/raw/train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "MNIST/raw/t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "MNIST/raw/t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
    "FashionMNIST/raw/train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "FashionMNIST/raw/train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "FashionMNIST/raw/t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "FashionMNIST/raw/t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}

FASHION_NAMES = {
    0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
    5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot",
}


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def fail(msg):
    print(f"\n  PREFLIGHT FAILED: {msg}\n", file=sys.stderr)
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--classes", default="0,6",
                    help="Fashion-MNIST class pair to sanity-check, e.g. 0,6")
    args = ap.parse_args()

    root = Path(args.data_root)
    print(f"Data root: {root.resolve()}")
    if not root.is_dir():
        fail(f"{root} does not exist. The data/ folder ships inside v7 -- "
             f"upload it, or symlink the v6 copy.")

    # ---- 1. every archive present and byte-correct -----------------------
    print("\n[1/4] Verifying staged archives")
    for rel, expect in EXPECTED_MD5.items():
        p = root / rel
        if not p.is_file():
            fail(f"missing archive: {p}")
        got = md5(p)
        state = "ok" if got == expect else "MISMATCH"
        print(f"      {state:8s} {rel}  ({p.stat().st_size:,} bytes)")
        if got != expect:
            fail(f"MD5 mismatch on {rel}\n"
                 f"      expected {expect}\n"
                 f"      got      {got}\n"
                 f"      Re-upload this file -- scp almost certainly truncated it.")

    # ---- 2. extract so _check_exists() passes without the network --------
    print("\n[2/4] Extracting archives (so torchvision never enters download())")
    for rel in EXPECTED_MD5:
        gz = root / rel
        out = gz.with_suffix("")           # strip .gz
        if out.is_file() and out.stat().st_size > 0:
            print(f"      present  {out.name}")
            continue
        with gzip.open(gz, "rb") as fin, open(out, "wb") as fout:
            shutil.copyfileobj(fin, fout)
        print(f"      extracted {out.name}  ({out.stat().st_size:,} bytes)")

    # ---- 3. open with download=False: proves no network is needed --------
    print("\n[3/4] Opening datasets with download=False (offline proof)")
    try:
        from torchvision import datasets
    except ImportError:
        fail("torchvision not importable. Did you `conda activate qcqcnn` "
             "and `export PYTHONPATH=\"\"`?")

    loaded = {}
    for name, cls in (("MNIST", datasets.MNIST),
                      ("FashionMNIST", datasets.FashionMNIST)):
        try:
            tr = cls(str(root), train=True, download=False)
            te = cls(str(root), train=False, download=False)
        except Exception as exc:
            fail(f"{name} failed to load offline: {exc}")
        loaded[name] = (tr, te)
        print(f"      {name:13s} train={len(tr):,}  test={len(te):,}")

    # ---- 4. the requested class pair is actually populated ---------------
    print(f"\n[4/4] Checking Fashion-MNIST class pair {args.classes}")
    try:
        pair = tuple(int(v) for v in args.classes.split(","))
    except ValueError:
        fail(f"could not parse --classes {args.classes!r}; expected 'a,b'")
    if len(pair) != 2 or not all(0 <= c <= 9 for c in pair) or pair[0] == pair[1]:
        fail(f"invalid class pair {pair}; need two distinct labels in 0-9")

    tr, te = loaded["FashionMNIST"]
    # Protocol needs 5000 train + 1250 val = 6250 total, balanced -> 3125/class.
    need_train, need_test = 3125, 500
    for c in pair:
        n_tr = int((tr.targets == c).sum())
        n_te = int((te.targets == c).sum())
        label = FASHION_NAMES.get(c, "?")
        print(f"      class {c} ({label:12s}) train={n_tr:,}  test={n_te:,}")
        if n_tr < need_train:
            fail(f"class {c} has {n_tr} train images, protocol needs {need_train}")
        if n_te < need_test:
            fail(f"class {c} has {n_te} test images, protocol needs {need_test}")

    print(f"\n  PREFLIGHT PASSED -- {FASHION_NAMES.get(pair[0],'?')} vs "
          f"{FASHION_NAMES.get(pair[1],'?')} is ready. No network required.\n")


if __name__ == "__main__":
    main()
