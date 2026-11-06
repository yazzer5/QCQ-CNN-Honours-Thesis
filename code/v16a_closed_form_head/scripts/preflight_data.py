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
    python scripts/preflight_data.py --data_root ./data --dataset mnist \
        --classes 0,1,2,3,4,5,6,7,8,9 --n_test 2000

FIX (2026-08-13): step 4 was a v7 leftover. It hardcoded Fashion-MNIST,
demanded EXACTLY two distinct classes, and checked per-class counts against
the binary protocol's 3125 train / 500 test. Any 10-class invocation failed
with "invalid class pair". Because run10_mnist10_grid.pbs calls this with
`|| exit 1`, every job in the v10 grid would have died before training
started. Step 4 now takes --dataset, accepts 2-10 classes, and derives the
per-class requirement from the protocol split rather than hardcoding it.
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
                    help="Comma-separated class labels to sanity-check. Two "
                         "for a binary task (0,6), ten for full MNIST "
                         "(0,1,2,3,4,5,6,7,8,9).")
    ap.add_argument("--dataset", choices=["mnist", "fashion_mnist"],
                    default="fashion_mnist",
                    help="Which dataset the classes belong to. Defaults to "
                         "fashion_mnist for backwards compatibility with the "
                         "v7-v9 invocations.")
    ap.add_argument("--n_train", type=int, default=5000)
    ap.add_argument("--n_val", type=int, default=1250)
    ap.add_argument("--n_test", type=int, default=1000,
                    help="Total test images. Per-class requirement is this "
                         "divided by the number of classes -- must match what "
                         "is passed to train_gadi.py, or the preflight checks "
                         "a different requirement from the one training uses.")
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

    # ---- 4. the requested classes are actually populated ----------------
    ds_key = "MNIST" if args.dataset == "mnist" else "FashionMNIST"
    print(f"\n[4/4] Checking {ds_key} classes {args.classes}")
    try:
        classes = tuple(int(v) for v in args.classes.split(","))
    except ValueError:
        fail(f"could not parse --classes {args.classes!r}; expected "
             f"comma-separated integers, e.g. '0,6' or '0,1,2,3,4,5,6,7,8,9'")
    if not 2 <= len(classes) <= 10:
        fail(f"got {len(classes)} classes {classes}; need between 2 and 10")
    if len(set(classes)) != len(classes):
        fail(f"duplicate labels in {classes}")
    if not all(0 <= c <= 9 for c in classes):
        fail(f"labels out of range in {classes}; must be 0-9")

    n_cls = len(classes)
    # Mirrors load_dataset() in train_gadi.py exactly: the train+val pool and
    # the test draw are both balanced, so each class supplies a 1/n_cls share.
    need_train = (args.n_train + args.n_val) // n_cls
    need_test = args.n_test // n_cls
    print(f"      protocol: {args.n_train}+{args.n_val} train+val and "
          f"{args.n_test} test across {n_cls} classes "
          f"-> need {need_train} train and {need_test} test PER CLASS")

    tr, te = loaded[ds_key]
    names = FASHION_NAMES if args.dataset == "fashion_mnist" else {i: f"digit {i}" for i in range(10)}
    for c in classes:
        n_tr = int((tr.targets == c).sum())
        n_te = int((te.targets == c).sum())
        label = names.get(c, "?")
        print(f"      class {c} ({label:12s}) train={n_tr:,}  test={n_te:,}")
        if n_tr < need_train:
            fail(f"class {c} has {n_tr} train images, protocol needs {need_train}")
        if n_te < need_test:
            fail(f"class {c} has {n_te} test images, protocol needs {need_test}")

    print(f"\n  PREFLIGHT PASSED -- {ds_key} {n_cls}-class "
          f"({', '.join(str(c) for c in classes)}) is ready. "
          f"No network required.\n")


if __name__ == "__main__":
    main()
