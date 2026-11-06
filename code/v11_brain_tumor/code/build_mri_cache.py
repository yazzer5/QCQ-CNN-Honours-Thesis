#!/usr/bin/env python3
"""
build_mri_cache.py  --  v11 brain tumour cache builder

Protocol (locked 2026-09-01, see DECISIONS_v11_brain_tumor.md):
  dataset  : 7k Kaggle brain tumour MRI set (4 folders, Training/Testing)
  mapping  : Option A -- malignant = glioma ; benign = meningioma + pituitary
             (no_tumor is DROPPED entirely)
  dedup    : exact byte-level md5 dedup, applied within each original split
  boundary : the ORIGINAL Training/Testing folder boundary is preserved.
             Test images are drawn only from Testing/, train+val only from
             Training/. This guards against patient-level leakage (adjacent
             MRI slices of the same patient), which byte-level dedup cannot
             detect.
  balance  : exact 50/50 malignant/benign in every split
  benign   : drawn 50/50 from meningioma and pituitary
  image    : force grayscale (source is mixed L / RGB), bilinear resize to
             28x28, scale to [0,1], normalise with dataset train mean/std
  output   : mri_v11_cache.npz  + mri_v11_cache.sha256

Resize to 28x28 is deliberate: it keeps the front-end input shape identical to
the MNIST/Fashion-MNIST grids, so M1-M4 parameter counts stay at
1604 / 1610 / 1600 / 1594 and the iso-parametric control carries over unchanged.
"""

import argparse, hashlib, json, os, sys
from collections import defaultdict

import numpy as np
from PIL import Image

MALIGNANT = ["glioma"]
BENIGN = ["meningioma", "pituitary"]
DROPPED = ["notumor"]

IMG_SIZE = 28
SEED = 42

# label convention: 0 = benign, 1 = malignant
LABEL_BENIGN, LABEL_MALIGNANT = 0, 1


def md5(path):
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def list_class(root, split, cls):
    d = os.path.join(root, split, cls)
    if not os.path.isdir(d):
        sys.exit(f"[FATAL] missing directory: {d}")
    out = []
    for fn in sorted(os.listdir(d)):
        if fn.lower().endswith((".jpg", ".jpeg", ".png")):
            out.append(os.path.join(d, fn))
    return out


def dedup(paths):
    """Byte-level dedup, order-stable, first occurrence wins."""
    seen, keep = set(), []
    for p in paths:
        h = md5(p)
        if h in seen:
            continue
        seen.add(h)
        keep.append(p)
    return keep


def load_gray28(path):
    im = Image.open(path).convert("L")           # force grayscale
    im = im.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.asarray(im, dtype=np.float32) / 255.0


def take(rng, pool, n, what):
    if len(pool) < n:
        sys.exit(f"[FATAL] need {n} for {what}, only {len(pool)} available")
    idx = rng.permutation(len(pool))[:n]
    chosen = [pool[i] for i in sorted(idx)]
    remaining = [p for i, p in enumerate(pool) if i not in set(idx)]
    return chosen, remaining


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir containing Training/ and Testing/")
    ap.add_argument("--out", required=True, help="output .npz path")
    ap.add_argument("--n_train", type=int, default=2240)
    ap.add_argument("--n_val", type=int, default=560)
    ap.add_argument("--n_test", type=int, default=772)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    for n, nm in [(args.n_train, "n_train"), (args.n_val, "n_val"), (args.n_test, "n_test")]:
        if n % 4 != 0:
            sys.exit(f"[FATAL] {nm}={n} must be divisible by 4 "
                     f"(50/50 class balance, benign split evenly across two subtypes)")

    rng = np.random.default_rng(args.seed)

    print("=" * 68)
    print("v11 BRAIN TUMOUR CACHE BUILDER")
    print("=" * 68)
    print(f"root   : {args.root}")
    print(f"mapping: malignant={MALIGNANT}  benign={BENIGN}  dropped={DROPPED}")
    print(f"sizes  : train={args.n_train} val={args.n_val} test={args.n_test}")
    print(f"seed   : {args.seed}")
    print()

    # ---- gather + dedup, per original split -------------------------------
    pools = {}
    for split in ["Training", "Testing"]:
        for cls in MALIGNANT + BENIGN:
            raw = list_class(args.root, split, cls)
            ded = dedup(raw)
            pools[(split, cls)] = ded
            print(f"[SCAN] {split:8s} {cls:11s} raw={len(raw):5d} dedup={len(ded):5d} "
                  f"removed={len(raw)-len(ded)}")
    print()

    # ---- integrity gate: no cross-split byte duplicates --------------------
    tr_h, te_h = set(), set()
    for cls in MALIGNANT + BENIGN:
        for p in pools[("Training", cls)]:
            tr_h.add(md5(p))
        for p in pools[("Testing", cls)]:
            te_h.add(md5(p))
    leak = tr_h & te_h
    print(f"[GATE] cross-split byte-identical images: {len(leak)}")
    if leak:
        sys.exit("[FATAL] train/test leakage detected -- refusing to build cache")

    # ---- allocate ---------------------------------------------------------
    # test comes ONLY from Testing/, train+val ONLY from Training/
    per_cls_test = args.n_test // 2
    per_cls_tv = (args.n_train + args.n_val) // 2

    sel = defaultdict(list)

    # malignant = glioma
    g_test, _ = take(rng, pools[("Testing", "glioma")], per_cls_test, "glioma test")
    g_tv, _ = take(rng, pools[("Training", "glioma")], per_cls_tv, "glioma train+val")
    sel["test_mal"] = g_test
    sel["tv_mal"] = g_tv

    # benign = meningioma + pituitary, split evenly
    for bucket, split, n in [("test_ben", "Testing", per_cls_test // 2),
                             ("tv_ben", "Training", per_cls_tv // 2)]:
        for cls in BENIGN:
            got, _ = take(rng, pools[(split, cls)], n, f"{cls} {bucket}")
            sel[bucket].extend(got)

    # split tv into train / val (stratified, class-balanced)
    def split_tv(paths, n_train_half):
        idx = rng.permutation(len(paths))
        shuf = [paths[i] for i in idx]
        return shuf[:n_train_half], shuf[n_train_half:]

    tr_mal, va_mal = split_tv(sel["tv_mal"], args.n_train // 2)
    tr_ben, va_ben = split_tv(sel["tv_ben"], args.n_train // 2)

    splits = {
        "train": [(p, LABEL_MALIGNANT) for p in tr_mal] + [(p, LABEL_BENIGN) for p in tr_ben],
        "val":   [(p, LABEL_MALIGNANT) for p in va_mal] + [(p, LABEL_BENIGN) for p in va_ben],
        "test":  [(p, LABEL_MALIGNANT) for p in sel["test_mal"]] +
                 [(p, LABEL_BENIGN) for p in sel["test_ben"]],
    }

    # ---- load pixels ------------------------------------------------------
    data = {}
    for name, items in splits.items():
        order = rng.permutation(len(items))
        items = [items[i] for i in order]
        X = np.stack([load_gray28(p) for p, _ in items]).astype(np.float32)
        y = np.array([lab for _, lab in items], dtype=np.int64)
        data[name] = (X, y, [p for p, _ in items])
        n_mal = int((y == LABEL_MALIGNANT).sum())
        print(f"[LOAD] {name:5s} n={len(y):5d}  malignant={n_mal:5d}  benign={len(y)-n_mal:5d} "
              f"shape={X.shape}")
    print()

    # ---- normalise with TRAIN statistics only -----------------------------
    mu = float(data["train"][0].mean())
    sd = float(data["train"][0].std())
    print(f"[NORM] train mean={mu:.6f} std={sd:.6f}  (applied to all splits)")
    for name in data:
        X, y, paths = data[name]
        data[name] = ((X - mu) / sd, y, paths)

    # ---- integrity gate: no path appears in two splits ---------------------
    allp = defaultdict(set)
    for name in data:
        for p in data[name][2]:
            allp[p].add(name)
    overlap = {p: s for p, s in allp.items() if len(s) > 1}
    print(f"[GATE] paths appearing in >1 split: {len(overlap)}")
    if overlap:
        sys.exit("[FATAL] split overlap -- refusing to build cache")

    # ---- write ------------------------------------------------------------
    meta = {
        "code_version": "v11",
        "dataset": "kaggle_brain_tumour_mri_7k",
        "mapping": "OptionA_benign=meningioma+pituitary_malignant=glioma_notumor=DROPPED",
        "label_convention": {"0": "benign", "1": "malignant"},
        "img_size": IMG_SIZE,
        "grayscale": True,
        "resize": "bilinear",
        "dedup": "md5_within_original_split",
        "split_boundary": "original Training/Testing folders preserved",
        "n_train": int(len(data["train"][1])),
        "n_val": int(len(data["val"][1])),
        "n_test": int(len(data["test"][1])),
        "balanced": True,
        "norm_mean": mu,
        "norm_std": sd,
        "seed": args.seed,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        X_train=data["train"][0], y_train=data["train"][1],
        X_val=data["val"][0],     y_val=data["val"][1],
        X_test=data["test"][0],   y_test=data["test"][1],
        meta=json.dumps(meta),
    )

    with open(args.out, "rb") as fh:
        sha = hashlib.sha256(fh.read()).hexdigest()
    with open(args.out + ".sha256", "w") as fh:
        fh.write(f"{sha}  {os.path.basename(args.out)}\n")

    print()
    print(f"[WRITE] {args.out}")
    print(f"[SHA256] {sha}")
    print()
    print("[META] " + json.dumps(meta, indent=2))
    print("=" * 68)
    print("CACHE BUILD OK")
    print("=" * 68)


if __name__ == "__main__":
    main()
