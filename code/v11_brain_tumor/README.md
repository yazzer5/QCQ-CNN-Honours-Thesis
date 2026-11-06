# v11 — Brain Tumour MRI (arm (a), iso-parametric)

Third dataset of the Long et al. replication. Binary: benign vs malignant.

## Run it

```bash
# on Gadi, from v11_brain_tumor/
cd pbs
./submit_run11_smoke.sh     # ONE job, M2 seed 42. Check integrity output.
./submit_run11_grid.sh      # 19 more jobs. Refuses to run without the smoke.
```

## Protocol

| | |
|---|---|
| Dataset | 7,200-image Kaggle brain tumour MRI |
| Mapping | Option A: malignant = glioma, benign = meningioma + pituitary, notumor dropped |
| Split | 2240 / 560 / 772, balanced 50/50 throughout |
| Input | grayscale, bilinear 28x28, z-scored on train stats |
| Loss | BCE (binary) |
| Seeds | 42-46 |
| Params | M1 1604 / M2 1610 / M3 1600 / M4 1594 |
| Expected MDE | 3-4 pp |

## Sanity band

Majority class 50.00%. Logistic regression on the same 28x28 input 78.24%.
Long reports 86.10% (MLP) to 95.18% (QCQ-CNN) on his MRI task, on a different
dataset and mapping. Near 50% means the loader is broken. Above ~95% means
check for leakage.

## Data

The cache is not in the repo (gitignored). Build it:

```bash
python scripts/build_mri_cache.py --root <dir with Training/ and Testing/> \
    --out data/mri_v11_cache.npz
python scripts/preflight_mri.py --cache data/mri_v11_cache.npz
```

Locked content hash: `1ea96a91e59b2b6b14bc22d366b053dfeef47ed682277d57e39d4ee8e9287e60`

## What changed from v10

1. `load_dataset()` MRI branch REPLACED. The v10 stub re-split the data
   itself, expected a file format that does not exist, described the wrong
   class mapping, applied no balancing, and would have produced EMPTY val and
   test splits (it asks for 7250 images from a 3572-image cache; numpy slices
   past the end silently).
2. `_TensorImgDS` takes an explicit `pre_normalised` flag. Its `imgs.max() > 1.0`
   sniff fires on z-scored data and divides by 255, collapsing std from 1.0 to
   0.0039. Silent and fatal. Default path unchanged, so v6-v10 behaviour on
   uint8 torchvision tensors is bit-identical.
3. Split-size flags no longer trip the REDUCED-SCALE banner for MRI. They are
   a cross-check against the pre-built cache, not a rescaling instruction.
4. `scripts/preflight_mri.py` — checks cache CONTENT hash (the `.npz` file hash
   is a zip container and is not portable across numpy/zlib builds), class
   mapping, split sizes, exact balance, normalisation, finiteness.

See `docs/DECISIONS_v11_brain_tumor.md` for all eight decisions and the
revisit list.
