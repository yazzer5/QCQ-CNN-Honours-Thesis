# Datasets

Datasets are **not committed** — they are ~41 MB of publicly available archives.
This file has everything needed to reconstruct them byte-for-byte.

## What is needed

Both are the standard IDX-format archives distributed with `torchvision`.

| Dataset | Used for |
|---|---|
| **MNIST** | 3-vs-5 binary grid; 10-class diagnostic |
| **Fashion-MNIST** | pairs 0,6 · 2,6 · 2,4 |

## Expected layout

Every training script expects a `data/` directory alongside it:

```
data/
├── MNIST/raw/
│   ├── train-images-idx3-ubyte.gz
│   ├── train-labels-idx1-ubyte.gz
│   ├── t10k-images-idx3-ubyte.gz
│   └── t10k-labels-idx1-ubyte.gz
└── FashionMNIST/raw/
    ├── train-images-idx3-ubyte.gz
    ├── train-labels-idx1-ubyte.gz
    ├── t10k-images-idx3-ubyte.gz
    └── t10k-labels-idx1-ubyte.gz
```

One copy can be shared across versions with a symlink:

```bash
cd code/v7_fashion && ln -s ../../data ./data
```

## Option 1 — torchvision (machine with internet)

```python
from torchvision import datasets
datasets.MNIST("./data", train=True,  download=True)
datasets.MNIST("./data", train=False, download=True)
datasets.FashionMNIST("./data", train=True,  download=True)
datasets.FashionMNIST("./data", train=False, download=True)
```

## Option 2 — direct download

Fashion-MNIST, from the official repository
(`github.com/zalandoresearch/fashion-mnist`):

```bash
mkdir -p data/FashionMNIST/raw && cd data/FashionMNIST/raw
BASE=http://fashion-mnist.s3-website.eu-central-1.amazonaws.com
for f in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz \
         t10k-images-idx3-ubyte.gz  t10k-labels-idx1-ubyte.gz; do
    wget "$BASE/$f"
done
```

MNIST's original host (`yann.lecun.com/exdb/mnist`) has been unreliable; mirrors
include `ossci-datasets.s3.amazonaws.com/mnist/` and the HuggingFace datasets
hub. Torchvision handles mirror fallback automatically, so **option 1 is
preferred**.

## Verify

MD5 checksums (torchvision's published values):

```
MNIST/raw/train-images-idx3-ubyte.gz   f68b3c2dcbeaaa9fbdd348bbdeb94873
MNIST/raw/train-labels-idx1-ubyte.gz   d53e105ee54ea40749a09fcbcd1e9432
MNIST/raw/t10k-images-idx3-ubyte.gz    9fb629c4189551a2d022fa330f9573f3
MNIST/raw/t10k-labels-idx1-ubyte.gz    ec29112dd5afa0611ce80d1b7f02629c

FashionMNIST/raw/train-images-idx3-ubyte.gz   8d4fb7e6c68d591d4c3dfef9ec88bf0d
FashionMNIST/raw/train-labels-idx1-ubyte.gz   25c81989df183df01b3e8a0aad5dffbe
FashionMNIST/raw/t10k-images-idx3-ubyte.gz    bef4ecab320f06d8554ea6380940ec79
FashionMNIST/raw/t10k-labels-idx1-ubyte.gz    bb300cfdad3c16e7a12a480ee83cd310
```

The repository ships a checker that verifies all eight, extracts them, proves an
offline open, and validates the requested class pair:

```bash
cd code/v7_fashion
python scripts/preflight_data.py --data_root ./data --classes 0,6
```

Must end with `PREFLIGHT PASSED`. A mismatch almost always means a truncated
transfer — re-download that file.

## HPC note (no internet on compute nodes)

NCI Gadi compute nodes cannot reach the network. Download on a **login node**,
then run the preflight there once — it extracts the archives so `torchvision`'s
existence check passes immediately and the download path is never entered.
`train_gadi.py` also attempts `download=False` first for the same reason.

## Class pairs used

Fashion-MNIST labels: 0 T-shirt/top · 1 Trouser · 2 Pullover · 3 Dress ·
4 Coat · 5 Sandal · 6 Shirt · 7 Sneaker · 8 Bag · 9 Ankle boot

| Pair | Classes | M4 baseline | Note |
|---|---|---|---|
| 0,6 | T-shirt/top vs Shirt | 84.2% | hardest sampled; primary |
| 2,6 | Pullover vs Shirt | 87.1% | queued |
| 2,4 | Pullover vs Coat | 88.4% | queued |
| 0,1 | T-shirt/top vs Trouser | 98.8% | Thesis A's pair — saturated |

Protocol: 5,000 train / 1,250 validation / 1,000 test, class-balanced, labels
remapped to {0,1}, seeds 42–46.
