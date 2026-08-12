# v7 — changes from v6

v7 is v6 with the Fashion-MNIST launch path fixed. **No model, protocol, or
numerical code was touched.** `train_gadi.py`, `quanv_torch.py`, `qnn_torch.py`
and `MODELS_SOURCE_OF_TRUTH.md` are byte-identical to v6, so the completed
MNIST α=0 grid remains valid and comparable.

## Fixed

**1. `qsub -v CLASSES=0,6` never worked.**
PBS's `-v` argument is itself a comma-separated list of assignments, so
`-v MODEL=M1,DATASET=fashion_mnist,SEED=42,CLASSES=0,6` parses as
`CLASSES=0` plus a stray variable named `6`. Every Fashion job would have
trained on a malformed class pair while the log printed something plausible.
The pair is now passed as `CLASSPAIR=0_6` and converted back to `0,6` inside
the job script.

**2. The `rc=99` continuation branch reverted to MNIST.**
`run6_fashion_grid.pbs` resubmitted `run6_full_grid.pbs` and dropped `CLASSES`
entirely, so any job hitting the internal time budget silently restarted as an
MNIST run writing into `results_v6_mnist/`. It now resubmits itself and carries
`CLASSPAIR` through. (At 2.5–4.4 s/epoch this branch should never fire, but a
silent dataset swap is not something to leave to luck.)

**3. Output namespacing.**
Job name `v7_fash`; results in `results_v7_fashion_c06/`, checkpoints in
`ckpt_v7_fashion_c06/`, GPU traces `gpu_util_v7f_*`. v6 MNIST outputs are
untouched.

**4. Input validation.**
`submit_run7_fashion.sh` rejects anything that isn't a single-digit `a,b` pair
before submitting 20 jobs.

## Renamed

| v6 | v7 |
|---|---|
| `pbs/run6_fashion_grid.pbs` | `pbs/run7_fashion_grid.pbs` |
| `pbs/submit_run6_fashion.sh` | `pbs/submit_run7_fashion.sh` |

`pbs/run6_full_grid.pbs` and `pbs/submit_run6_full_grid.sh` are unchanged and
retained — the MNIST grid they produced is complete and should not be re-run.

## Verify before trusting any Fashion result

```bash
grep -cE "INVERSION" v7_fash.o*          # every count MUST be 0
grep -h "CLASSES=" v7_fash.o* | sort -u  # MUST show exactly CLASSES=0,6
grep -h "REDUCED-SCALE" v7_fash.o*       # MUST be empty
```

The middle check is the one that catches defect 1 above. Run it before reading
a single accuracy number.

---

# v7 — offline dataset handling (Fashion-MNIST)

Gadi compute nodes have no internet, and Fashion-MNIST had never actually been
run, so the offline path was verified rather than assumed.

## Shipped

`data/` is **inside the v7 folder**, so the zip is self-contained:

```
data/MNIST/raw/*.gz            4 files
data/FashionMNIST/raw/*.gz     4 files   (26.4 MB train images)
```

All eight verified against torchvision's published MD5s.

## Changed

**`train_gadi.py` — new `_open_offline_first()` helper.** v6 called
`datasets.FashionMNIST(..., download=True)` and relied on torchvision noticing
the staged `.gz` had a matching MD5 and skipping the fetch. That does work
(tested on torchvision 0.28 with the CDN unreachable), but it works via a
fallback *inside* the download path, and that code has moved between versions.
v7 tries `download=False` first, so with the archives extracted no network call
is ever attempted; it falls back to the v6 behaviour otherwise. This is the only
change to `train_gadi.py` and it touches no model, protocol or numerical code.

**`scripts/preflight_data.py` — new.** MD5-verifies all eight archives,
extracts them into `raw/` so torchvision's `_check_exists()` passes on the first
call, re-opens both datasets with `download=False` as an offline proof, and
confirms the requested class pair has enough images for the 5000/1250/1000
protocol. Run once on the login node; also invoked at the top of
`run7_fashion_grid.pbs`, where it fails the job loudly instead of letting it
stall on a network timeout.

## Verified locally

| Check | Result |
|---|---|
| 8/8 archive MD5s | match |
| FashionMNIST offline, CDN unreachable | loads, 60,000 train / 10,000 test |
| Class pair 0,6 | 6,000 train + 1,000 test per class (need 3,125 / 500) |
| Split under the locked protocol | 5000 / 1250 / 1000 exactly |
| Label remapping | `{0,6} → {0,1}`, test split 500/500 |

Note `data/` grows to ~110 MB after extraction. `.gitignore` already excludes
it; that is on the `/g/data/wa66` filesystem, not `$HOME`.
