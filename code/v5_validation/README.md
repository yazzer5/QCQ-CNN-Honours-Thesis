# QCQ-CNN Thesis — v5

Yasser Arafat, UNSW ELEC4951–4953.
Thesis: *End-to-End Trainable Quantum-Classical-Quantum CNNs for Image
Classification.*

**Scope of this version: ideal (noiseless) MNIST 3-vs-5, all four models.**
Fashion-MNIST, MRI, and the noise sweep come after MNIST reproduces Long et
al.'s ~99.4%.

---

## What changed from v4

The persistent 0% GPU utilisation is fixed, and the root cause was not device
selection.

1. **4-qubit circuits cannot occupy a V100.** 16 amplitudes, a few dozen FLOPs
   per gate, ~5–10 µs of launch overhead per call. `lightning.gpu` is *slower*
   than `lightning.qubit` below ~18 qubits.
2. **The v4 `unfold()` batching fix never reduced the number of circuit calls.**
   `TorchLayer` only issues one device execution if the device supports
   parameter broadcasting. The lightning devices don't, so PennyLane applied
   `broadcast_expand` and re-serialised the batch into `B × 196` sequential
   tapes. That is the whole of the M3 ~490–519 s/epoch "anomaly."

**Fix:** the front-end circuits are computed directly as batched torch tensor
algebra (`quanv_torch.py`) instead of being handed to a simulator. The circuits
are unchanged; only how they're evaluated changed.

Consequences:

- Whole model runs on `cuda`.
- `batch_size=1` is no longer required — there is no parameter-shift tape
  machinery left. **Batch size is now 32 for all four models.**
- `--broadcast_shift` and `verify_broadcast_shift.py` are gone (dead code).
- A permanent `[GRADCHECK]` tripwire aborts before training if any trainable
  tensor receives no gradient — the M1 near-random-accuracy failure mode.

Verified numbers (reproducible, see `scripts/`):

| Check | Result |
|---|---|
| Forward vs brute-force 4-qubit reference | 8.9e-16 |
| Autograd vs four-term parameter-shift | ~1e-15 (float64), 2.8e-07 rel (float32) |
| torch vs PennyLane front-end, forward | 3.6e-07 rel |
| Speedup, CPU only | 282× (M2/M3), 292× (M1) |

---

## Run order — do not skip steps

Upload, then `chmod +x pbs/*.sh` (scp from Windows drops the exec bit).

```bash
source /g/data/wa66/yh2481/miniconda3/etc/profile.d/conda.sh
conda activate qcqcnn
export PYTHONPATH=""
```

### 0. Equivalence evidence (login node, ~1 min)

```bash
python scripts/verify_torch_quanv.py     # torch vs brute-force 4-qubit reference
python scripts/equivalence_check.py      # torch vs PennyLane, fwd + grad, M1 and M2/M3
```

Save both outputs. This is the methods-section evidence. If either fails, stop.

### 1. Timing run (`pbs/`, ~1 h walltime)

```bash
cd pbs && ./submit_run5_timing.sh mnist 42
```

Full data volume, 3 epochs. Check in the `.o` files:

- `[GRADCHECK]` prints a non-zero `|grad|` for **every** trainable tensor
- per-epoch time is seconds, not hundreds of seconds
- `gpu_util_v5timing_*.csv` shows non-zero utilisation

The M1/M2 QNN **back-end** still runs on CPU through qiskit's Estimator and is
now the slowest remaining component. This run is what tells you whether it
matters.

### 2. Batch-size control (~10 h walltime for the bs=1 arm)

```bash
./submit_run5_control.sh
```

M2, MNIST, seed 42, bs=1 vs bs=32. This is the evidence that the v5 batch-size
change is benign, and it is what a marker will ask for. Compare final test
accuracy and the loss curves.

### 3. Full MNIST grid (20 jobs)

```bash
./submit_run5_mnist_grid.sh
```

4 models × 5 seeds, locked protocol: 5,000/1,250/1,000, 100 max epochs,
patience 10, `min_epochs` 30 for M2/M3, bs=32, lr=1e-3, Adam. Self-chains on
exit code 99 if a job hits its time budget.

```bash
python scripts/collect_results.py results_v5_mnist
```

Prints per-model mean ± std test accuracy and the three attribution deltas
(M2−M1 front-end trainability, M2−M3 quantum back-end, M3−M4 quantum
front-end), and warns on mixed batch sizes, reduced-scale runs, or missing
seeds.

---

## Layout

```
train_gadi.py                     training script
quanv_torch.py                    torch-native quantum front-ends (production)
pennylane_frontends_legacy.py     PennyLane reference front-ends (checking only)
MODELS_SOURCE_OF_TRUTH.md         canonical locked spec — overrides everything
PROGRESS_2026-07-28_*.md          root cause, batch-size reasoning, next actions
scripts/verify_torch_quanv.py     torch vs brute-force 4-qubit reference
scripts/equivalence_check.py      torch vs PennyLane
scripts/collect_results.py        aggregate result_*.json -> summary table
pbs/run5_timing.pbs               step 1
pbs/run5_batchsize_control.pbs    step 2
pbs/run5_mnist_grid.pbs           step 3
data/MNIST/raw/                   staged — gpuvolta nodes have no internet
```

---

## Gadi gotchas (carried forward, all still true)

- `chmod +x` after every `scp` from Windows.
- `export PYTHONPATH=""` before anything, or system Python 3.9 contaminates the
  3.11 env.
- `#PBS -l storage=gdata/wa66` must be declared or files silently aren't found.
- PBS scripts live in `pbs/` but the repo root is one level up — they all
  `cd "$PBS_O_WORKDIR/.."` and then `ls train_gadi.py` so a path bug fails
  loudly instead of as one stderr line.
- `gpuvolta` is **not** bonus-time eligible. Shorter walltime = shorter queue.
- Stage data before upload; compute nodes have no internet.

---

## Open items

- Rewrite thesis §4.4.1 (the minibatch gradient-cancellation claim doesn't
  hold — see `MODELS_SOURCE_OF_TRUTH.md` Sec 6). Flag to Sethu.
- M1/M2 QNN back-end is still CPU-only (qiskit Estimator, no GPU code path).
  Out of scope until the noise phase, but it's now the bottleneck.
- Pull and archive the v4 `174853951`–`954` outputs for the record.
