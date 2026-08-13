# QCQ-CNN Thesis — v8 (Fashion-MNIST 2,6 arm (a))

Yasser Arafat, UNSW ELEC4951–4953.
*End-to-End Trainable Quantum-Classical-Quantum CNNs for Image Classification.*

**v8 runs FMNIST pair 2,6 (Pullover vs Shirt), experiment 2 of the difficulty
gradient.** Models, protocol and numerics are unchanged from v7 — the only
behavioural change is that the generalisation gap is measured on the restored
best-validation checkpoint inside the trainer instead of being reconstructed
afterwards from the final epoch. See `CHANGELOG_v8.md`.

## Quick start on Gadi

```bash
cd /g/data/wa66/yh2481/qcq_cnn_v8
ln -s ../qcq_cnn_v7/data data          # archives are offline-only; do not re-download
export PYTHONPATH=""
source /g/data/wa66/yh2481/miniconda3/etc/profile.d/conda.sh && conda activate qcqcnn
python scripts/preflight_data.py --data_root ./data --classes 2,6

cd pbs
./submit_run8_smoke.sh 2,6             # ONE job. Wait. Check integrity outputs.
./submit_run8_fashion.sh 2,6           # 19 more (M2/seed42 comes from the smoke test)
```

Read `PREREGISTRATION_2_6.md` before looking at any result.

---

## Locked protocol

| | |
|---|---|
| Split | 5000 / 1250 / 1000 |
| Batch size | 32, uniform across all four models |
| Optimiser | AdamW, lr 1e-3, weight_decay 1e-4 |
| Dropout | 0.0 (all four models) |
| Epochs | max 100, patience 10, min_epochs 30 for M1/M2/M3 |
| QNN back-end | `torch` (exact; verified vs `EstimatorQNN` to ~1e-6) |
| Seeds | 42, 43, 44, 45, 46 |
| Noise | α = 0 (ideal). Noise sweep is a later phase. |

Parameter counts: M1 1604, M2 1610, M3 1600, M4 1594.
Iso-parametric: |1610−1594| / 1610 = 0.99%, inside Thesis A's 1% bound.

## Validation behind this protocol

Established across three reduced scales (500 → 1000 → 3000), all archived
in `pbs/archive_v5/`:

| Fix | Why it mattered |
|---|---|
| Torch-native front-end (`quanv_torch.py`) | 4-qubit circuits cannot occupy a V100; `TorchLayer` re-serialised batches into B×196 sequential tapes |
| Torch-native QNN back-end (`qnn_torch.py`) | `EstimatorQNN` has no CUDA path at any size. 190× at batch 32, 1647× at batch 128 on Gadi |
| Early-stopping `patience_ctr` bug | Counter saturated during warmup, so every run stopped on the first epoch past `min_epochs` — would have produced a full grid of degenerate ~50% results looking like a real finding |
| Warmup widened to QNN back-end | The plateau belongs to the back-end, not the front-end (M1 frozen-front plateaus; M3 trainable-front does not) |
| Dropout → 0.0 | `Dropout2d` was the only train/eval-mode-dependent module; it inverted M1/M2's decision globally (train 0.96 / val 0.04) |

Measured throughput at n=3000: M1 1.6 s/epoch, M2 2.5, M3 1.5, M4 0.3.

## Run

```bash
source /g/data/wa66/yh2481/miniconda3/etc/profile.d/conda.sh
conda activate qcqcnn
export PYTHONPATH=""

# equivalence evidence — keep the output for the methods section
python scripts/verify_torch_quanv.py      # front-end vs brute-force 4-qubit reference
python scripts/equivalence_check.py       # front-end vs PennyLane
python scripts/verify_qnn_torch.py        # back-end vs Qiskit EstimatorQNN

cd pbs && ./submit_run6_full_grid.sh      # 20 jobs
qstat -u $USER
```

Collect:

```bash
grep -E "INVERSION|FINAL" v6_grid.o*      # must be zero INVERSION lines
python scripts/collect_results.py results_v6_mnist
```

`collect_results.py` reports mean ± std test accuracy per model and the three
attribution deltas with a **paired two-sample t-test** across seeds (Thesis A
Sec 5.3; paired because all models share the same five seeds). It warns on
mixed batch sizes, reduced-scale runs, and missing seeds.

## Layout

```
train_gadi.py                     training script
quanv_torch.py                    torch-native quantum front-end
qnn_torch.py                      torch-native QNN back-end
pennylane_frontends_legacy.py     PennyLane reference (equivalence checks only)
MODELS_SOURCE_OF_TRUTH.md         canonical spec — overrides everything
scripts/verify_torch_quanv.py     front-end vs brute-force reference
scripts/equivalence_check.py      front-end vs PennyLane
scripts/verify_qnn_torch.py       back-end vs Qiskit
scripts/collect_results.py        summary table + paired t-tests
pbs/run6_full_grid.pbs            the grid
pbs/archive_v5/                   reduced-scale scaffolding, kept for the record
data/MNIST/raw/                   staged (compute nodes have no internet)
```

## Known gaps against Thesis A

- **Noise sweep (Aim 3) not implemented.** `--noise_alpha` is accepted and
  ignored with a warning. That is 5/6 of the 450-run matrix. Requires Aer's
  `NoiseModel.from_backend('ibm_brisbane')`, and **cannot use the torch fast
  paths** — budget it against the original ~735 s/epoch back-end cost.
- **Barren-plateau measurement not implemented.** `fe_gradvar` in the logs is
  minibatch gradient noise across steps, a convergence diagnostic — *not*
  plateau evidence. The real measurement is gradient variance over random
  initialisations as a function of qubit count.
- **M1 is not a strict replication** of Long et al. (batch size, gradient
  path, early stopping, seed count all differ). Describe it as a
  re-implementation under a harmonised protocol.
- **Circuit 4's `w[5]` is structurally dead** (final CRZ is diagonal, followed
  by Z measurement). 5 effective front-end parameters, not 6. Must be excluded
  from any barren-plateau analysis.

## Gadi gotchas

- `conda activate qcqcnn` first — the prompt must show `(qcqcnn)`.
- `export PYTHONPATH=""` or system Python 3.9 contaminates the 3.11 env.
- `chmod +x pbs/*.sh` after every scp from Windows.
- `#PBS -l storage=gdata/wa66` must be declared.
- gpuvolta is not bonus-time eligible — shorter walltime schedules faster.
