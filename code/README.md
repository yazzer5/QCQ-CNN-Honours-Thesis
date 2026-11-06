# Code

Versions are sequential; each supersedes the last. Earlier versions are kept
because their results are cited in the logs and their PBS scripts document what
was actually run.

## v5_validation
Correctness debugging at reduced scale (500 → 1000 → 3000 samples). **Three
silent defects were found here**, each of which would have produced a full grid
of plausible-looking but wrong results:

1. early-stopping patience incrementing from epoch 1 (would have returned the
   degenerate epoch-1 state everywhere)
2. `Dropout2d(0.5)` globally inverting M1/M2 — train 96% / val 4%
3. the QNN warm-up keyed to trainable front-ends rather than the QNN back-end

Also contains the dropout A/B and M1 inversion investigations.

## v6_mnist
The MNIST 3-vs-5 grid (20 runs, complete). Also where the quantum simulation was
reformulated as batched torch tensor algebra — **214× at batch 32, 803× at batch
128**, verified to ~1e-6 relative error against the reference implementation.

## v7_fashion
Fashion-MNIST 0,6. Two silent launch bugs fixed: `qsub -v CLASSES=0,6` parsed as
`CLASSES=0` plus a stray variable (PBS's `-v` is itself comma-separated), and the
time-budget continuation branch silently reverted to MNIST. Added offline data
handling for compute nodes with no internet. See `docs/CHANGELOG_v7.md`.

## v8_fashion
Fashion-MNIST 2,6. Two changes from v7: qiskit imports moved behind a lazy
`_import_qiskit()` so a broken/stale qiskit install only degrades the
(unused, alpha=0) reference QNN path instead of killing every run — this is
what job 175978087 hit — and the generalisation gap is now measured as
`train_acc − test_acc` on the same restored best-validation checkpoint
(`gap_at_best`), fixing the different-models bug described in
`FMNIST_LOG.md` §6.1. See `docs/CHANGELOG_v8.md` and
`code/v8_fashion/PREREGISTRATION_2_6.md`.

## v9_fashion
Fashion-MNIST 2,4. Byte-for-byte identical to `v8_fashion/` except the
`CODE_VERSION` string (diffed directly to confirm this). Kept as a separate
folder for provenance even though it duplicates v8's code, since each grid's
result JSONs are stamped with the code version that actually produced them.
See `code/v9_fashion/PREREGISTRATION_2_4.md`.

## v10_mnist10
10-class MNIST grid — **reduced-scale/feasibility run, not the locked full
protocol** (`N_TEST=2000, MAX_EPOCHS=200` vs the binary grids'
`N_TEST=1000, MAX_EPOCHS=100`; every job log prints `REDUCED-SCALE /
FEASIBILITY RUN` explicitly — see `docs/AMENDMENT_2026-08-14_v10_reduced_scale.md`).
v9 plus: a 3-observable QNN readout (`<Z0>`, `<Z1>`, `<Z0Z1>`, same circuit,
zero extra circuit parameters), an iso-parametric re-pad for 10 classes (M1
1640 / M2 1646 / M3 1649 / M4 1643, spread 0.55% — **not** the binary
parameter counts), `--deterministic` (confirmed enabled in every job log),
and `--save_predictions` for McNemar's test (now computed — pooled McNemar
agrees in sign with the accuracy-mean test, but 4/5 individual seeds are
independently significant with the sign flipping seed to seed). Loss
function amended from BCE to cross-entropy before the grid ran
(`docs/AMENDMENT_2026-08-13_loss.md`). Raw per-run result JSONs and job logs
are now in the repo. See `docs/CHANGELOG_v10.md` and `logs/MNIST_LOG.md` §4b.

## v1_v4_replication and v3_v4_first_grid
v1–v2: the local pilot and full-scale Gadi replication of Long et al.
(`train_replication.py`, `qcq_cnn_mnist3_5.ipynb`), and the front-end circuit screen
(`m2_circuit_selection.ipynb`; Circuit 4 vs Circuit 6, five seeds). v3–v4: the first
four-model grid at batch size 1 with CPU-simulated circuits. Every trainable-front-end
job hit its 12-hour walltime before warm-up ended; that, and the idle GPU under
PennyLane, led to the exact batched simulator in v5. Checkpoints are not committed.

## v11_brain_tumor
Brain-tumour MRI (Nickparvar), the H4 real-data test. `code/build_mri_cache.py` builds
the cached tensors; `code/preflight_mri.py` includes the leakage audit. Env snapshots
in `env/`.

## v12_class_count
Class count K = 3, 4, 5 on two digit subsets, ten seeds (seed fix included). Applied
on Gadi by `setup_v12.sh`; see `docs/PREREGISTRATION_v12.md`.

## v13_readout_width
QNN readout width w = 3, 6, 15 (`patch_v13_readout.py`); see
`docs/PREREGISTRATION_v13.md`.

## v13_5_m5
v13.5: adds **M5 = M3 with the quantum front-end frozen** to the v13 code
(v12 seed fix + v13 readout-width patch, now in `v12_class_count/` and `v13_readout_width/`). Applied
on Gadi by `setup_v135.sh` → `patch_v135_m5.py`, which wraps `build_model` /
`expected_params` so M1–M4 call the originals unchanged, adds M5 to `--model`, adds M5
to the warmup rule (`needs_warmup`, amendment A1) and logs the resolved warmup per run.
Every M5 run prints a bit-exact front-end check at exit. `scripts/verify_v135_m5.py`
proves M1–M4 bit-identical to the pre-patch file at all 8 (K, width) cells. PBS:
smoke → gate (`check_v135.sh`) → grid → gate; 1 h walltime (A2); the `MODEL` clash fixed
as `GRID_MODEL` (defect D1). See `docs/PREREGISTRATION_v13_5.md`.

## v16a_closed_form_head, v16b_redesigned_head, v16bx_redesign_controls
v16a replaces the QNN head with its exact classical closed form (`closed_form_zz_v16.py`)
and dumps the head input z. v16b tests a redesigned head (`bfr_head_v16b.py`); v16b-ext
adds the controls (`bfr_ext_v16b.py`). Each folder carries the patched trainer it ran
with. See the matching `docs/PREREGISTRATION_*` and each folder's README.

## diagnostics/smoke10_mnist
10-class MNIST across all four models, at two QNN readout widths. Established
that the back-end is measurement-limited. **Not iso-parametric** — read
`README_smoke10.md` before quoting it.

## diagnostics/reps_ablation
`RealAmplitudes` depth 1 vs 2, resolving a contradiction between Long et al.'s
paper (6 params) and their published code (4 params). Prediction registered in
`docs/ABLATION_README.md` **before** the runs. Result: null.

## Environment

NCI Gadi, `gpuvolta` (Tesla V100-SXM2-32GB), project `wa66`, conda env `qcqcnn`.
Pinned: `qiskit==0.45.3`, `qiskit-machine-learning==0.7.2`, PennyLane
(lightning.gpu), PyTorch. `export PYTHONPATH=""` before pip operations.

Production runs use `--qnn_backend torch` (exact batched simulation, autograd
gradients equal to parameter-shift). The qiskit path exists for cross-validation
and is required for the noise phase.
