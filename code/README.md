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
Fashion-MNIST. Two silent launch bugs fixed: `qsub -v CLASSES=0,6` parsed as
`CLASSES=0` plus a stray variable (PBS's `-v` is itself comma-separated), and the
time-budget continuation branch silently reverted to MNIST. Added offline data
handling for compute nodes with no internet. See `docs/CHANGELOG_v7.md`.

*(v8 and v9 are v7 with a version tag and, for v8+, a corrected gap@best
metric — see `docs/CHANGELOG_v8.md`. Their code files are not yet checked
into this repo; only results, job logs, and docs have been added so far.)*

## v10_mnist10
Full 10-class MNIST grid. v9 plus: a 3-observable QNN readout (`<Z0>`, `<Z1>`,
`<Z0Z1>`, same circuit, zero extra circuit parameters), an iso-parametric
re-pad for 10 classes (M1 1640 / M2 1646 / M3 1649 / M4 1643, spread 0.55% —
**not** the binary parameter counts), `--deterministic`, and
`--save_predictions` for McNemar's test. Loss function amended from BCE to
cross-entropy before the grid ran (`docs/AMENDMENT_2026-08-13_loss.md`). See
`docs/CHANGELOG_v10.md` and `logs/MNIST_LOG.md` §4b. **Raw per-run result
JSONs for this grid are not yet in the repo** — only code, docs, and the four
summary figures; see `results/results_v10_mnist10/summary_v10.json`.

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
