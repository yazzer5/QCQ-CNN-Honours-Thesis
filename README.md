# QCQ-CNN — Honours Thesis

**End-to-End Trainable Quantum-Classical-Quantum Convolutional Neural Networks
for Image Classification**

Yasser Arafat · UNSW Sydney, Bachelor of Electrical Engineering (Honours), 2026
Supervisor: A/Prof Vidhyasaharan Sethu

This repository holds the code, raw results, pre-registrations and write-ups for
every experiment in the thesis (v1–v16, about 3,000 training runs on NCI Gadi).

---

## Research question

Long et al. (2025, *Scientific Reports*) propose a QCQ-CNN: a **frozen,
randomly-initialised** quanvolutional front-end, a small classical CNN, and a
QNN classifier head. This work asks whether replacing that frozen front-end with
a **trainable Circuit 4 ansatz** (Hur et al. 2022), optimised end-to-end,
improves classification, and whether any improvement can be attributed to the
quantum components once model capacity is held constant.

## The models

Iso-parametric by design (within about 1% of each other at every class count), so
accuracy differences cannot be explained by capacity.

| | Front-end | Back-end |
|---|---|---|
| **M1** | frozen random quanvolution (= Long et al.) | QNN head |
| **M2** | trainable Circuit 4 | QNN head |
| **M3** | trainable Circuit 4 | classical head |
| **M4** | bilinear resize (= 2×2 mean), no quantum | classical head |
| **M5** | Circuit 4, **frozen** at initialisation | classical head (added v13.5) |
| **M6** | fixed cos(πx) per pixel, 0 parameters | classical head (added v15.8) |

Each contrast isolates one factor:

- **M2 − M1** → training the front-end
- **M3 − M2** → the quantum back-end (cost of the QNN head)
- **M3 − M4** → the quantum front-end vs none
- **M3 − M5** → front-end training with the circuit held fixed
- **M3 − M6**, **M6 − M4** → trained circuit vs a fixed classical cosine encoding

Shared backbone: `Conv2d(4,8,3)/ReLU/MaxPool → Conv2d(8,16,3)/ReLU/MaxPool → Flatten`
→ 64-dim (1,464 params). Full specification: [`docs/MODELS_SPEC.md`](docs/MODELS_SPEC.md).

## Headline findings

1. **On clean data the quantum front-end changes nothing.** M3 − M4 is within ±1 pp
   (TOST) at 3, 4 and 5 classes (v12), unchanged by readout width (v13), low data
   (v14) or IBM-calibrated hardware noise (v15). On a brain-MRI task it is harmful
   (−3.7 pp, Holm p = .002, v11).
2. **The QNN head is the bottleneck, and its cost is classical.** It costs 1.4 → 8.2 pp
   as classes grow (v12), ~2.3 pp at full readout (v13) and 13–17 pp at 50 images per
   class (v14). A closed-form classical replica of the head (12 trig features)
   reproduces the whole cost; the circuit is worth at most 0.9 pp (v16a). The head
   runs at a fixed, high gain (~9 rad per unit input; v16c). A bounded, gated redesign
   recovers 75–95% of the cost but never beats the classical head (v16b).
3. **The trained front-end is more robust to noisy input images (v15.5), but the
   effect is the angle encoding, not the circuit.** Under clipped Gaussian pixel noise,
   M3 beats M4 by 9.1 pp (σ = 0.3, K = 5, Holm p = .001). The noisy cloud is not
   smaller; it is displaced less from the first conv layer on (v15.6), and M3 gains this
   within ~3 epochs while M4 loses it as it fits clean data (v15.7). A fixed,
   zero-parameter cos(πx) map (M6) is at least as robust as the trained circuit and at
   K = 5 more so (+10.6 pp over M4, +4.2 pp over M3, Holm p = .026; v15.8).
4. **Seed 46 is an exact dead-channel initialisation.** One channel's amplitude
   √(cos²w4 + sin²w4·cos²(w2/2)) is 0.063 at that seed's init, which explains its
   failures across every experiment (v16c).
5. **MNIST 3-vs-5 is saturated** (all models within 0.08 pp, v6), a ceiling effect
   that is not evidence about quantum layers.

The full argument, with every number, is in
[`docs/analysis/V16_SYNTHESIS.md`](docs/analysis/V16_SYNTHESIS.md) and the per-experiment
results files. [`docs/EXPERIMENT_INVENTORY.md`](docs/EXPERIMENT_INVENTORY.md) has one
row per experiment.

## Method

Every experiment from v12 on followed the same protocol:

1. **Pre-register** the question, design, contrasts, predictions and stop rules
   before any code runs (`docs/preregistrations/`).
2. **Verify** the patch: existing model paths must stay bit-identical.
3. **One full-scale smoke test**, then an **integrity gate** (protocol, parameter
   counts, splits, bit-exact reproductions of earlier runs). No accuracy is read
   before the gate passes.
4. **Grid**, gate again, then analysis: paired over seeds, Holm within each family,
   MDE at 80% power for every null, TOST ±1 pp for any equivalence claim.

Defects, amendments and failed predictions are logged, dated, in the
pre-registration and the results file rather than removed.

## Repository layout

```
docs/
  MODELS_SPEC.md          architecture specification (source of truth for M1–M4)
  EXPERIMENT_INVENTORY.md one row per experiment, v1–v16
  preregistrations/       written before each experiment ran
  results/                results write-up per experiment (v12–v16c)
  analysis/               cross-experiment synthesis, closed-form front-end maths,
                          noise mechanism and theory, deep analysis of v6–v12
  process/                changelogs, amendments, GPU root-cause note
logs/
  MNIST_LOG.md            running log for the MNIST work (v5–v15), with retractions
  FMNIST_LOG.md           running log for the Fashion-MNIST work (v7–v9)
code/                     one folder per experiment version (see code/README.md)
results/                  raw result JSONs, job logs and analysis per experiment
                          (see results/README.md)
figures/                  early analysis figures (superseded ones in figures/superseded/)
reference/                citations for the two papers this work builds on
DATASETS.md               how to obtain and verify the datasets
```

## What is and is not in the repository

- **Code and raw results:** every experiment from v1 to v16b-ext. v16c was analysis
  only (no training).
- **v15.6, v15.7 and v15.8** (noise geometry, training dynamics, classical cos
  control): pre-registrations and results write-ups are in `docs/`; the code
  packages and raw result files are not yet in the repository.
- **Not committed:** datasets (see `DATASETS.md`), model checkpoints (deleted after
  each job's self-check from v15 on), and archive tarballs.

## Reproducing a run

```bash
cd code/v15_5_input_noise   # each version folder has its own README and setup script
cat README_v155.md
```

The earliest grids run directly:

```bash
cd code/v7_fashion
python scripts/preflight_data.py --data_root ./data --classes 0,6
python train_gadi.py --model M2 --dataset fashion_mnist --classes 0,6 \
                     --seed 42 --output_dir results_test --data_root ./data
```

Hardware: NCI Gadi, `gpuvolta` queue (Tesla V100-SXM2-32GB). Pinned:
`qiskit==0.45.3`, `qiskit-machine-learning==0.7.2`, PennyLane, PyTorch. Exported
environments: `code/v12_class_count/env_qcqcnn_v10.yml` and `pip_freeze_v10.txt`.

## Caveats — read before quoting any number

- **Every null is bounded by an MDE.** A non-significant contrast is a power
  statement, never evidence of equivalence; equivalence is claimed only from TOST.
- **v10 (10-class MNIST) ran reduced-scale**, not the locked full protocol. Treat it
  as feasibility; v12/v13 at 10 seeds supersede it. See
  `docs/process/AMENDMENT_2026-08-14_v10_reduced_scale.md`.
- **v6–v9 are not bit-reproducible at a fixed seed** (GPU reduction order, ~0.8 pp).
  From v10 on, runs use `--deterministic` and later experiments reproduce earlier
  runs bit-exactly.
- **Two early figures use a withdrawn metric** (generalisation gap computed against
  a different checkpoint). They are kept in `figures/superseded/` for the record;
  see `logs/FMNIST_LOG.md` §6.1.
- **The 10-class diagnostic** (`results/results_smoke10_mnist10/`) is not
  iso-parametric. It is a diagnostic, not a result.

## Citation

If you use this code or these results, please cite the thesis (see `CITATION.cff`).

## Licence

Code is released under the MIT licence (`LICENSE`). Datasets and the cited papers
remain under their own licences.
