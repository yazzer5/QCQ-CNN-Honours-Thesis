# QCQ-CNN — Honours Thesis

**End-to-End Trainable Quantum-Classical-Quantum Convolutional Neural Networks
for Image Classification**

Yasser (z5477389) · UNSW Electrical Engineering Honours
Supervisor: A/Prof Vidhyasaharan Sethu

> **Private repository — unsubmitted thesis work.** Do not distribute.

---

## Research question

Long et al. (2025, *Scientific Reports*) propose a QCQ-CNN: a **frozen,
randomly-initialised** quanvolutional front-end, a small classical CNN, and a
QNN classifier head. This work asks whether replacing that frozen front-end with
a **trainable Circuit 4 ansatz** (Hur et al. 2022), optimised end-to-end,
improves classification — and whether any improvement is attributable to the
quantum components once model capacity is held constant.

## The four models

Iso-parametric by design: all within **0.93%** of ~1,600 parameters, so accuracy
differences cannot be explained by capacity.

| | Front-end | Back-end | Params |
|---|---|---|---|
| **M1** | frozen Henderson quanvolution | QNN head | 1,604 |
| **M2** | trainable Circuit 4 | QNN head | 1,610 |
| **M3** | trainable Circuit 4 | classical head | 1,600 |
| **M4** | bilinear resize (no quantum) | classical head | 1,594 |

Each adjacent contrast isolates one factor:

- **M2 − M1** → front-end trainability
- **M2 − M3** → the quantum back-end
- **M3 − M4** → the quantum front-end

Shared backbone: `Conv2d(4,8,3)/ReLU/MaxPool → Conv2d(8,16,3)/ReLU/MaxPool →
Dropout2d → Flatten` → 64-dim (1,464 params).

## Headline findings

**1. MNIST 3-vs-5 is saturated and cannot evaluate the question.**
All four models land within **0.08 pp** (99.14–99.22%) against a design
resolution of ~0.25–0.40 pp. M4, which contains no quantum components at all, is
statistically indistinguishable from every quantum variant. This is a **ceiling
effect, not a null result about quantum layers** — and it is the control that
shows why MNIST-class results in this literature should be read cautiously.

**2. Across all three Fashion-MNIST pairs, the QNN head is consistently the
weakest component — but which classical-head model wins is not stable.**
On 0-vs-6 (T-shirt/top vs Shirt), the three pre-registered contrasts remain
null (MDE ≈ 1.8 pp), but the end-to-end comparison is significant: models
carrying the QNN head sit **1.18 pp behind** the classical heads (p = 0.011),
and M4 (fully classical) is best overall. On 2-vs-6 and 2-vs-4, however, **M3**
(trainable quantum front-end + classical head) overtakes M4 as the best single
model — a pre-registered prediction that M4 would stay best was falsified on
both remaining pairs. The pre-planned difficulty gradient across the three
pairs also collapsed: 2,6 and 2,4 landed 0.18 pp apart in M4 accuracy, inside
the ~0.8 pp non-determinism floor. See `FMNIST_LOG.md` §5.7 for the full
cross-experiment synthesis.

**3. The QNN back-end is measurement-limited, not parameter-limited.**
Widening the readout from one observable ⟨ZZ⟩ to three (⟨Z0⟩, ⟨Z1⟩, ⟨ZZ⟩) — at
**zero circuit-parameter cost** — roughly doubles 10-class MNIST accuracy
(33.5% → 66.5% for M1). Conversely, adding 50% more variational parameters to
the ansatz changes nothing (−0.02 pp, p = 0.975). The single-scalar readout is a
real architectural limit, visible independently in an early dropout-inversion
defect and in the 10-class diagnostic.

**4. M1 faithfully reproduces Long et al.'s published implementation.**
Verified line by line against their source. The only difference is the QNN
ansatz depth, where **Long's paper and their own code contradict each other**
(6 vs 4 variational parameters). This repository follows the paper; the
difference was measured across 5 seeds and found immaterial.

## Repository layout

```
code/
  v5_validation/     correctness debugging; three silent defects found here
  v6_mnist/          MNIST 3-vs-5 grid; batched torch simulation (214-803x)
  v7_fashion/        Fashion-MNIST; offline data handling; two launch bugs fixed
  v10_mnist10/       full 10-class MNIST grid; 3-observable readout, re-padded
                     iso-parametric budget, --deterministic, --save_predictions
  diagnostics/
    smoke10_mnist/   10-class MNIST diagnostic (readout width)
    reps_ablation/   reps=1 vs reps=2 ansatz depth ablation

results/
  results_v6_mnist_3v5/              20 runs — 4 models x 5 seeds
  results_v7_fashion_c06/            20 runs — Fashion 0,6
  results_v8_fashion_c26/            20 runs — Fashion 2,6 (gap@best metric fix)
  results_v9_fashion_c24/            20 runs — Fashion 2,4 (closes arm-a sequence)
  results_v10_mnist10/               20 runs — full 10-class grid (summary only,
                                      raw JSONs not yet in repo — see note below)
  results_reps_fashion_mnist_c06/    20 runs — ansatz depth ablation
  results_smoke10_mnist10/            6 runs — 10-class diagnostic
  results_v5_*/                      reduced-scale validation
  results_v*/analysis/               plots + standalone HTML report, per experiment
  analysis_index.html                links all five per-experiment reports
  job_logs/                          PBS stdout/stderr — full epoch histories
  gpu_traces/                        GPU utilisation CSVs

logs/        MNIST_LOG.md, FMNIST_LOG.md  <- START HERE
figures/     analysis figures (see caveat below)
docs/        architecture spec, changelogs, handover notes
reference/   Long et al. 2025, Hur et al. 2022
```

## Start here

- **`logs/MNIST_LOG.md`** — the four models, the 3-vs-5 grid, the three silent
  defects, the 10-class diagnostic, and the full M1-vs-Long verification (§8).
  §9 has the current position and the experiment queue.
- **`logs/FMNIST_LOG.md`** — task selection, the Fashion-MNIST 0,6 grid, and the
  statistical analysis. §9 has the next experiments and registered predictions.

Both are written for supervisor review and include retractions where earlier
claims did not survive scrutiny.

## Datasets

Not committed. See **[`DATASETS.md`](DATASETS.md)** for download instructions and
MD5 checksums.

## Reproducing a run

```bash
cd code/v7_fashion
python scripts/preflight_data.py --data_root ./data --classes 0,6
python train_gadi.py --model M2 --dataset fashion_mnist --classes 0,6 \
                     --seed 42 --output_dir results_test --data_root ./data
```

Hardware used: NCI Gadi, `gpuvolta` queue (Tesla V100-SXM2-32GB), project
`wa66`. Pinned: `qiskit==0.45.3`, `qiskit-machine-learning==0.7.2`, PennyLane
(lightning.gpu), PyTorch.

## Caveats — read before quoting any number

- **Two figures use a withdrawn metric.** `fig1_accuracy_and_gap.png` panel (b)
  and `fig3_gap_vs_acc_and_epochs.png` compute the generalisation gap as
  final-epoch train accuracy minus test accuracy — but the trainer restores the
  best-validation checkpoint before testing, so those are different models. See
  `FMNIST_LOG.md` §6.1. **Regenerate before use.**
- **Runs are not bit-reproducible at fixed seed.** GPU float reduction order is
  non-deterministic; identical configurations differ by ~0.8 pp, comparable to
  the between-seed SD of 0.92 pp. Disclosed in `FMNIST_LOG.md` §6.5.
- **Every null is bounded by an MDE.** With 5 seeds the design detects ~1.8 pp on
  Fashion-MNIST and ~0.25–0.40 pp on MNIST. A null is a power statement, never
  evidence of equivalence.
- **The 10-class diagnostic is not iso-parametric.** M3/M4 carry ~480 more
  parameters than M1/M2 there. It is a diagnostic, not a result.

## Status

Complete: MNIST 3-vs-5 grid · Fashion-MNIST 0,6 / 2,6 / 2,4 grids (all of arm
(a), 80 runs total) · full MNIST 10-class grid (v10, arm (a), null at MDE
16 pp with a notable QNN-head seed-instability finding) · 10-class diagnostic
· ansatz-depth ablation · M1-vs-Long verification · cross-experiment synthesis
across all four binary grids.

Next (prioritised six-step plan, `logs/FMNIST_LOG.md` §9): save per-image
predictions for McNemar's test · use the full 2,000-image Fashion-MNIST test
set · enable deterministic algorithms to shrink the non-determinism floor ·
pilot a better difficulty ladder before the next grid · [10-class multiclass
readout now fixed in v10] · run McNemar on the v10 predictions once raw
result JSONs are available. Also outstanding: Long-faithful arm (b) for all
three Fashion pairs and for 10-class; v8/v9/v10 code partially checked into
`code/` (v10 code is in; v8/v9 code files still missing, only results).
