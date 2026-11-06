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
| **M5** | Circuit 4, **frozen** at init | classical head | = M3 (6 fewer trainable) |

Each adjacent contrast isolates one factor:

- **M2 − M1** → front-end trainability
- **M2 − M3** → the quantum back-end
- **M3 − M4** → the quantum front-end
- **M3 − M5** → front-end training, with the circuit held fixed (added in v13.5)

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

**5. Under IBM-calibrated hardware noise, the trained quantum front-end still gives
no advantage, and noise costs nothing once the model trains under it (v15).**
With ibm_brisbane-calibrated noise on the front-end (0.5–4× the device), M3's accuracy
at device noise is equivalent to its noiseless accuracy within ±1 pp (TOST, K=5 and
K=10), and M3 − M4 is not detected at any noise level. Models trained without noise
and then run on noisy hardware are a different story. There, the QNN head is the
fragile part (exploratory): with the same front-end and the same noise, it loses
7.5–13.8 pp at 4× device noise against 0.3–3.2 pp for the classical head. See
`docs/V15_RESULTS.md` and `logs/MNIST_LOG.md` §11.

## Repository layout

```
code/
  v5_validation/     correctness debugging; three silent defects found here
  v6_mnist/          MNIST 3-vs-5 grid; batched torch simulation (214-803x)
  v7_fashion/        Fashion-MNIST 0,6; offline data handling; two launch bugs fixed
  v8_fashion/        Fashion-MNIST 2,6; lazy qiskit import, gap@best metric fix
  v9_fashion/        Fashion-MNIST 2,4; identical to v8 except CODE_VERSION tag
  v1_v4_replication/  v1-v2 Long et al. replication; circuit screen (Circuits 2/4/6)
  v3_v4_first_grid/   first four-model grid at batch size 1 (did not finish; see logs)
  v11_brain_tumor/    brain-tumour MRI grid
  v12_class_count/    class count K = 3, 4, 5 (10 seeds)
  v13_readout_width/  QNN readout width w = 3, 6, 15
  v13_5_m5/         v13.5: M5 (frozen-front-end control) patch, verify, PBS, analysis
  v14_lowdata/      v14: low data (50-500/class), M1-M5 at w15; split audit, PBS, analysis
  v15_noise/        v15: IBM-calibrated front-end noise (ibm_brisbane), Aer-verified
                     noise model, patch, verify, PBS, gate, analysis
  v15_5_input_noise/ v15.5: Gaussian pixel noise on the input, M1-M5, two arms
  v16a_closed_form_head/   closed-form classical replica of the QNN head
  v16b_redesigned_head/    redesigned quantum head (BFR)
  v16bx_redesign_controls/ controls for the redesign
  v10_mnist10/       10-class MNIST grid — **reduced-scale/feasibility run,
                     not the locked full protocol, see caveat below**;
                     3-observable readout, re-padded iso-parametric budget,
                     --deterministic, --save_predictions
  diagnostics/
    smoke10_mnist/   10-class MNIST diagnostic (readout width)
    reps_ablation/   reps=1 vs reps=2 ansatz depth ablation

results/
  results_v6_mnist_3v5/              20 runs — 4 models x 5 seeds
  results_v7_fashion_c06/            20 runs — Fashion 0,6
  results_v8_fashion_c26/            20 runs — Fashion 2,6 (gap@best metric fix)
  results_v9_fashion_c24/            20 runs — Fashion 2,4 (closes arm-a sequence)
  results_v10_mnist10/               20 runs — 10-class, reduced-scale/feasibility
                                      (raw JSONs + job logs now in repo)
  results_replication/               v1-v2 replication summary (xlsx + figures)
  results_v3_grid/                   17 runs — first grid, batch size 1 (partial)
  results_v11_mri/                   20 runs — brain-tumour MRI
  results_v12/                      240 runs — class count K=3/4/5
  results_v13/                      142 runs — readout width
  results_v135/                     131 runs — v13.5 M5 control, K=3/4/5/10 (own README)
  results_v14/                      512 runs — v14 low data, M1-M5 at w15, K=5/10 (own README)
  results_v15/                      630 runs — v15 front-end noise, K=5/10, alpha 0–4,
                                      cross-evaluated to alpha 16 (own README)
  results_v15_5/                    750 runs — v15.5 pixel noise, K=5/10, sigma 0–0.5,
                                      deployed and matched arms
  results_v16a/                     closed-form head replica (+ z dumps, calib)
  results_v16b/                     redesigned head
  results_v16bx/                    redesign controls
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
- **The v10 10-class grid ran reduced-scale, not the locked full protocol.**
  Every job in that grid (`N_TEST=2000, MAX_EPOCHS=200`) is tagged
  `reduced_scale: true` in its result JSON and prints `REDUCED-SCALE /
  FEASIBILITY RUN` in its job log — caught 2026-08-14 on receipt of the raw
  per-run JSONs, after the grid had already been written up as complete.
  Numbers are unchanged and confirmed correct; treat every v10 finding as a
  feasibility result, not a locked confirmation. See
  `docs/AMENDMENT_2026-08-14_v10_reduced_scale.md`.
- **Runs are not bit-reproducible at fixed seed.** GPU float reduction order is
  non-deterministic; identical configurations differ by ~0.8 pp, comparable to
  the between-seed SD of 0.92 pp. Disclosed in `FMNIST_LOG.md` §6.5. (v10 runs
  with `--deterministic` enabled — this applies to v6–v9 only.)
- **Every null is bounded by an MDE.** With 5 seeds the design detects ~1.8 pp on
  Fashion-MNIST and ~0.25–0.40 pp on MNIST. A null is a power statement, never
  evidence of equivalence.
- **The 10-class diagnostic (`results_smoke10_mnist10/`) is not iso-parametric.**
  M3/M4 carry ~480 more parameters than M1/M2 there. It is a diagnostic, not
  a result. (Separate from the v10 grid above, which is iso-parametric but
  reduced-scale.)

## Status

> **Repo coverage:** every experiment from v1 to v16b-ext is in the repo: code, result
> JSONs, analysis and job logs. v16c was analysis only. v15.6 (2026-11-04) is not here
> yet. The results write-ups for v13 and v16a–c are kept in the thesis project notes,
> not yet in `docs/`. Datasets, checkpoints and archive tarballs are not committed (see
> `DATASETS.md`).
>
> **v14 headline:** no sample-efficiency advantage from the quantum front-end down to
> 50 images per class (M3 − M4 never detected, bounded at about +1 pp). The quantum head's
> cost grows from 2.3 pp at 500/class to 13.5 pp (K=5) and 17.1 pp (K=10) at 50/class,
> almost entirely as a generalisation gap. Write-up: `docs/V14_RESULTS.md`.

Complete: MNIST 3-vs-5 grid · Fashion-MNIST 0,6 / 2,6 / 2,4 grids (all of arm
(a), 80 runs total, code for all three now in `code/`) · 10-class MNIST
feasibility run (v10, arm (a), 20/20 clean but reduced-scale — see caveat
above; null at MDE 16 pp with a notable QNN-head seed-instability finding,
and per-image McNemar tests now computed: pooled McNemar agrees in sign with
the accuracy-mean test, but 4/5 individual seeds show highly significant
M1-vs-M2 differences that flip direction seed to seed) · 10-class diagnostic
· ansatz-depth ablation · M1-vs-Long verification · cross-experiment synthesis
across all four binary grids.

Next (prioritised six-step plan, `logs/FMNIST_LOG.md` §9): save per-image
predictions for McNemar's test (done for v10; still needed for the binary
grids) · use the full 2,000-image Fashion-MNIST test set · enable
deterministic algorithms on the binary grids · pilot a better difficulty
ladder before the next Fashion-MNIST grid · **run a locked full-protocol
10-class confirmation grid** (the current v10 result is feasibility-scale
only — this is now the top priority, ahead of further class-pair sweeps).
Also outstanding: Long-faithful arm (b) for all three Fashion pairs and for
10-class.
