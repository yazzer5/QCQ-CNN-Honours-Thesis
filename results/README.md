# Results

Every `result_*.json` is a complete record of one run: per-epoch history
(train/val accuracy and loss, wall time, gradient variance), the final test
accuracy, the parameter count, and the full protocol under which it ran.

| Directory | Runs | What |
|---|---|---|
| `results_v6_mnist_3v5/` | 20 | MNIST 3-vs-5, 4 models × 5 seeds, α=0 |
| `results_v7_fashion_c06/` | 20 | Fashion-MNIST 0,6, 4 models × 5 seeds |
| `results_v8_fashion_c26/` | 20 | Fashion-MNIST 2,6, 4 models × 5 seeds — corrected gap@best metric |
| `results_v9_fashion_c24/` | 20 | Fashion-MNIST 2,4, 4 models × 5 seeds — closes the arm (a) sequence |
| `results_v10_mnist10/` | 20 | 10-class MNIST — raw JSONs + job logs now in repo; **reduced-scale/feasibility, not locked full protocol** |
| `results_replication/` | — | v1–v2 replication of Long et al.: accuracy table and figures |
| `results_v3_grid/` | 17 | First four-model grid at batch size 1 (MNIST, Fashion, MRI); partial — jobs hit walltime |
| `results_v11_mri/` | 20 | Brain-tumour MRI, 4 models × 5 seeds |
| `results_v12/` | 240 | Class count K = 3/4/5, two digit subsets, 4 models × 10 seeds |
| `results_v13/` | 142 | Readout width w = 6/15 at K = 5 and 10, plus bit-exact reproductions |
| `results_v135/` | 131 | **v13.5**: M5 (M3 with frozen front-end), MNIST K=3/4/5/10 × widths 3/6/15, 10 seeds + 1 M3 reproduction — own README |
| `results_v14/` | 512 | **v14**: low data, M1–M5 at w15, MNIST K=5/10 × N=50/100/250 (+ M1/M4 at 500), 10 seeds + 2 reproductions — own README |
| `results_v15/` | 630 | **v15**: IBM-calibrated (ibm_brisbane) noise on the front end, α = 0–4, cross-evaluated to α = 16 — own README |
| `results_v15_5/` | 750 | **v15.5**: Gaussian pixel noise on the input, σ = 0–0.5, M1–M5, matched and deployed arms |
| `results_v16a/` | 147 | Closed-form classical replica of the QNN head; `z_*` cells hold the head-input dumps (`.npz`); `calib/` the Aer reference |
| `results_v16b/` | 121 | Redesigned quantum head (BFR) against M2–M4 |
| `results_v16bx/` | 42 | Controls for the redesign (M4 trim, BFR with ZZ restored) |
| `results_reps_fashion_mnist_c06/` | 20 | ansatz depth ablation, M1/M2 × reps{1,2} × 5 seeds |
| `results_v10_loss_smoke/` | 2 | v10 loss amendment smoke test: M2 seed 42, BCE vs cross-entropy |
| `results_smoke10_mnist10/` | 6 | 10-class diagnostic — **not iso-parametric** |
| `results_v5_scale1000/`, `results_v5_scale3000/`, `results_v5_smoke/` | 12 | reduced-scale validation |
| `job_logs/` | — | PBS stdout/stderr, including full epoch traces |
| `gpu_traces/` | — | GPU utilisation CSVs |

Each of the five main grids (v6, v7, v8, v9, v10) also has an `analysis/`
subfolder with `plots/` and a standalone `report.html`. Index:
`results/analysis_index.html`.

## Headline numbers

**MNIST 3-vs-5** — M1 99.22 ± 0.26 · M2 99.18 ± 0.23 · M3 99.18 ± 0.13 ·
M4 99.14 ± 0.21. Total spread 0.08 pp; saturated.

**Fashion-MNIST 0,6** — M1 83.98 ± 0.62 · M2 84.60 ± 1.11 · M3 85.26 ± 1.03 ·
M4 85.68 ± 0.91. Adjacent contrasts null (MDE ≈ 1.8 pp); QNN head vs classical
head −1.18 pp, p = 0.011.

**Fashion-MNIST 2,6** — M1 85.70 ± 0.41 · M2 86.74 ± 0.58 · M3 87.70 ± 0.79 ·
M4 87.00 ± 0.42. All attribution contrasts null (MDE 1.6–1.9 pp); M2−M1
+1.04 pp, p = 0.075. M3 is the best model (not M4). Gap@best monotone in
quantum content, M2−M4 p = 0.0040.

**Fashion-MNIST 2,4** — M1 87.30 ± 0.80 · M2 87.00 ± 1.19 · M3 88.42 ± 0.80 ·
M4 87.18 ± 0.40. M2−M1 −0.30 pp (sign flip vs 2,6), below MDE of 2.26 pp.
M3−M4 +1.24 pp nominally significant (p = 0.022) but does not survive Holm
correction and sits below its own MDE. M3 again the best model. Gap@best
monotone in quantum content, M2−M4 p = 0.0027.

**Pooled 2,6 + 2,4** — M2 beat M1 in 7/10 seeds; both grid-level contrasts
remain individually null. The pre-planned difficulty gradient collapsed
between 2,6 and 2,4 (0.18 pp apart in M4 accuracy, inside the ~0.8 pp
non-determinism floor). Full synthesis in `logs/FMNIST_LOG.md` §5.7.

**MNIST 10-class (v10) — reduced-scale/feasibility run, not locked full
protocol** (`docs/AMENDMENT_2026-08-14_v10_reduced_scale.md`) — M1 70.59 ± 5.17
· M2 70.92 ± 5.40 · M3 81.96 ± 2.33 · M4 82.41 ± 1.98. M2−M1 +0.33 pp, p = 0.94,
MDE = 16.08 pp — far underpowered. QNN-head models show 12–15 pp seed spread
vs 5–6 pp for classical-head models; this instability, not the M2−M1 point
estimate, is the primary 10-class finding. Classical head beats QNN head by
11.4 pp overall (p = 2.2×10⁻⁵). Pooled McNemar (10,000 image-predictions per
model) agrees in sign with the accuracy-mean test (M2 > M1, both null,
p = 0.574) — but per-seed McNemar shows 4/5 seeds individually significant
(p < 0.0005) with the sign flipping seed to seed, which pooling hides. Full
write-up in `logs/MNIST_LOG.md` §4b.

**v13.5, M5 control (MNIST K=3/4/5/10)** — training the Circuit-4 front-end
(M3 − M5) helps in all 13 cells: +0.6 pp (K=3) → +0.7 (K=4) → +1.5 (K=5) →
≈ +3 (K=10), detected after Holm at K=10 w6/w15. An untrained front-end is slightly
worse than none (M5 − M4 = −0.5 → −2.6 pp). Exploratory: one bad random
initialisation (seed 46) drives most of the K ≤ 5 effect; K=10 keeps ≈ 2 pp without
it. M5 is 2–6× less seed-stable than M3. Full write-up `docs/V13_5_RESULTS.md`.

**Ansatz depth** — M1 −0.02 pp (p = 0.975), M2 −0.14 pp (p = 0.773). Null.

**10-class diagnostic** — M1 33.5% / M2 29.6% (1 observable) → 66.5% / 72.9%
(3 observables) → M3 95.6% / M4 95.8% (classical heads, ~480 more params).

## Before quoting anything

Check the integrity fields. `protocol.reduced_scale` must be `false` for any
result presented as a full run, and the class pair in `protocol.classes` must
match what was intended — an early PBS bug silently dropped it (`CHANGELOG_v7.md`).

Note also that fixed-seed runs are not bit-reproducible (~0.8 pp of GPU
non-determinism), and that `results_smoke10_mnist10/` is explicitly tagged
`ISO_PARAMETRIC: false`.
