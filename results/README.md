# Results

Every `result_*.json` is a complete record of one run: per-epoch history
(train/val accuracy and loss, wall time, gradient variance), the final test
accuracy, the parameter count, and the full protocol under which it ran.

| Directory | Runs | What |
|---|---|---|
| `results_v6_mnist_3v5/` | 20 | MNIST 3-vs-5, 4 models × 5 seeds, α=0 |
| `results_v7_fashion_c06/` | 20 | Fashion-MNIST 0,6, 4 models × 5 seeds |
| `results_reps_fashion_mnist_c06/` | 20 | ansatz depth ablation, M1/M2 × reps{1,2} × 5 seeds |
| `results_smoke10_mnist10/` | 6 | 10-class diagnostic — **not iso-parametric** |
| `results_v5_scale1000/`, `results_v5_scale3000/`, `results_v5_smoke/` | 12 | reduced-scale validation |
| `job_logs/` | — | PBS stdout/stderr, including full epoch traces |
| `gpu_traces/` | — | GPU utilisation CSVs |

## Headline numbers

**MNIST 3-vs-5** — M1 99.22 ± 0.26 · M2 99.18 ± 0.23 · M3 99.18 ± 0.13 ·
M4 99.14 ± 0.21. Total spread 0.08 pp; saturated.

**Fashion-MNIST 0,6** — M1 83.98 ± 0.62 · M2 84.60 ± 1.11 · M3 85.26 ± 1.03 ·
M4 85.68 ± 0.91. Adjacent contrasts null (MDE ≈ 1.8 pp); QNN head vs classical
head −1.18 pp, p = 0.011.

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
