# v15: IBM-calibrated noise on the quantum front-end (MNIST, K = 5/10, w15, N = 500/class)

**Noise is applied to the quantum front-end only**, from the ibm_brisbane calibration
(Qiskit FakeBrisbane snapshot 2025-02-26, sha256 `da9374fa89e6…`), scaled by α
(1 = the calibrated device). The QNN head (M1, M2) is noiseless by design. M4 has no
quantum front-end, so it runs at α = 0 only and the same value is used at every α.

| Folder | Runs | What |
|---|---|---|
| `k5_{A,B}_a{0,0.5,1,2,4}/` | 2 × (50 + 4 × 40) | K=5 subsets A {1,3,5,8,9}, B {0,2,6,8,9}; M1–M5 at α=0, M1/M2/M3/M5 at α>0 |
| `k10_all_a{0,0.5,1,2,4}/` | 50 + 4 × 40 | K=10, all digits |
| `analysis/` | | `v15_{runs,summary,contrasts}.csv`, `collect_v15.log`, `plots/` (standard set), `plots_deep/` (D1–D6, exploratory), `deep_v15.py` |

630 runs in total (63 smoke + 567 grid), seeds 42–51. Each run has two files:

- `result_<M>_mnist_seed<S>_alpha0.0.json`: the trainer's result, with per-image test
  predictions. The `alpha0.0` in the name is the old back-end `--noise_alpha` flag,
  which stays 0. The front-end α is the folder's `a<α>`.
- `v15_xeval_<M>_seed<S>.json`: cross-evaluation of the trained model on the test
  set at α ∈ {0, 0.5, 1, 2, 4, 8, 16}, with per-image predictions. It only counts if
  `selfcheck == "MATCH"`, meaning that at its own training α it reproduces the saved
  predictions exactly. All 630 are MATCH.

Two arms come from one grid:
- **Matched:** trained and tested at α (`test_acc` of the folder's runs).
- **Deployed:** trained at α = 0 and tested at α (the `a0` runs' cross-evaluations).

Job logs: `../job_logs/v15/` (63 smoke `v15s_*`, 567 grid `v15g_*`, plus `patch_v15.log`
and `verify_v15.log`). Code: `code/v15_noise/`. Pre-registration with amendment A1:
`docs/preregistrations/PREREGISTRATION_v15.md`. Write-up: `docs/results/V15_RESULTS.md`. Its figure paths
`figures_v15/` and `figures_v15_deep/` are `analysis/plots/` and `analysis/plots_deep/` here.

**Reproductions.** All 150 α = 0 runs equal the stored runs bit-exactly (test_acc,
best/stop epoch, every prediction): M1/M4 = v14 (N=500, w15), M2/M3 = v13 (w15),
M5 = v13.5 (w15). The v13.5 M5 runs are in `../results_v135/k{5_A,5_B,10_all}_w15/`;
v13 and v14 are not yet in this repo.

## Headline (K=5 pooled over subsets; 10 seeds; Holm within K, 4 primary tests)

| Contrast at α = 1 | K = 5 | K = 10 |
|---|---|---|
| NC3 = M3(1) − M3(0): cost of device noise, trained under it | −0.05 pp, **equivalent (TOST ±1, p = .002)** | +0.10 pp, **equivalent (p = .024)** |
| F = M3 − M4: trained quantum front-end vs none | −0.54 [−1.26, +0.18] | +0.09 [−1.78, +1.95] |
| T = M3 − M5: trained vs frozen front-end | +1.34, p_holm 0.29 | +2.88, p_holm 0.085 |
| slope of F in α | −0.01 pp/α, n.s. | +0.08, n.s. |

Trained under the noise, every model is flat up to 4× device noise, and noise neither
creates nor reveals a front-end advantage. Trained clean and run on noise (deployed
arm), the QNN-head models fail first: at α = 4, M2 loses 7.5 / 13.8 pp vs M3's
0.3 / 3.2 pp with the same front-end and noise (exploratory). Seed 46's frozen front-end is
again the outlier: the worst M5 in every cell, and the only seed that degrades at
device noise.
