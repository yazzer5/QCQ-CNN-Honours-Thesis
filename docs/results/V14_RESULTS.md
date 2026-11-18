# v14 results: low data (sample efficiency), M1–M5 at full readout (w15)

**Status:** complete. 512 new runs (53 smoke + 459 grid), every integrity gate passed, both reproductions bit-exact. Pre-registration: `PREREGISTRATION_v14.md` (no amendments).
**Date:** 2026-10-17.
**Figures:** `results/results_v14/analysis/plots/` (standard set) and `plots_deep/` (D1–D5, `code/v14_lowdata/scripts/deep_v14.py`; exploratory). Data: `results/results_v14/`. Code: `code/v14_lowdata/`.

## 1. Answer in one paragraph

**Shrinking the training set from 500 to 50 images per class gives the quantum front-end no advantage at all.** The trained quantum front-end (M3) never beats no front-end (M4). At N = 50 its point estimate is 1.5 pp (K=5) and 1.7 pp (K=10) *below* M4, and the 95% CI rules out an M3 advantage larger than **+0.8 pp (K=5)** and **+1.1 pp (K=10)**. The front-end effect does not change with data size (slope null). **What does change with data size is the quantum head.** Its cost against the classical head grows from 2.3 pp at N = 500 to **13.5 pp (K=5) and 17.1 pp (K=10) at N = 50**, about 3.5–4.4 pp per halving of the data (p < 10⁻⁴). The cost is almost entirely a generalisation gap: both heads fit the training set equally well. **So the quantum head is sample-*in*efficient even at full readout, and the quantum front-end is neither sample-efficient nor harmful beyond about 1 pp.**

## 2. Integrity

- **Verify** (`verify_v14.log`): M1–M5 bit-identical to the pre-patch code at w3 and w15, K=5 and 10. The w15 parameter table matches v13: K=5 1680/1686/1725/1719/1725, K=10 1760/1766/1805/1799/1805 (M1–M5). The split audit is invisible to training (audit on/off: identical predictions). There is no LR scheduler, so the scaled epoch cap is only a cap.
- **Reproductions (N=500, w15, K=5 A, seed 42):** M3 = v13 and M5 = v13.5, exact on test_acc, best/stop epoch and all 1000 predictions. Pooling stored N=500 M2/M3/M5 with v14 is valid.
- **Gates:** smoke 53/53, grid 459/459. Every model at each (K, subset, N, seed) saw identical train/val/test images (108 groups). One fixed test set per (K, subset). No continuations, no resubmissions. **0 runs at their epoch cap**: early stopping ended every run (latest stop in the smoke: epoch 126, cap 2000).
- **Split facts (recorded before results):** at N = 50 the train class counts range 42–57 (K=5) and 39–55 (K=10), from the seed-dependent draw. Val changes with N; test is fixed. All models at a seed share the draw, so every contrast is paired on it.

## 3. Accuracy (%), mean ± seed SD, n = 10 per cell

| K | N | M1 | M2 | M3 | M4 | M5 |
|---|---|---|---|---|---|---|
| 5 pooled | 50 | 66.8 ± 3.3 | 68.0 ± 4.8 | 81.5 ± 2.9 | **82.9 ± 1.5** | 79.8 ± 4.7 |
| | 100 | 74.8 ± 3.5 | 75.8 ± 3.6 | 85.5 ± 1.3 | **86.9 ± 0.9** | 84.1 ± 4.1 |
| | 250 | 85.5 ± 2.4 | 86.7 ± 2.2 | 90.8 ± 0.9 | **91.4 ± 0.6** | 89.5 ± 3.1 |
| | 500 | 90.0 ± 1.8 | 90.7 ± 1.6 | 92.9 ± 0.5 | **93.4 ± 0.7** | 91.4 ± 2.6 |
| 10 | 50 | 45.0 ± 4.6 | 47.0 ± 6.5 | 64.1 ± 3.9 | **65.9 ± 2.5** | 62.1 ± 7.5 |
| | 100 | 55.1 ± 3.2 | 58.3 ± 4.2 | 69.0 ± 3.7 | **71.9 ± 2.5** | 66.5 ± 7.6 |
| | 250 | 70.1 ± 2.5 | 72.5 ± 1.9 | 77.8 ± 1.9 | **78.7 ± 1.8** | 74.5 ± 4.9 |
| | 500 | 78.7 ± 1.4 | 79.2 ± 2.2 | 81.6 ± 1.5 | **81.6 ± 2.2** | 78.5 ± 3.8 |

M4 (no front-end) has the highest mean in all 8 rows. N=500: M1, M4 from v14; M2, M3 from v13 w15; M5 from v13.5 w15. Per-subset tables: `v14_summary.csv`.

## 4. Pre-registered primary contrasts (Holm within K, 4 tests)

| Contrast | K=5 pooled | K=10 |
|---|---|---|
| **F@50** = M3 − M4 | −1.47 [−3.77, +0.82], p_holm 0.36, MDE 3.2 | −1.74 [−4.54, +1.05], p_holm 0.50, MDE 3.9 |
| **T@50** = M3 − M5 | +1.64 [−0.32, +3.60], p_holm 0.30, MDE 2.7 | +2.08 [−1.70, +5.85], p_holm 0.50, MDE 5.3 |
| **R@50** = M5 − M4 | −3.11 [−6.62, +0.39], p_holm 0.30, MDE 4.9 | −3.82 [−8.87, +1.23], p_holm 0.48, MDE 7.0 |
| **slope(F)**, pp per doubling of N | +0.34 [−0.24, +0.93], p_holm 0.36, MDE 0.8 | +0.66 [−0.33, +1.65], p_holm 0.50, MDE 1.4 |

**None is detected, and none passes TOST ±1.** For F the useful bound is one-sided: an advantage of M3 over M4 at N = 50 larger than **+0.82 pp (K=5) / +1.05 pp (K=10)** is excluded at 97.5%. The slope point estimates are positive, meaning F is, if anything, *more negative* at low N. The trend runs opposite to a sample-efficiency advantage.

## 5. Secondary contrasts (uncorrected; full table in `v14_contrasts.csv`)

| Contrast | K=5 pooled: N = 50 / 100 / 250 / 500 | K=10: N = 50 / 100 / 250 / 500 |
|---|---|---|
| F = M3 − M4 | −1.47 / −1.41 / −0.57 / −0.49 (TOST ✓ at 500) | −1.74 / **−2.89** / −0.87 / −0.01 |
| AULC(F) | **−0.99** [−1.97, −0.00], p = .050 | **−1.58** [−2.93, −0.23], p = .027 |
| T = M3 − M5 | +1.64 / +1.38 / +1.34 / **+1.59** | +2.08 / +2.50 / **+3.32** / **+3.08** |
| slope(T) | −0.02, MDE 0.3 | +0.35, MDE 0.8 |
| R = M5 − M4 | −3.11 / −2.79 / −1.91 / **−2.07** | −3.82 / −5.39 / **−4.19** / **−3.09** |
| **BE = M3 − M2** | **+13.52 / +9.69 / +4.15 / +2.25** | **+17.14 / +10.75 / +5.33 / +2.37** |
| **slope(BE)** | **−3.50** [−4.18, −2.82], p = 1×10⁻⁶ | **−4.40** [−5.79, −3.01], p = 5×10⁻⁵ |
| Q = M2 − M1 | +1.15 / +0.95 / +1.12 / +0.70 (all null) | +2.05 / +3.17 / **+2.42** / +0.54 |

Bold = p < .05 uncorrected (not Holm). The BE values at N = 500 are v13's C contrast, reproduced (2.25 / 2.37). Every BE cell is detected on each subset alone too (A and B, all N).

## 6. Findings

### L1. No sample-efficiency advantage from the quantum front-end (primary)

F is never detected, has negative point estimates at every N and K (8/8), and its slope is null. The one-sided bound at N = 50 excludes an M3 advantage above about 1 pp. Across the whole learning curve (AULC), M3 trails M4 by about 1–1.6 pp (secondary, p ≈ .03–.05 uncorrected). At full data F is equivalent to 0 within ±1 pp at K=5 (TOST p = .041), replicating v12 F2 at w15.

**This retires the last clean-data condition where a front-end advantage could appear.** The earlier project hypothesis that QCQ-CNNs are more sample-efficient is not supported for the front-end.

### L2. The quantum head's cost grows steeply as data shrinks (secondary, strongest effect in v14)

BE rises monotonically as N falls, at both K and on both subsets: 2.3 → 4.2 → 9.7 → 13.5 pp (K=5) and 2.4 → 5.3 → 10.8 → 17.1 pp (K=10). That is 3.5–4.4 pp per halving of the data, p ≤ 5×10⁻⁵. M1 and M2, both quantum heads with different front-ends, sit together at every N (Q null everywhere except K=10 N=250). So this is a head effect, not a front-end one.

### L3. That cost is generalisation, not fitting (exploratory; Figs. D1–D3)

At the best checkpoint both heads fit the training set (K=5: 98.5–99.6%; K=10: 90.8–95.5%). The quantum head generalises much worse:

| | M3 − M2 test (= BE) | of which: train-fit difference | of which: gap difference |
|---|---|---|---|
| K=5 N=50 | 13.5 | 1.1 | **12.4** (p = 2×10⁻⁷) |
| K=5 N=100 | 9.7 | 0.6 | **9.1** |
| K=5 N=250 | 4.2 | 0.3 | **3.9** |
| K=10 N=50 | 17.1 | 1.5 | **15.6** (p = 2×10⁻⁵) |
| K=10 N=100 | 10.8 | 3.3 | **7.4** |
| K=10 N=250 | 5.3 | 1.1 | **4.3** |

This extends v13 V3 (the w15 residual is a generalisation gap) along the data axis. The gap the quantum head adds **scales with data scarcity**. It fits v13's hypothesis that the 12 trig features of the ZZ encoding form a high-frequency feature map with a poor inductive bias: easy to fit, hard to generalise from, and worse the fewer examples constrain it.

### L4. Training the front-end helps a little, and the help doesn't depend on data size

T is positive in 8/8 pooled cells (+1.4 to +1.6 pp at K=5, +2.1 to +3.3 pp at K=10), with a slope near 0 at both K. Training the front-end is not more valuable when data is scarce.

### L5. Seed 46 replicates exactly (exploratory; Figs. D4, D5)

Seed 46's frozen front-end is the worst M5 in **all 9** new low-data cells (mean within-cell z = −2.39; v13.5 found −2.3 at N = 500). M3 at seed 46 is unremarkable. Without seed 46:
- At K=5, T falls to +0.4 to +0.8 pp.
- At K=10 N=250, T = +2.0 [90% CI 1.0, 3.1] remains, as at K=10 in v13.5.

Same reading as v13.5 R3: most of the training benefit is a rescue of one bad random initialisation. A real ~2 pp gain appears only at 10 classes.

### L6. A frozen random front-end hurts, and freezing makes results seed-dependent

R is negative in 8/8 cells (−1.9 to −5.4 pp), detected uncorrected at K=5 N=500 and K=10 N=250/500. M5 is the least stable model in every cell: seed SD 4.7 (K=5) and 7.5 pp (K=10) at N = 50, against M4's 1.5 and 2.5. This replicates v13.5 R2 and R4 across all data sizes.

### L7. Convergence speed (exploratory, descriptive)

The median best epoch is lower for M3 than M4 in all 6 low-data cells, for example K=10 N=50: 64 vs 109 epochs. This matches the earlier v12 observation that the quantum front-end reaches its best checkpoint in fewer epochs. It doesn't buy anything: median wall time per run is 2–2.5× M4's (simulation cost), and final accuracy is lower.

## 7. Pre-registered predictions: verdicts

| Prediction | Verdict |
|---|---|
| P1: F ≈ 0 (TOST) at N = 250, 500; F and slope(F) not detected at low N | **Main claim holds**: F and slope(F) not detected anywhere. Equivalence shown only at K=5 N=500; elsewhere underpowered (MDE 1.2–2.2 pp) with negative point estimates. |
| P2: T ≥ 0 at every N | **Holds**: 8/8 pooled cells positive. |
| P3: R ≤ 0 at every N | **Holds**: 8/8 negative. |
| P4: BE > 0 at every N, slope(BE) < 0 | **Strongly confirmed**: every cell, both subsets, slope p ≤ 5×10⁻⁵. |
| P5 (exploratory): seed SD rises as N falls; M2 SD within 2× M3 | **Mostly.** SD rises with less data for all models except M4 at K=10 (flat, about 2–2.5). M2 within 2× M3 at every N at K=10, but only at N=50 at K=5 (2.4–3.2× at N=100–500). |

Outcome per pre-registration §7: **"F not detected at every N, slope null → no sample-efficiency benefit from the quantum front-end down to 50 images per class (bounded at about +1 pp)"**, and **"BE grows as N falls → the quantum head is sample-inefficient even at full readout."**

## 8. What it means for the thesis

1. **Front-end: the clean-data story is closed.** Across class count (v12), readout width (v13), training vs freezing (v13.5) and data size (v14), the trained quantum front-end never beats having no front-end, and at low data it is bounded to within about +1 pp of it. Noise (v15) is the only remaining condition.
2. **Back-end: v14 gives the strongest result in the thesis.** At full readout, the quantum head's cost is not a fixed ~2.3 pp. It grows to 13–17 pp when data is scarce, and it is a generalisation failure, not a capacity one. This turns v13's "encoding inductive bias" hypothesis into a measured, dose-dependent effect.
3. **Training the front-end** helps by ~1.5–3 pp regardless of data size, mostly by rescuing bad initialisations. Freezing it makes models seed-fragile.
4. Without M6 (a matched classical front-end), "no front-end" is still the comparator. Since M3 doesn't beat even that, M6 is now of low priority for the front-end question.

### Supervisor summary

> With 50 to 500 training images per class, the trained quantum front-end never outperforms a model with no front-end. At 50 images per class it is 1.5–1.7 pp behind (not significant), and any advantage above about 1 pp is ruled out. Its effect does not change with data size, so there is no sample-efficiency benefit. The quantum head, even with full Pauli readout, becomes much worse as data shrinks: its deficit to a classical head grows from 2.3 pp at 500 per class to 13.5 pp (5 classes) and 17 pp (10 classes) at 50 per class. Both heads fit the training data equally well, so this is a generalisation failure that scales with data scarcity. Training the front-end, rather than freezing it, helps by 1.5–3 pp at every data size, mostly by rescuing one bad random initialisation (seed 46, the worst frozen model in every cell, as in v13.5).

## 9. Caveats

- L3, L5, L6 (stability) and L7 are exploratory, found after reading the results, and are not substituted for the pre-registered contrasts.
- Holm is applied within each K over the 4 primary tests. The secondary contrasts are many and uncorrected; read them as patterns (sign consistency across N, subsets and K), not individual tests.
- The train/test fit at N = 500 for the stored M2/M3/M5 runs isn't in this archive. L3's decomposition covers N = 50–250.
- MNIST only. The val set changes with N, and train class counts vary ±15% at N = 50 (paired across models).

## 10. Next

- **v15 (noise):** IBM-calibrated noise on the front-end. It is now the only open front-end condition.
- **Optional, cheap, strengthens L3:** the classical-replica control from v13's follow-up list. A classical head computing the same 12 trig features of fc2's outputs, run at N = 50 and 500. If it matches M2's low-data collapse, the sample inefficiency belongs to the feature map, with nothing specifically quantum in it.
- Results chapter: lead with `contrasts_vs_n.png` (BE rising, F flat below 0), then `learning_curves.png` and D2.

## 11. Figure map

**Standard set (`results/results_v14/analysis/plots/`)**

| Figure | Shows | Finding |
|---|---|---|
| `learning_curves.png` | Accuracy vs N (log x), M1–M5, 95% CI | Two bands: classical heads (M3, M4, M5) on top, quantum heads (M1, M2) below; the bands converge as N grows. M4 is highest at every N. |
| `contrasts_vs_n.png` | F, T, R, BE vs N, 95% CI, ±1 pp band | **Headline.** BE falls steeply with N (L2); F stays just below 0 (L1); T just above 0, R below 0 (L4, L6). |
| `bars_k5.png`, `bars_k10.png` | Mean accuracy per N and model | Same as §3, as bars. |
| `strip_k5.png`, `strip_k10.png` | Every seed, per subset | The spread of M1/M2/M5 at low N; M5's isolated low point is seed 46. |
| `paired_k5.png`, `paired_k10.png` | Per-seed F, T, R at every N | F's seeds cluster just below 0; one large T/R outlier per cell (seed 46). |
| `effect_vs_mde.png` | \|effect\| vs MDE, every contrast | All primary points sit below the line: none resolvable at n = 10. BE points sit far above it. |
| `seed_sd_vs_n.png` | Seed SD vs N per model | M5 the least stable everywhere; M4 the most stable; SD rises as N falls (P5). |

**Deep set (`results/results_v14/analysis/plots_deep/`, exploratory)**

| Figure | Shows | Finding |
|---|---|---|
| D1 `D1_fit_vs_gap.png` | Dumbbell per model and N: train acc at best checkpoint (hollow) to test acc (filled); bar length = generalisation gap, labelled in pp | Train accuracy is compressed (98.5–99.6% K=5); test accuracy spreads 67–93%. Quantum-head bars 31–32 pp (K=5) and 47–49 pp (K=10) at N=50, classical 17–20 and 26–32. The models differ in generalisation, not fitting (L3). At K=10, M3 fits more than M4 (93.4–95.5 vs 91.4–92.2) yet tests lower (L1). |
| D2 `D2_be_decomposition.png` | M3 − M2 split into fitting and gap differences | Gap difference 12.4 / 9.1 / 3.9 pp (K=5) and 15.6 / 7.4 / 4.3 (K=10) at N = 50/100/250; fitting difference 0.3–3.3 pp (L3). |
| D3 `D3_gap_vs_n.png` | Train − test gap vs N per model | Same order at every N: M4 < M3 < M5 < M2 < M1. Quantum-head gaps 30–49 pp at N = 50, classical 17–32 (L3, L1). |
| D4 `D4_seed46_m5.png` | M5 within-cell z-score, seed × cell | Seed 46 is z = −2.0 to −2.7 in all 9 cells; no other seed below −1.4 (L5). |
| D5 `D5_t_without46.png` | T with and without seed 46, 95% CI | K=5: T drops from ~1.4–1.6 to 0.4–0.8 pp. K=10 N=250: 2.0 [0.8, 3.3] remains (L5). |
