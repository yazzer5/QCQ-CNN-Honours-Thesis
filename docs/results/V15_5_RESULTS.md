# v15.5 results: noisy input images (Gaussian pixel noise), M1–M5 (w15, N = 500/class)

**Status:** complete. 750 runs (75 smoke + 675 grid). Every integrity gate passed, and all 150 σ = 0 runs reproduce the stored v13 / v13.5 / v14 runs bit-exactly. Pre-registration: `PREREGISTRATION_v15_5.md` (no amendments).
**Date:** 2026-10-21.
**Figures:** `figures_v155/` (standard set) and `figures_v155_deep/` (D0–D8, `scripts/deep_v155.py`, exploratory).

## 1. Answer in one paragraph

**With noisy input images, the trained quantum front-end is more robust than no front-end. This is the first condition in v12–v15.5 where M3 beats M4, and it is the opposite of my pre-registered prediction.** Trained on clean images and tested at pixel noise σ = 0.3, M3 is **+9.1 pp** above M4 at K = 5 (95% CI +5.5 to +12.7, Holm p = .001), and the gap grows with noise (slope +2.3 pp per 0.1 σ, Holm p = .018). At K = 10 the deployed gap is +3.6 pp, not detected after Holm. Trained *on* noisy images, M3 also beats M4: +1.0 pp at K = 5 (Holm p = .018) and **+4.1 pp at K = 10** (Holm p = .008), growing to +2.5 / +7.3 pp at σ = 0.5. The robustness needs the front-end to be **trained**. A frozen random Circuit 4 (M5) is no better than no front-end (R ≈ −1 to −8 pp, not detected). Training it gives +13 / +12 pp in the deployed arm (T, Holm p = .027 at K = 5). **The quantum head goes the other way:** with the same front-end, the QNN head (M2) loses 18 pp more than the classical head at σ = 0.3 when trained clean, as in v15. A first-order check (exploratory) suggests why the front-end is robust: the RY(πx) encoding is flat at x = 0 and x = 1, where 92% of MNIST pixels sit, so it suppresses small pixel perturbations. That is a property of a cosine feature map and can be reproduced classically, **so this is "more robust than no front-end", not a quantum advantage.**

## 2. Integrity

- **Verify** (`verify_v155.log`, ALL CHECKS PASSED):
  - M1–M5 are bit-identical to the v14 file at K = 5 and 10, with V155 unset and set (weights, outputs, gradients, RNG states). Parameter table: K = 5 1680/1686/1725/1719/1725, K = 10 1760/1766/1805/1799/1805.
  - Noise wrapper:
    - test and val noise are fixed and independent of the seed
    - training noise repeats for the same (seed, σ) and is fresh on every fetch
    - the global RNGs are untouched
    - the statistics match: clipped background mean 0.1199 vs σ/√(2π) = 0.1197 at σ = 0.3
  - Real-data dry runs: σ = 0 is invisible to training (identical predictions), and every cross-eval is MATCH.
- **Gates:** smoke 75/75, grid 675/675.
  - Every model and σ at a (K, subset, seed) saw identical clean images (27 groups).
  - One noisy test set per (K, subset, σ): 15 cells. One per cross-evaluation level: 10 levels × 3 test sets. Every contrast is paired on the same corrupted images.
  - Every cross-evaluation reproduced its job's saved predictions (750/750). No continuations.
- **Reproductions (σ = 0):** the K = 5 A seed-42 gate is exact for all five models (20/20 checks), plus 10/10 other smoke cells and 135/135 grid runs, so **150/150 in total**. Every σ = 0 number here is the same run as v13 (M2, M3), v13.5 (M5) or v14 (M1, M4), and as v15's α = 0.
- **Epoch cap:** no smoke run hit the 200-epoch cap; the latest smoke stop was 165. In the grid, **4 of 675 runs hit the cap, all seed 46**: M5 at K = 5 A σ = 0.3 and K = 10 σ = 0.2, M4 at K = 5 A σ = 0.5 and K = 10 σ = 0.5. Best epochs were 181–194. Each cell has 1/9 at the cap, below the pre-registered "budget-limited" threshold of > 2/9. Those four runs may be slightly under-trained. They are reported, not excluded.

## 3. Accuracy (%), mean ± seed SD, n = 10 per cell

**Matched arm (trained, validated and tested at σ):**

| K | σ | M1 | M2 | M3 | M4 | M5 |
|---|---|---|---|---|---|---|
| 5 pooled | 0 | 90.0 ± 1.8 | 90.7 ± 1.6 | 92.9 ± 0.5 | **93.4 ± 0.7** | 91.4 ± 2.6 |
| | 0.1 | 92.2 ± 0.7 | 92.4 ± 1.2 | **92.8 ± 0.6** | 92.6 ± 0.6 | 91.3 ± 2.5 |
| | 0.2 | 92.4 ± 0.9 | 92.7 ± 0.8 | **92.7 ± 0.9** | 92.5 ± 0.6 | 91.5 ± 2.9 |
| | 0.3 | 91.5 ± 1.0 | 92.5 ± 1.1 | **92.6 ± 0.4** | 91.6 ± 0.8 | 90.4 ± 3.2 |
| | 0.5 | 84.1 ± 3.5 | 89.3 ± 1.2 | **89.4 ± 0.8** | 86.9 ± 1.2 | 85.1 ± 5.8 |
| 10 | 0 | 78.7 ± 1.4 | 79.2 ± 2.2 | 81.6 ± 1.5 | **81.6 ± 2.2** | 78.5 ± 3.8 |
| | 0.1 | 81.4 ± 1.9 | **81.8 ± 1.4** | 81.7 ± 2.9 | 80.5 ± 1.7 | 79.4 ± 3.7 |
| | 0.2 | 81.9 ± 1.4 | **83.3 ± 0.9** | 81.7 ± 2.3 | 81.1 ± 2.2 | 79.1 ± 3.9 |
| | 0.3 | 79.3 ± 2.8 | 81.6 ± 1.6 | **81.8 ± 2.3** | 77.7 ± 2.0 | 77.1 ± 5.6 |
| | 0.5 | 67.4 ± 3.9 | 73.7 ± 1.2 | **74.1 ± 2.5** | 66.8 ± 4.1 | 66.9 ± 7.3 |

**Deployed arm (trained clean, tested at σ):**

| K | σ_test | M1 | M2 | M3 | M4 | M5 |
|---|---|---|---|---|---|---|
| 5 pooled | 0.1 | 88.5 | 89.4 | **92.4** | 92.2 | 88.9 ± 7.7 |
| | 0.2 | 77.9 | 81.0 | **90.0** | 85.1 | 80.9 ± 15.5 |
| | 0.3 | 59.1 | 64.5 | **82.5 ± 3.3** | 73.4 ± 5.6 | 69.4 ± 18.3 |
| | 0.5 | 33.9 | 38.1 | **57.9** | 48.5 | 48.3 |
| | 0.7 | 27.0 | 29.0 | **41.3** | 36.2 | 37.3 |
| | 1.0 | 23.7 | 25.1 | **31.6** | 29.6 | 29.6 |
| 10 | 0.1 | 76.9 | 76.7 | **80.9** | 78.4 | 74.3 ± 10.2 |
| | 0.2 | 66.4 | 63.7 | **74.6** | 70.0 | 63.1 ± 16.1 |
| | 0.3 | 46.1 | 43.3 | **61.1 ± 6.7** | 57.5 ± 4.6 | 49.5 ± 15.7 |
| | 0.5 | 22.6 | 21.9 | **34.2** | 33.1 | 28.4 |
| | 0.7 | 16.0 | 15.6 | **22.1** | 21.6 | 19.1 |
| | 1.0 | 13.0 | 12.7 | 15.9 | **16.0** | 14.4 |

σ = 0.7 and 1.0 are stress levels (deployed arm only). Full tables: `v155_summary.csv`; train-σ × test-σ transfer: `v155_transfer.csv`.

## 4. Pre-registered primary contrasts (Holm within K, 4 tests)

| Contrast (σ = 0.3 unless slope) | K = 5 pooled | K = 10 |
|---|---|---|
| **F_dep@0.3** = M3 − M4, trained clean | **+9.12** [+5.51, +12.73], p_holm **.001**, MDE 5.0 | +3.59 [−0.33, +7.52], p = .068, p_holm .145, MDE 5.5 |
| **slope_dep(F)**, pp per 0.1 σ | **+2.27** [+0.83, +3.71], p_holm **.018**, MDE 2.0 | +0.15 [−0.92, +1.22], p_holm .76, MDE 1.5 |
| **F_mat@0.3** = M3 − M4, trained at σ | **+0.98** [+0.35, +1.61], p_holm **.018**, MDE 0.9 (TOST ±1: no) | **+4.07** [+1.94, +6.19], p_holm **.008**, MDE 3.0 |
| **T_dep@0.3** = M3 − M5, trained clean | **+13.16** [+1.88, +24.45], p_holm **.027**, MDE 15.7 | +11.58 [+0.11, +23.05], p = .048, p_holm .145, MDE 16.0 |

**All four primaries are detected at K = 5, and F_mat is detected at K = 10.** Every primary point estimate is positive at both K.

- **K = 10, deployed arm:** F_dep and T_dep are positive but not detected after Holm, and slope_dep(F) is null with MDE 1.5.
- **Per subset at K = 5:**
  - F_dep@0.3 is +11.8 (A) and +6.4 (B), both detected.
  - F_mat@0.3 is +1.11 (A, detected) and +0.85 (B, n.s.).
  - T_dep@0.3 is +13.7 / +12.7, both detected.
  - slope_dep(F) is +2.8 (A, detected) and +1.7 (B, n.s.).
  - Same sign on both subsets for all four.
- **Per seed (exploratory):** F_dep@0.3 is positive in 9/10 seeds at K = 5 and 8/10 at K = 10. F_mat@0.3 is positive in 8/10 and 9/10.
- **Without seed 46 (exploratory):** F_dep +9.6 (K = 5) / +3.5 (K = 10); F_mat +1.04 / +3.93. The F result is not a seed-46 effect.

## 5. Secondary contrasts (uncorrected; full table in `v155_contrasts.csv`)

**Deployed arm (σ_test = 0.1 / 0.2 / 0.3 / 0.5 / 0.7 / 1.0):**

| Contrast | K = 5 pooled | K = 10 |
|---|---|---|
| F = M3 − M4 | +0.22 (TOST ✓) / **+4.96 / +9.12 / +9.45** / +5.12 / +2.02 | **+2.55 / +4.57** / +3.59 / +1.17 / +0.58 / −0.16 |
| T = M3 − M5 | +3.49 / +9.14 / **+13.16 / +9.62** / +4.01 / +2.01 | +6.69 / +11.46 / **+11.58** / +5.80 / **+3.03** / +1.41 |
| R = M5 − M4 | −3.27 / −4.18 / −4.04 / −0.16 / +1.11 / 0.00 | −4.14 / −6.89 / −7.99 / −4.63 / −2.45 / −1.57 |
| BE = M3 − M2 | **+3.05 / +9.01 / +18.09 / +19.82 / +12.27 / +6.53** | **+4.23 / +10.87 / +17.75 / +12.31 / +6.54 / +3.19** |
| Q = M2 − M1 | +0.83 / +3.08 / **+5.38 / +4.14** / +2.02 / +1.38 | −0.24 / −2.67 / −2.75 / −0.72 / −0.37 / −0.36 |

**Matched arm (σ = 0.1 / 0.2 / 0.3 / 0.5):**

| Contrast | K = 5 pooled | K = 10 |
|---|---|---|
| F | +0.19 (TOST ✓) / +0.29 / **+0.98 / +2.54** | +1.13 / +0.67 / **+4.07 / +7.32** |
| T | +1.53 / +1.27 / +2.19 / **+4.28** | +2.23 / +2.60 / **+4.71 / +7.20** |
| R | −1.34 / −0.98 / −1.21 / −1.74 | −1.10 / −1.93 / −0.64 / +0.12 |
| BE | +0.46 / +0.02 / +0.09 / +0.10 | −0.11 / −1.51 / +0.15 / +0.41 |
| Q | +0.17 / +0.32 / **+1.02 / +5.21** | +0.34 / **+1.32 / +2.32 / +6.28** |

**Noise costs:**

| | K = 5 pooled | K = 10 |
|---|---|---|
| NC_m (matched, σ = 0.1 / 0.2 / 0.3 / 0.5) | M1 **+2.2 / +2.4 / +1.5 / −5.9**; M2 **+1.7 / +2.0 / +1.8 / −1.4**; M3 −0.1 / −0.2 / −0.3 (TOST ✓ ×3) / **−3.5**; M4 **−0.8 / −1.0 / −1.8 / −6.6**; M5 −0.1 / +0.1 (TOST ✓) / −0.9 / **−6.2** | M1 **+2.8 / +3.2** / +0.6 / **−11.3**; M2 **+2.6 / +4.0 / +2.4 / −5.6**; M3 +0.1 / +0.1 / +0.2 / **−7.5**; M4 −1.1 / −0.5 / **−3.9 / −14.8**; M5 +0.9 / +0.6 / −1.5 / **−11.6** |
| DEP_m at σ = 0.3 (trained clean) | M1 **−30.9**, M2 **−26.2**, M3 **−10.4**, M4 **−20.0**, M5 **−22.0** | M1 **−32.6**, M2 **−35.9**, M3 **−20.5**, M4 **−24.2**, M5 **−29.0** |
| REC_m at σ = 0.3 | M1 **+32.4**, M2 **+28.1**, M3 **+10.1**, M4 **+18.2**, M5 **+21.0** | M1 **+33.2**, M2 **+38.3**, M3 **+20.7**, M4 **+20.2**, M5 **+27.6** |
| slopes (pp per 0.1 σ) | matched: F **+0.59**, T **+0.55**, BE **−0.35**, Q **+0.92**; deployed: T **+1.89**, BE **+4.03**, R null | matched: F **+1.50**, T **+0.94**, Q **+1.19**, BE null; deployed: BE **+2.45**, T, R, Q null |

Bold = p < .05 uncorrected (not Holm).

## 6. Findings

### L1. A trained quantum front-end makes the model more robust to noisy images than no front-end (primary)

This is the first detected F > 0 in the thesis.

- **Deployed arm:** M3 loses 10.4 pp at σ = 0.3 (K = 5), against 20.0 pp for M4. The gap opens from σ = 0.2, peaks at σ = 0.3–0.5 (+9 pp), and closes again at σ ≥ 0.7, where every model approaches chance. At K = 10 the deployed advantage is smaller (+2.6 to +4.6 pp at σ = 0.1–0.3, detected uncorrected) and not detected after Holm.
- **Matched arm:** M3's own accuracy is flat to σ = 0.3 (TOST equivalent to its clean accuracy at K = 5), while M4 loses 1.8 pp (K = 5) and 3.9 pp (K = 10). At σ = 0.5, M3 leads M4 by +2.5 pp (K = 5) and +7.3 pp (K = 10).
- **Both subsets and almost every seed agree** (§4). The transfer matrix (D2, exploratory) shows the same at every train/test combination: F > 0 in 34 of 35 (train σ, test σ) cells at K = 5 and 27 of 35 at K = 10. The exceptions are the clean cell and mismatched high-σ cells at K = 10. The pre-registered reading for F > 0 (§7 of the pre-registration) says to check T and R first, which L2 does.

### L2. The robustness needs training: a frozen random Circuit 4 does not have it (primary T, secondary R)

- **R = M5 − M4 is never detected** and its point estimates are negative in 15 of 16 cells. A frozen random front-end is, if anything, *less* robust than none, and enormously seed-dependent: M5's deployed SD is 15–18 pp at σ = 0.2–0.3.
- **T_dep@0.3 = +13.2 / +11.6 pp, without seed 46: +9.1 / +7.5** (exploratory). The training effect is far larger under noise than on clean data (+1.6 / +3.1 at σ = 0).
- So the robustness is not "any quantum feature map smooths noise". The front-end has to be trained, and training adapts it to images (D5).

### L3. The QNN head amplifies input noise; noise augmentation erases the head's cost (secondary)

M2 and M3 share the Circuit-4 front-end and differ only in the head.

- **Deployed arm:** BE grows from +2.3 pp (clean) to **+18 pp at σ = 0.3** at both K (slope p < 10⁻⁴). M1 and M2, the QNN-head models, are the least robust at every σ ≥ 0.2. This replicates v15 L3 with pixel noise instead of circuit noise.
- **Matched arm:** **BE ≈ 0 at every σ ≥ 0.1** (K = 5: +0.02 to +0.46; K = 10: −1.5 to +0.4), and the QNN-head models *gain* 1.5–4 pp from mild noise (NC_M1, NC_M2 > 0 at σ = 0.1–0.2, both K).
- Training on noisy images removes the ~2.3 pp clean-data head cost (v13). That fits v14 L3, which found the head's cost is a generalisation gap: mild input noise is a regulariser, and the trig feature map is the component that needs it. Found after reading results.

### L4. Why the front-end is robust: the encoding is flat where MNIST pixels are (exploratory; Fig. D0)

The front-end encodes each pixel as RY(πx), and its first observable is ⟨Z⟩ = cos(πx). At x = 0 and x = 1 its slope is zero, and 92% of MNIST test pixels are within 0.1 of 0 or 1. On the 8 stored example digits, at the same [0, 1] scale:

| σ | 0.1 | 0.2 | 0.3 | 0.5 | 0.7 | 1.0 |
|---|---|---|---|---|---|---|
| mean \|Δ\| after M4's 2×2 bilinear resize | 0.038 | 0.078 | 0.115 | 0.188 | 0.243 | 0.300 |
| mean \|Δ cos(πx)\| / 2, then 2×2 mean | 0.015 | 0.049 | 0.091 | 0.178 | 0.240 | 0.300 |

The encoding transmits 40–60% less perturbation at σ = 0.1–0.2 and about 20% less at 0.3. The advantage disappears at σ ≥ 0.5, where noise pushes pixels off the flat ends.

This matches the shape of F_dep: it opens at σ = 0.2, peaks around 0.3–0.5 as M4 collapses, and closes by σ = 0.7–1.0. It is a first-order, single-qubit picture. It ignores Circuit 4's entangling gates and the trained weights, and it does not explain why M5 (same encoding) is not robust (L2).

**What it implies:** this suppression is a property of a saturating periodic feature map. A classical front-end computing cos(πx) per pixel would have it too. **The result should be worded as "more robust than no front-end", never "quantum advantage".**

### L5. Seed 46 (exploratory; Figs. D4, D5)

- Seed 46's frozen front-end is the most noise-fragile model in the grid. Trained clean, M5 seed 46 falls from 84.9 to 41 (σ = 0.2) and 27 (σ = 0.3) at K = 5, while the other seeds are mostly above 80 at σ = 0.2. With training (M3) it is normal.
- All 4 cap hits are seed 46, and two of them are **M4**, which has no front-end. So seed 46's slow training is not only about its circuit initialisation. Its data shuffle or split also trains slowly under strong noise (σ = 0.5 for M4).
- The F primaries barely move without seed 46 (§4).

### L6. Salt-and-pepper corruption gives the same ranking (exploratory; Fig. D7)

Models trained clean and tested on salt-and-pepper noise rank the same as under Gaussian noise. At p = 0.2:
- M3 > M4: F = +8.2 pp (K = 5) and +6.2 pp (K = 10), both p < .01.
- The QNN-head models are again the most fragile: M1 loses 37–44 pp, against 14–24 pp for M3.

Salt-and-pepper sets pixels exactly to 0 or 1, where cos(πx) is flat. Against flipped pixels the encoding has no special protection, but the trained front-end + backbone still degrade less than M4. So L4 is not the whole story.

### L7. Training on noisy images roughly doubles training length (exploratory; Fig. D8)

The median best epoch rises with the training σ for every model:
- K = 5: about 30–37 at σ = 0, rising to 58–80 at σ = 0.5.
- K = 10: about 40–55 at σ = 0, rising to 85–120 at σ = 0.3–0.5.

The noisy validation set makes early-stopping improvements noisier, and the noisy training target is harder. No run in the smoke reached the cap (latest stop 165, against 122 in v15). Four seed-46 grid runs did (§2). A larger replication should raise the epoch cap for σ ≥ 0.3.

## 7. Pre-registered predictions: verdicts

| Prediction | Verdict |
|---|---|
| **P1:** F_dep@0.3 < 0 and slope_dep(F) < 0, detected at both K | **Refuted, opposite sign.** F_dep@0.3 = +9.1 (K = 5, Holm p = .001) and slope_dep(F) = +2.3 (Holm p = .018). At K = 10, +3.6 and +0.15, not detected. My reasoning in the pre-registration had the sign of the effect wrong. I argued the quadratic response of cos(πx) at the background makes the front-end *more* sensitive. But a zero first derivative means small perturbations are *suppressed*, and the clipped background noise is small at σ ≤ 0.3 (L4, exploratory). |
| **P2:** F_mat@0.3 not detected, point ≤ 0 | **Refuted.** F_mat@0.3 = +0.98 (K = 5, Holm p = .018; inside ±1 pp in size, but TOST fails) and **+4.07 (K = 10, Holm p = .008)**. |
| **P3:** T_dep@0.3 ≥ 0; not predicted to survive Holm; seed 46 the worst deployed M5 at σ ≥ 0.2 | **Holds** on the sign (+13.2 / +11.6) and on seed 46 (D4). Holm: survives at K = 5 (p = .027), contrary to "not predicted to survive"; not at K = 10. |
| **P4:** NC_m(0.5) < 0 for all five models, detected for at least M4 and M3; REC_m ≥ 0 at σ ≥ 0.2 | **Holds**: NC_m(0.5) is detected negative for all five models at both K, and REC_m is detected positive for all models at σ ≥ 0.2. **But** the framing "pixel noise costs every model" fails at low σ in the matched arm: M1 and M2 *gain* 1.5–4 pp at σ = 0.1–0.2 (L3). |
| **P5:** slope_dep(BE) > 0; DEP_M1, DEP_M2 the most negative at σ = 0.3 | **Holds.** slope_dep(BE) = +4.0 / +2.5 (p < 10⁻⁴). The QNN-head models lose the most at σ = 0.3 at both K: 26–36 pp, against 10–21 pp for M3. |

Outcome per pre-registration §7: **"F_dep > 0 detected → contradicts P1: the front-end is more robust. Check T and R first (a frozen random circuit doing the same means fixed-feature smoothing, not training), then both subsets."**
- T is positive and R is not detected, so it is not fixed-feature smoothing by any random circuit; training is needed.
- Both subsets agree.
- Worded "more robust than no front-end", never "quantum advantage" (no matched classical front-end).

## 8. What it means for the thesis

1. **The front-end story gains its first positive result, and it is specific.** On clean data (v12–v14) and under circuit noise (v15), a trained Circuit-4 front-end never beats no front-end. On *corrupted images* it does: 9 pp more robust when deployed clean at K = 5, and 1–7 pp better when trained on the noise. Training is necessary; a frozen random circuit is no better than nothing.
2. **The likely mechanism is classical.** RY(πx) encoding plus ⟨Z⟩ readout behaves like a saturating cos(πx) feature map, flat at black and white pixels (L4, exploratory). The thesis should say "the angle-encoded front-end acts as a denoising feature map", and should not claim a quantum advantage until a classical control with the same encoding has been run.
3. **The QNN head story gets a fourth angle:**
   - ~2.3 pp clean cost (v13)
   - growing with data scarcity (v14)
   - amplifying front-end noise (v15)
   - amplifying input noise (v15.5)

   New in v15.5: **its cost disappears with noise augmentation** (L3), which strongly supports the generalisation-gap / regularisation reading.
4. **Wording fixes:**
   - Replace "no front-end advantage in any condition" with "no front-end advantage on clean data or under hardware noise; more robust to corrupted inputs once trained."
   - The project's earlier "noise-robustness advantage" claim is now partly supported, but for **input** noise only, and with a classical mechanism.

### Supervisor summary

> With Gaussian noise added to the input pixels, the trained quantum front-end is more robust than no front-end, the first condition where it wins. Trained on clean images and tested at σ = 0.3, it is 9 pp more accurate than the no-front-end model at 5 classes (Holm p = .001), and the gap grows with noise. Trained on the noisy images, it is 1 pp (5 classes) and 4 pp (10 classes) better. A frozen random version of the same circuit gives no benefit, so the robustness comes from training the front-end. The likely reason is that the angle encoding cos(πx) is flat at black and white pixels and suppresses small pixel noise; that is reproducible classically, so this is robustness, not quantum advantage. The quantum head again amplifies the noise when trained clean (18 pp worse than the classical head at σ = 0.3), but its usual 2 pp cost disappears entirely when it is trained on noisy images.

## 9. Caveats

- **No matched classical front-end.** L1 compares against no front-end (bilinear resize). A classical model with the same cos(πx) encoding, or a learned classical convolutional front-end, may be equally robust. Until that control exists, nothing in L1 is quantum-specific.
- **P1 and P2 were refuted.** L4 is a post-hoc mechanism found after reading the results and is not a test. It is first-order, uses 8 digits, and ignores the entangling gates and the trained weights.
- **K = 10 is weaker in the deployed arm:** F_dep and slope_dep(F) are not detected after Holm.
- **Epoch cap:** 4 seed-46 runs at the 200-epoch cap (1/9 per cell, below the budget-limited threshold).
- **Matched-arm training statistics are on noisy images:** `train_acc_at_best` and the gap use noisy training images.
- **Secondary contrasts** are many and uncorrected; read them as patterns. L3–L7 are exploratory.
- MNIST only, N = 500/class, additive Gaussian noise clipped to [0, 1] (salt-and-pepper exploratory).

## 10. Next

- **The control this result needs (cheap, about 60 jobs):**
  - M4 fed (1 − cos πx)/2 per pixel, with zero parameters added (the classical analogue of the encoding), deployed and matched at σ = 0, 0.3 for K = 5 and 10.
  - Optionally, a trained 2×2 classical convolution front-end with 6 parameters.
  - If the cos-encoded M4 matches M3's robustness, L1 is a feature-map effect with nothing quantum in it.
  - This fits naturally with v16a's classical-replica idea; coordinate with v16 before submitting.
- **Results chapter:** add v15.5 as the sixth condition in the front-end table, with F_dep as the one positive row. Add the noise-augmentation result (L3) to the QNN-head story.

## 11. Figure map

**Standard set (`figures_v155/`)**

| Figure | Shows | Finding |
|---|---|---|
| `deploy_curves.png` | Deployed arm: accuracy vs test σ to 1.0, 95% CI | **Headline.** M3 on top from σ = 0.2; the QNN-head models fall first; M4 in between (L1, L3). |
| `noise_curves.png` | Matched arm: accuracy vs σ | M3 flat to 0.3; M4 declines; M1/M2 rise at 0.1–0.2 (L1, L3). |
| `contrasts_vs_sigma.png` | F, T, R, BE, Q vs σ, both arms, ±1 pp band | F and T positive and peaking at 0.3–0.5 (deployed); BE large deployed, ≈ 0 matched. |
| `recovery.png` | REC_m vs σ | Noise-aware training recovers most for M1/M2, least for M3 (which loses least). |
| `bars_k5.png`, `bars_k10.png` | Both arms, mean ± seed SD | §3 as bars. |
| `strip_k5.png`, `strip_k10.png` | Every seed, both arms | M5's isolated low points are seed 46 (L5). |
| `paired_k5.png`, `paired_k10.png` | Per-seed F, T, R, BE, Q, both arms | F_dep positive in 9/10 seeds (K = 5). |
| `effect_vs_mde.png` | \|effect\| vs MDE | All K = 5 primaries above the line. |
| `seed_sd_vs_sigma.png` | Seed SD vs σ, both arms | M5's deployed SD explodes (15–18 pp) at σ = 0.2–0.3. |

**Deep set (`figures_v155_deep/`, exploratory; `README_deep.md` gives the what and why)**

| Figure | What and why |
|---|---|
| D0 `D0_noise_examples.png` | What the corruption looks like at each σ and salt-and-pepper p, so the accuracy numbers have a picture. |
| D1 `D1_robustness_curves.png` | How fast each model loses accuracy as images get noisier, and how much training on noisy images gives back (L1, L3). |
| D2 `D2_transfer_matrix.png` | Whether training at one noise level protects at others, and whether the front-end changes that (F per cell). |
| D3 `D3_head_amplification.png` | Whether v15's finding (the QNN head magnifies noise) holds for pixel noise: it does (L3). |
| D4 `D4_seed46.png` | Seed 46's frozen front-end is the most noise-fragile; trained, it is normal (L5). |
| D5 `D5_T_without46.png` | How much of T is the seed-46 rescue: T stays large without it (L2). |
| D6 `D6_subset_consistency.png` | The four primaries on subsets A and B, pooled and K = 10: all positive (§4). |
| D7 `D7_salt_and_pepper.png` | A different corruption gives the same ranking (L6). |
| D8 `D8_training_dynamics.png` | Noise augmentation lengthens training (L7). |
