# v15 results: IBM-calibrated noise on the quantum front-end, M1–M5 (w15, N = 500/class)

**Status:** complete. 630 runs (63 smoke + 567 grid). Every integrity gate passed, and all 150 α = 0 runs reproduce the stored v13 / v13.5 / v14 runs bit-exactly. Pre-registration: `PREREGISTRATION_v15.md` (amendment A1, resource only, plus the K = 10 SU decision, both logged before the smoke).
**Date:** 2026-10-18.
**Figures:** `figures_v15/` (standard set) and `figures_v15_deep/` (D1–D6, `scripts/deep_v15.py`, exploratory).

## 1. Answer in one paragraph

**IBM-calibrated noise on the quantum front-end costs nothing once the model is trained under it, and it changes none of the front-end conclusions.** At the calibrated device level (α = 1), the trained quantum front-end (M3) is *equivalent* to its own noiseless accuracy within ±1 pp at both K (TOST p = .002 at K = 5, p = .024 at K = 10). The same holds up to 4× device noise. It still does not beat no front-end (F@1 = −0.54 pp at K = 5, +0.09 pp at K = 10, neither detected), and noise does not change that (slope null). **The noise matters only when a model trained without noise is run on noisy hardware, and then the head decides how much.** With the same Circuit-4 front-end and the same noise, the QNN head (M2) loses 7.5 pp (K = 5) and 13.8 pp (K = 10) at 4× device noise, against 0.3 and 3.2 pp for the classical head (M3) (exploratory). **So the shallow quantum front-end is robust to current-hardware noise, but it gives no noise-robustness *advantage*, and the quantum head is the noise-fragile component.**

## 2. Integrity

- **Verify** (`verify_v15.log`):
  - M1–M5 are bit-identical to the v14 file with noise off, K = 5 and 10. The w15 parameter table matches v13/v14: K = 5 1680/1686/1725/1719/1725, K = 10 1760/1766/1805/1799/1805.
  - Noise model vs Qiskit: our native channels equal `NoiseModel.from_backend(FakeBrisbane)`'s SX, ECR and readout to ≤ 2e-15. Circuit 4 and a 4-qubit M1-gate-set circuit equal Aer density-matrix simulation to ≤ 5e-8 at α = 0.5–16.
  - Against a full transpiled ibm_brisbane simulation, the model's noise shift correlates 0.987 with the device's and is 9% smaller (the model is slightly optimistic).
  - Installing the noise leaves weights, parameter counts and all RNG states unchanged. At α = 0 it is invisible to training.
- **Gates:** smoke 63/63, grid 567/567.
  - Every model and α at a (K, subset, seed) saw identical images (27 groups), with one fixed test set per (K, subset).
  - Every job's cross-evaluation reproduced its own saved predictions exactly (630/630, all from the in-memory model).
  - No run hit the epoch cap: latest stop 122, cap 200. No continuations, no resubmissions.
- **Reproductions (α = 0):** K = 5 A seed 42 exact for all five models (the pre-registered gate), plus 10/10 other smoke cells and 135/135 grid runs, so 150/150 in total. Every α = 0 number in v15 is the same run as v13 (M2, M3), v13.5 (M5) or v14 (M1, M4).
- **Calibration:** FakeBrisbane snapshot 2025-02-26, sha256 `da9374fa89e6…`, checked by every job. Gadi's `qcqcnn_noise` Qiskit is too old to ship FakeBrisbane, so the packaged calibration and Aer reference were used, as pre-registered.

## 3. Accuracy (%), mean ± seed SD, n = 10 per cell

**Matched arm (trained and tested at α):**

| K | α | M1 | M2 | M3 | M4 | M5 |
|---|---|---|---|---|---|---|
| 5 pooled | 0 | 90.0 ± 1.8 | 90.7 ± 1.6 | 92.9 ± 0.5 | **93.4 ± 0.7** | 91.4 ± 2.6 |
| | 0.5 | 90.2 ± 1.9 | 90.8 ± 1.4 | 93.2 ± 0.5 | **93.4** | 91.5 ± 2.8 |
| | 1 | 90.1 ± 2.3 | 91.0 ± 1.0 | 92.9 ± 0.7 | **93.4** | 91.5 ± 2.2 |
| | 2 | 90.3 ± 1.4 | 90.8 ± 1.4 | 93.0 ± 0.7 | **93.4** | 91.3 ± 2.7 |
| | 4 | 90.1 ± 1.7 | 91.1 ± 1.4 | 93.0 ± 0.7 | **93.4** | 91.4 ± 2.3 |
| 10 | 0 | 78.7 ± 1.4 | 79.2 ± 2.2 | 81.6 ± 1.5 | **81.6 ± 2.2** | 78.5 ± 3.8 |
| | 0.5 | 78.6 ± 1.7 | 79.3 ± 1.5 | 81.3 ± 2.1 | **81.6** | 78.9 ± 3.6 |
| | 1 | 78.7 ± 1.9 | 79.7 ± 1.3 | **81.7 ± 2.3** | 81.6 | 78.8 ± 3.4 |
| | 2 | 79.3 ± 1.9 | 80.5 ± 1.5 | **81.9 ± 1.9** | 81.6 | 78.7 ± 3.8 |
| | 4 | 77.9 ± 1.7 | 80.2 ± 2.2 | **81.7 ± 2.4** | 81.6 | 78.4 ± 3.9 |

M4 has no quantum front-end, so its value is the same at every α (the α = 0 runs). Every row is flat in α to within its seed SD.

**Deployed arm (trained at α = 0, tested at α):**

| K | α_test | M1 | M2 | M3 | M4 | M5 |
|---|---|---|---|---|---|---|
| 5 pooled | 1 | 89.7 | 90.2 | 93.0 | 93.4 | 90.5 ± 5.2 |
| | 2 | 89.1 | 88.9 | 92.9 | 93.4 | 89.2 ± 9.3 |
| | 4 | 85.5 | 83.2 | 92.6 | 93.4 | 87.7 ± 13.2 |
| | 8 | 67.4 | 59.1 | 90.8 | 93.4 | 83.4 |
| | 16 | 35.9 | 25.8 | 70.6 | 93.4 | 60.6 |
| 10 | 1 | 78.5 | 78.6 | 81.3 | 81.6 | 78.1 |
| | 2 | 77.5 | 76.2 | 80.8 | 81.6 | 77.0 |
| | 4 | 73.4 | 65.4 | 78.4 | 81.6 | 73.0 |
| | 8 | 54.2 | 36.1 | 62.2 | 81.6 | 55.1 |
| | 16 | 21.0 | 12.7 | 16.2 | 81.6 | 16.1 |

Full table: `v15_summary.csv`. α = 8 and 16 are stress levels, deployed arm only.

## 4. Pre-registered primary contrasts (Holm within K, 4 tests)

| Contrast (α = 1 unless slope) | K = 5 pooled | K = 10 |
|---|---|---|
| **NC3@1** = M3(1) − M3(0) | −0.05 [−0.60, +0.50], p_holm 1, MDE 0.76, **TOST ✓ (p = .002)** | +0.10 [−0.80, +0.99], p_holm 1, MDE 1.24, **TOST ✓ (p = .024)** |
| **F@1** = M3 − M4 | −0.54 [−1.26, +0.18], p_holm 0.38, MDE 1.00 | +0.09 [−1.78, +1.95], p_holm 1, MDE 2.59 |
| **T@1** = M3 − M5 | +1.34 [−0.14, +2.83], p_holm 0.29, MDE 2.07 | +2.88 [+0.54, +5.22], p = .021, **p_holm 0.085**, MDE 3.25 |
| **slope(F)**, pp per unit α | −0.01 [−0.09, +0.07], p_holm 1, MDE 0.11 | +0.08 [−0.24, +0.39], p_holm 1, MDE 0.43 |

**One primary is established: device noise costs the trained quantum front-end nothing (equivalence at both K).** F and slope(F) are not detected. At K = 5 the CI excludes an M3 advantage over M4 above +0.18 pp at device noise. T at K = 10 is detected uncorrected but not after Holm. Independently recomputed from `v15_runs.csv`: identical to `collect_v15.log`. Per subset at K = 5: F@1 is −0.40 (A) and −0.68 (B), T@1 is +1.55 / +1.14, NC3@1 is +0.26 / −0.36. Same signs on both subsets, none detected.

## 5. Secondary contrasts (uncorrected; full table in `v15_contrasts.csv`)

**Matched arm, every training α (0 / 0.5 / 1 / 2 / 4):**

| Contrast | K = 5 pooled | K = 10 |
|---|---|---|
| F = M3 − M4 | −0.49 / −0.23 / −0.54 / −0.44 / −0.45 (TOST ✓ at 0, 0.5, 2) | −0.01 / −0.28 / +0.09 / +0.31 / +0.14 |
| T = M3 − M5 | **+1.59** / +1.71 / +1.34 / +1.64 / **+1.55** | **+3.08 / +2.40 / +2.88 / +3.22 / +3.39** |
| R = M5 − M4 | **−2.07** / −1.94 / **−1.88 / −2.08 / −2.00** | **−3.09 / −2.68 / −2.79 / −2.91 / −3.25** |
| BE = M3 − M2 | **+2.25 / +2.43 / +1.87 / +2.13 / +1.90** | **+2.37 / +2.02 / +1.95** / +1.43 / +1.51 |
| Q = M2 − M1 | +0.70 / +0.59 / +0.94 / +0.53 / +0.92 | +0.54 / +0.75 / +1.05 / +1.15 / **+2.33** |
| NC_m (α = 0.5 to 4) | M3 4/4 and M5 4/4 TOST ✓. M1 2/4, M2 2/4 TOST ✓. None detected | M5: TOST ✓ 4/4. M3: TOST ✓ at 0.5 and 1. One detection: NC_M1@2 = **+0.65** (positive) |
| slopes in α | all null (T, R, BE, Q, NC_m) | all null |

**Deployed arm (α_test = 1 / 2 / 4 / 8):**

| | K = 5 pooled | K = 10 |
|---|---|---|
| DEP_M1 | −0.24 / **−0.94 / −4.53 / −22.6** | −0.22 / **−1.23 / −5.24 / −24.5** |
| DEP_M2 | **−0.44 / −1.84 / −7.50 / −31.6** | **−0.67 / −2.99 / −13.8 / −43.1** |
| DEP_M3 | +0.07 / +0.01 / −0.33 / **−2.10** (TOST ✓ to α = 4) | **−0.34 / −0.84 / −3.21 / −19.4** |
| DEP_M5 | −0.84 / −2.13 / −3.63 / −7.95 (seed 46, §6 L4) | −0.42 / −1.54 / **−5.49 / −23.4** |
| REC_m = matched − deployed, α = 4 | M1 **+4.7**, M2 **+7.9**, M3 +0.4, M5 +3.7 | M1 **+4.5**, M2 **+14.8**, M3 **+3.4**, M5 **+5.3** |
| F, deployed | −0.42 / −0.48 / −0.82 / **−2.58** | −0.35 / −0.85 / **−3.22 / −19.4** |

Bold = p < .05 uncorrected (not Holm).

## 6. Findings

### L1. Device-level noise costs the trained quantum front-end nothing (primary)

NC3@1 is equivalent to 0 within ±1 pp at both K. Across the 16 matched-arm noise-cost cells for M3 and M5 (α = 0.5 to 4, both K), equivalence holds in 14 and no cost is detected in any. The models train *through* the noise: at α = 1 the front-end output moves by only ~0.04 on a [−1, 1] scale (a contraction from readout and depolarizing error, plus a small T1 bias), and the first convolution absorbs it. Even at 4× device noise the matched accuracy of every model is flat. **Within this noise model, current IBM hardware noise is not a reason for the front-end to fail.**

### L2. Noise does not create or reveal a front-end advantage (primary)

F is never detected in the matched arm (10/10 cells), and slope(F) is null with a tight MDE (0.11 pp per unit α at K = 5). At K = 5 the point estimates sit at −0.2 to −0.5 pp at every α, with TOST equivalence at three of five. **This closes the last pre-registered front-end condition:** across class count (v12), readout width (v13), training vs freezing (v13.5), data size (v14) and hardware noise (v15), the trained quantum front-end never beats having no front-end. In the deployed arm, F becomes negative from α = 4 (K = 10: −3.2 pp) because M4 has nothing to perturb. Front-end noise can only cost.

### L3. The QNN head amplifies front-end noise (exploratory; Figs. D2, D3)

M2 and M3 have the same Circuit-4 front-end and see the same noise; only the head differs. Trained clean and tested under noise, the QNN head loses much more:

| α_test | M2 loss − M3 loss, K = 5 | K = 10 |
|---|---|---|
| 1 | −0.51 [−0.88, −0.14], p = .012 | −0.32, n.s. |
| 2 | −1.85, p = .003 | −2.15, p = .010 |
| 4 | **−7.17**, p = .001 | **−10.58**, p = .0008 |
| 8 | −29.5, p = 1e-6 | −23.8, p = 9e-6 |

The front-end weights are trained separately in M2 and M3, so this is not a pure head-only comparison. M1 (RandomLayers + QNN head) also loses more than M3 at every α ≥ 2 at both K. This fits the v13/v14 reading of the QNN head as a high-frequency trig feature map (v14 L3: easy to fit, poor generalisation). A map that is sensitive to small changes in its input is, for the same reason, sensitive to small perturbations of that input. Noise-aware training recovers most of it: REC_M2 at α = 4 is +7.9 / +14.8 pp. **Exploratory, found after reading results.** Not substituted for any primary.

### L4. Seed 46 again: its frozen front-end is the noise-fragile one (exploratory; Fig. D4)

Seed 46 is the worst M5 in **all 15** matched cells (mean within-cell z = −2.27; v13.5: −2.3, v14: −2.39). It is also the only seed whose *clean-trained* model breaks at device noise. At K = 5, M5 seed 46 goes 85.0 → 76.1 at α = 1 and 50.6 at α = 4, while the other nine seeds are flat to α = 4. At K = 10 it drops from 71.1 to 48.1 at α = 4, against a median of 79.7 to 75.4. Without seed 46, DEP_M5@4 is −0.21 (K = 5) instead of −3.63. With noise-aware training, seed 46 is flat (84.4–86.2 at K = 5). Its random initialisation puts the frozen circuit in a noise-sensitive region, and training the front-end (M3 seed 46: 92.0 → 91.2 at α = 4) moves it out. T without seed 46 is +0.8 to +1.0 (K = 5) and +2.0 to +2.4 (K = 10): the same reading as v13.5 R3 and v14 L5.

### L6. Noise tolerance is set by the head (exploratory; Fig. D3)

Per seed: the noise level at which a clean-trained model first loses 2 pp. Medians at K = 5 / K = 10:

| | M1 (QNN head) | M2 (QNN head) | M3 (classical head) | M5 (classical head) |
|---|---|---|---|---|
| K = 5 | 2.9× | 2.2× | **8.7×** | 8.3× (seed 46: 0.3× and 1.1×) |
| K = 10 | 2.2× | 1.7× | **2.9×** | 2.5× (seed 46: 0.8×) |

Every seed of every model tolerates device-level noise (above 1×) except seed 46's frozen front-end and two M2 runs right at about 1.0×. The classical-head models tolerate about 3–4× more noise than the QNN-head models at K = 5. At K = 10 all models are more fragile and the gap narrows.

### L7. Noise does not change how training runs (exploratory; Fig. D6)

The median best epoch shows no trend with the training α for any model (K = 5: 25–45 epochs, K = 10: 37–55). Training under noise does not slow convergence or need more of the epoch budget, consistent with no run near the cap. M4 (no front-end) reaches its best epoch later than the quantum front-end models at K = 10 (median 55 vs 37–47), the same pattern as v14 L7.

### L5. Everything else replicates under noise

- BE ≈ +1.4 to +2.4 pp at every α, matching v13 C (2.3 pp), and slope(BE) is null. The noiseless head keeps its w15 cost unchanged.
- T > 0 in 10/10 cells, and R < 0 in 10/10.
- M5 has the largest seed SD in 9/10 matched cells (2.2–3.9 pp vs M3's 0.5–2.4), as in v13.5 and v14.
- One positive noise effect is detected uncorrected (NC_M1@2, K = 10, +0.65). That is 1 of 32 matched-arm cost tests, which is chance level.

## 7. Pre-registered predictions: verdicts

| Prediction | Verdict |
|---|---|
| P1: NC3@1 ≡ 0 (TOST) at both K; NC_m not detected for any model at α ≤ 2 | **Holds.** TOST ✓ at both K. One detection at α ≤ 2 (NC_M1@2, K = 10), positive and at chance level. |
| P2: F@1 not detected, point ≤ 0; slope(F) point ≤ 0, not detected | **Main claim holds** (nothing detected). The sign sub-predictions hold at K = 5 and narrowly fail at K = 10 (F@1 +0.09, slope +0.08, both inside ±0.1). |
| P3: T ≥ 0 at every α | **Holds:** 10/10. |
| P4: DEP_m ≤ 0, more negative as α grows; REC_m ≥ 0 | **Holds** for M1, M2, M5. DEP_M3 at K = 5 is +0.01 to +0.07 up to α = 2 (zero within noise, TOST ✓). REC ≥ 0 in 30/32; the two exceptions are −0.1 and −0.2, n.s. |
| P5 (exploratory): slope(BE) null; deployed M1 degrades less than the Circuit-4 models | **Half.** slope(BE) is null at both K. M1 degrades less than M2 (same QNN head), but **more** than M3 (both K, α ≥ 2) and about as much as M5. Deployment fragility is set by the head more than by the size of the front-end perturbation (L3). |

Outcome per pre-registration §7: **"NC3 ≈ 0 (TOST) at α = 1, F not detected → device-level noise costs nothing measurable once the model trains under it, and still no front-end advantage. Noise does not change the v12–v14 conclusion."** And: **"DEP ≪ 0 while NC ≈ 0 → noise is harmful only to models trained without it: a deployment message, not an architecture one."**

## 8. What it means for the thesis

1. **The front-end question is closed, under all five conditions.** A trainable 6-parameter Circuit-4 front-end never beats having no front-end on MNIST: not across class count, readout width, training vs freezing, data size, or calibrated hardware noise. The result is bounded at about +1 pp everywhere it was tested. It is a clean, well-powered negative result.
2. **Hardware noise is not the obstacle for this front-end.** Two CRZ gates per block (4 ECR) are shallow enough that ibm_brisbane noise is absorbed by training, and models stay unaffected up to 4× device noise. This is a positive engineering finding: the end-to-end trainable front-end is NISQ-compatible. It is just not useful.
3. **The earlier project claim of a QCQ-CNN noise-robustness advantage is not supported.** The quantum models are at best as robust as the classical-head ones. The QNN head is the most noise-fragile part when a model is deployed without noise-aware training. Correct the wording in the thesis to "noise-tolerant when trained under noise", not "more robust".
4. **The QNN head result now has three angles:** a fixed ~2.3 pp cost on clean data (v13), a cost that grows with data scarcity (v14), and amplification of input noise (v15, exploratory). All three point at the trig feature map's poor inductive bias.

### Supervisor summary

> Under ibm_brisbane-calibrated noise on the quantum front-end, scaled from 0.5× to 4× the device, a model trained under the noise loses nothing. The trained quantum front-end's accuracy at device noise is equivalent to its noiseless accuracy within ±1 pp at both 5 and 10 classes. It still does not beat a model with no front-end, and noise does not change that. Noise only matters when a model trained without noise is run on noisy hardware. Then the quantum head is the weak point: with the same front-end and the same noise, the QNN head loses 7–14 pp at 4× device noise against 0.3–3 pp for the classical head (exploratory). Seed 46's frozen front-end is again the outlier: the worst M5 in every cell, and the only seed that degrades at device noise. That makes five conditions in which the trainable quantum front-end gives no advantage.

## 9. Caveats

- **Noise model limits:** device-median calibration, no qubit mapping or routing SWAPs (optimistic for M1's non-neighbouring CNOT), no crosstalk or idle errors, exact expectation values (no shot noise). The model is about 9% optimistic against a full transpiled device simulation, so α = 1 here is roughly α ≈ 1.1 on the device.
- **Front-end only:** the QNN head was noiseless by design, so L3 is about the head's sensitivity to front-end noise, not about noise in the head.
- **Exploratory items:** L3, L4 and the M1 comparison in P5 were found after reading results and are not substituted for the pre-registered contrasts. Secondary contrasts are many and uncorrected; read them as patterns.
- MNIST only, N = 500/class.

## 10. Next

- **Experiments v12–v15 are complete.** The remaining time goes to writing. Lead the results chapter with the five-condition front-end summary (one table, F in every condition with its bound), then the QNN-head story (v13 → v14 → v15).
- **Optional, cheap, strengthens v14 L3 and v15 L3 at once:** the classical-replica control (a classical head computing the same 12 trig features of fc2's outputs). If it shows both M2's low-data collapse and its noise amplification, the fragility belongs to the feature map, not to anything quantum. Worth it only if time allows.
- Thesis wording fix: "noise robustness advantage" becomes "noise-tolerant under noise-aware training; no robustness advantage".

## 11. Figure map

**Standard set (`figures_v15/`)**

| Figure | Shows | Finding |
|---|---|---|
| `noise_curves.png` | Matched arm: accuracy vs α, M1–M5, 95% CI | **Headline 1.** Every curve is flat to α = 4; M4 and M3 on top (L1, L2). |
| `deploy_curves.png` | Deployed arm: accuracy vs test α to 16, 95% CI | **Headline 2.** Flat to α ≈ 1–2, then the QNN-head models (M1, M2) fall first; M3 holds longest at K = 5 (L3). |
| `contrasts_vs_alpha.png` | F, T, R, BE, NC_M3 vs α, both arms, ±1 pp band | Matched contrasts flat; the deployed arm diverges only at α ≥ 4. |
| `recovery.png` | REC_m vs α | Noise-aware training recovers most for M2, little for M3 (which loses little). |
| `bars_k5.png`, `bars_k10.png` | Matched arm, mean ± seed SD | §3 as bars. |
| `strip_k5.png`, `strip_k10.png` | Every seed | M5's isolated low point at every α is seed 46 (L4). |
| `paired_k5.png`, `paired_k10.png` | Per-seed F, T, NC_M3 | NC_M3 clusters on 0; one large T seed (46). |
| `effect_vs_mde.png` | \|effect\| vs MDE | Primaries below the line; NC3 well inside its MDE. |
| `seed_sd_vs_alpha.png` | Seed SD vs α | No model becomes less stable with noise in the matched arm. |

**Deep set (`figures_v15_deep/`, exploratory)**

| Figure | Shows | Finding |
|---|---|---|
| D1 `D1_train_under_noise.png` | Per model: accuracy change vs α, trained at α vs trained clean; M4 as the no-quantum control | Matched curves stay inside ±1 pp; clean-trained curves fall from α ≈ 2 (L1). M4 is exactly 0 in both arms: all 30 M4 runs give identical predictions at every α, confirming the noise touches only the quantum front-end. |
| D2 `D2_head_amplification.png` | Deployed loss vs α for M2 (QNN head) and M3 (classical head), same front-end | QNN head loses 7–11 pp more at α = 4 (L3). |
| D3 `D3_noise_tolerance.png` | Per seed, the α where a clean-trained model first loses 2 pp | Classical head ≈ 8–9× vs QNN head ≈ 2–3× at K = 5; only seed 46's frozen front-end fails below device level (L6). |
| D4 `D4_seed46.png` | Every seed's deployed curve, M5 and M3 | Seed 46's frozen front-end falls from α = 0.5; trained (M3) seed 46 is normal (L4). |
| D5 `D5_subset_consistency.png` | NC3@1, F@1, T@1 on subsets A, B, pooled and K = 10 | Same sign on both subsets for all three; NC3 inside ±1 pp everywhere. |
| D6 `D6_training_dynamics.png` | Median best epoch vs training α | No trend; noise does not change training (L7). |
