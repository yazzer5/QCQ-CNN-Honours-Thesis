# v16a results: classical replica of the w15 QNN head (M2c)

**Status:** complete. 73 runs (11 smoke + 62 grid), both integrity gates passed. All 13 reproductions (M2 K5 A N500, plus 12 `--save_z` M2/M3 runs) equal the stored v13/v14 runs bit-exactly. Pre-registration: `PREREGISTRATION_v16a.md` (A0 walltime, A0b invocation, D1 drift-diagnostic shim; all logged before the smoke).
**Date:** 2026-10-21. **Figures:** `figures_v16a/`. **Data for v16c:** `results_v16a_z_k10_all_n{50,500}/v16_z_{M2,M3}_seed{42,43,44}.npz`.

## 1. Answer in one paragraph

**Replacing the quantum circuit with 12 closed-form trig features reproduces everything the QNN head does, and nowhere does the circuit help.** M2c reproduces M2's back-end cost against the classical head at every (K, N): 14.0 vs 13.5 pp (K = 5) and 15.8 vs 17.1 pp (K = 10) at N = 50; 1.9 vs 2.3 pp at N = 500. It also reproduces the full low-data collapse. Its accuracy differs from M2 by −0.5 to +1.3 pp, never detected. The circuit's parameterisation is worth **at most +0.75 pp (K = 5) and +0.88 pp (K = 10) at N = 500** (95% one-sided bound on M2 − M2c), and every point estimate favours the closed form, not the circuit. Formal ±1 pp equivalence (P1) was not reached in any cell. At N = 500 the 90% CIs, [−0.75, +1.39] and [−0.88, +1.85], overshoot the margin only on the side where the circuit would be *worse*. At N = 50 the seed noise (MDE 6–7 pp) is too large to show equivalence. M2c errs on the same images as M2 about as much as two M2 seeds do (P2 holds in 3/4 cells; the fourth misses by 0.0001 in Jaccard). **So the QNN head's cost belongs to its classical feature map, and none of it is attributable to anything quantum.**

## 2. Integrity

- **Verify (on Gadi):**
  - The real `QNNBackendTorch` equals ZZFeatureMap(2, **reps = 1**) + RealAmplitudes(2, reps = 2), column by column (1.8e-15); reps = 2 does not match.
  - Span check: all 15 Paulis are linear combinations of the 12 features + a constant, residual 2.6e-14 (pre-registered threshold 1e-6). The converse holds (3.2e-14), so the two heads have the same function class.
  - M2c with its readout set to W_fc3·C reproduces M2's logits to 3.9e-7.
  - M1–M5 are bit-identical to the v15 file.
  - `--save_z` is invisible to training.
- **D1 (defect, fixed before the smoke):** `train_model` reads `back_end.qnn.weight` for the v5 drift print. M2c's head got a constant, non-parameter, non-persistent 6-element buffer; drift prints 0. No effect on parameters, outputs or training.
- **Gates:** smoke 11/11, grid 62/62.
  - 54 (K, subset, N, seed) groups; identical images within each; one test set per (K, subset).
  - **All 36 N = 50 runs share their split with the stored v14 M2 log of the same cell.**
  - No continuations, no resubmissions, no cap hits: latest stop epoch 105 against a cap of 2000; longest job 8.5 min.
- **Parameters:** K = 5: M2 1686 → **M2c 1665** (M3 1725). K = 10: 1766 → **1730** (1805).
- Collect numbers independently recomputed from `v16a_runs.csv` and the result JSONs: identical.

## 3. Accuracy (%), mean ± seed SD, n = 10

| K | N | M2 (QNN head, w15) | M2c (closed form) | M3 (classical head) |
|---|---|---|---|---|
| 5 pooled | 50 | 67.95 ± 4.82 | 67.43 ± 9.08 | **81.47 ± 2.92** |
| | 500 | 90.68 ± 1.61 | 91.01 ± 1.41 | **92.93 ± 0.51** |
| 10 | 50 | 47.01 ± 6.45 | 48.34 ± 6.59 | **64.14 ± 3.88** |
| | 500 | 79.22 ± 2.18 | 79.71 ± 2.22 | **81.59 ± 1.51** |

Per subset (K = 5): A 67.4 / 65.9 at N = 50, 90.4 / 90.9 at N = 500; B 68.5 / 69.0 and 90.9 / 91.1 (M2 / M2c).

## 4. Primary: D = M2c − M2 (paired; TOST ±1 pp; Holm within K)

| K | N | D | 95% CI | 90% CI | p (Holm) | MDE | TOST p (Holm) | Circuit helps by at most (95% one-sided) |
|---|---|---|---|---|---|---|---|---|
| 5 | 50 | −0.52 | [−5.36, +4.32] | [−4.44, +3.40] | 1.00 | 6.73 | 0.41 | 4.44 pp |
| 5 | 500 | +0.32 | [−1.00, +1.64] | [−0.75, +1.39] | 1.00 | 1.84 | 0.28 | **0.75 pp** |
| 10 | 50 | +1.33 | [−3.27, +5.93] | [−2.40, +5.06] | 1.00 | 6.39 | 0.56 | 2.40 pp |
| 10 | 500 | +0.49 | [−1.21, +2.18] | [−0.88, +1.85] | 1.00 | 2.35 | 0.51 | **0.88 pp** |

Nothing is detected, and nothing is shown equivalent. Per subset, D is −1.54 / +0.45 (A) and +0.50 / +0.19 (B) at N = 50 / 500. Three of four cells, and both subsets at N = 500, have D ≥ 0.

## 5. Secondary

**Back-end cost with each head (P3):**

| K | N | BE = M3 − M2 | BEc = M3 − M2c | BEc / BE |
|---|---|---|---|---|
| 5 | 50 | +13.52 [11.46, 15.58], p = 1e-7 | **+14.04 [8.20, 19.88], p = 4e-4** | 1.04 |
| 5 | 500 | +2.25 [1.22, 3.27] | +1.93 [0.91, 2.94], p = .002 | 0.86 |
| 10 | 50 | +17.14 [12.38, 21.89] | **+15.81 [10.97, 20.64], p = 4e-5** | 0.92 |
| 10 | 500 | +2.37 [0.36, 4.38] | +1.88 [−0.16, 3.93], p = .067 | 0.80 |

Collapse, BEc(50) − BEc(500): **+12.1 pp** (p = .001, K = 5) and **+13.9 pp** (p = .0003, K = 10), against M2's +11.3 and +14.8.

**Per-image error agreement (P2):** Jaccard of the two error sets; κ on correct/incorrect.

| K | N | M2c_s vs M2_s (same seed) | M2_s vs M2_t (two M2 seeds) | M2c_s vs M2_t | κ same seed / M2–M2 |
|---|---|---|---|---|---|
| 5 | 50 | 0.300 | 0.272 | 0.268 | 0.205 / 0.162 |
| 5 | 500 | 0.2048 | 0.2048 (+0.0001) | 0.203 | 0.273 / 0.271 |
| 10 | 50 | 0.439 | 0.419 | 0.417 | 0.183 / 0.132 |
| 10 | 500 | 0.311 | 0.305 | 0.295 | 0.339 / 0.329 |

Cross-seed M2c–M2 agreement equals M2–M2 agreement to within 0.01 in every cell. An M2c run is as interchangeable with an M2 run as another M2 seed is.

## 6. Findings

### F1. The circuit adds nothing the closed-form features don't already give (primary, bounded)
D is never detected, and its sign favours the closed form in 3/4 cells. With enough data (N = 500), the circuit's parameterisation is worth at most +0.75 pp (K = 5) and +0.88 pp (K = 10). This is the only direction that would matter for a quantum claim, and it is excluded above 1 pp. P1 fails as pre-registered: equivalence is not shown. At N = 500 the CI overshoots +1 only on the "circuit is worse" side. At N = 50 the test has no power (MDE 6–7 pp).

### F2. The low-data collapse is a property of the feature map (P3, holds)
M2c loses 14–16 pp to the classical head at N = 50 and about 2 pp at N = 500, the same as M2 (BEc/BE = 0.80–1.04). With no circuit at all, the feature map alone reproduces v14 L2–L3: the sample-inefficiency of the "quantum head" is a property of its 12 high-frequency trig features of z.

### F3. Same errors, not just the same average (P2: 3/4 cells)
Same-seed M2c–M2 agreement is at or above M2–M2 cross-seed agreement in every cell. The one formal miss (K = 5, N = 500) is a tie: 0.20476 vs 0.20484. The cross-seed comparison (M2c_s vs M2_t ≈ M2_s vs M2_t) is the clearer reading: the two heads are statistically exchangeable at the per-image level.

### F4. Seed 46 again; M2c's wider spread at K = 5 N = 50 is one init (exploratory)
M2c's seed SD at K = 5 N = 50 is 9.1 against M2's 4.8 (Pitman–Morgan p = .03; the only cell with a variance difference). Seed 46 is the worst M2c (40.3% on A, 47.8% on B) and also the worst M2 in that cell. It is the same Circuit-4 initialisation that is the worst M5 in every cell of v13.5–v15. Without seed 46: SD 4.1 vs 3.3, and D = +0.91 [90% CI −2.40, +4.22]. At N = 500 seed 46 is unremarkable for both heads. Found after reading results; not substituted for any primary.

## 7. Pre-registered predictions: verdicts

| Prediction | Verdict |
|---|---|
| P1: D ≡ 0 within ±1 pp (Holm TOST) in all 4 cells | **Fails (not shown).** Nothing detected; N = 500 bounded to [−0.75, +1.39] (K = 5) and [−0.88, +1.85] (K = 10) at 90%; N = 50 underpowered (MDE 6–7). |
| P2: same-seed M2c–M2 agreement ≥ M2–M2 cross-seed, all 4 cells | **Fails formally, 3/4.** The fourth is a 0.0001 tie. |
| P3: BEc(50) > 0 detected and collapse BEc(50) − BEc(500) > 0 detected, both K | **Holds** (p ≤ 4e-4 and ≤ .0014). |

Outcome per pre-registration §7: **"P1 fails, nothing detected → underpowered at n = 10; report CI and MDE. No equivalence claim."** Plus **"P3 holds → the low-data collapse is a property of the 12-feature map itself."** The "D > 0 → circuit adds optimisation difficulty" branch is not supported (nothing detected), and the "D < 0 → check the implementation" branch does not apply (no cell is detected below 0, and verify proved function-class equality).

## 8. What it means for the thesis

1. **The QNN-head story closes causally.** v13 found a ~2.3 pp clean cost, v14 a 13–17 pp low-data collapse, v15 noise amplification (exploratory). v16a now shows that a classical head computing the circuit's features in closed form reproduces the cost and the collapse. The cause is the ZZFeatureMap's feature map, not quantum computation. The thesis can state it as: *"the trainable quantum head is a fixed classical trig feature map in disguise; replacing the circuit with its closed form changes nothing measurable, and any benefit of the circuit is bounded below 0.9 pp."*
2. **Wording:** "not detected, bounded at +0.75 / +0.88 pp in the circuit's favour", **not** "equivalent". Equivalence within ±1 pp was pre-registered and not shown.
3. **v16b's premise holds.** If the cost is in the feature map, a redesigned map (BFR, bounded frequency) is the right lever. M2c (`ZZTrig12`) is validated as v16b's twin.
4. **v16c is unblocked:** the z-dumps reproduce v13/v14 exactly.

### Supervisor summary

> We replaced the quantum circuit in the QNN head with its exact classical equivalent: 12 cosine/sine features of the two bottleneck values, computed in closed form, followed by a linear layer. We first proved numerically that every quantum measurement is a linear combination of these features. Across 5 and 10 classes and 50 or 500 images per class, the classical replica matches the quantum head's accuracy within noise, reproduces its full deficit to a classical head (14–16 pp at 50 per class, ~2 pp at 500) and its collapse with less data, and fails on the same images about as often as two runs of the quantum head do. The circuit's trainable parameters are worth at most 0.75–0.9 pp at full data. The quantum head's cost comes from its feature map, which is classical, not from anything quantum. Formal ±1 pp equivalence was not reached; the low-data cells are too noisy for that test.

## 9. Caveats

- P1's equivalence was not shown; the claim is a one-sided bound plus "not detected". n = 10 seeds; MDE 1.8–2.4 pp at N = 500 and 6–7 pp at N = 50.
- M2 and M2c share front-end, backbone and fc2 initialisation and batch order at each seed by design. The pairing is exact, but same-seed agreement (P2) is partly shared initialisation, which is why the cross-seed comparison is reported alongside.
- The comparator M2/M3 runs are the stored v13/v14 runs, proven reproducible by 13/13 exact reproductions.
- F4 is exploratory. MNIST only, w15 only.

## 10. Figure map (`figures_v16a/`)

| Figure | Shows |
|---|---|
| `paired_D.png` | **Headline (primary).** Per-seed M2c − M2 with mean, 95% CI and the ±1 pp band. Tight around +0.3–0.5 at N = 500; wide at N = 50 (one −13.4 pp point is seed 46). |
| `backend_cost.png` | **Headline (P3).** BE and BEc vs N: the two lines lie on top of each other at both K. |
| `bars.png`, `strip.png` | §3 as bars and every seed per subset. |
| `agreement.png` | P2: same-seed, cross-seed M2–M2 and cross-seed M2c–M2 error agreement. Nearly equal bars. |
| `effect_vs_mde.png` | Every D below its MDE; BE/BEc at N = 50 far above. |
