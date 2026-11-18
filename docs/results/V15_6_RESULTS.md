# v15.6 results: noise geometry and Fisher ratio of M3, M4, M5

**Run:** 2026-11-04/05, 180 jobs (M3/M4/M5 × trained at σ = 0 or 0.3 × K5 A, K5 B, K10 × seeds 42–51).
**Pre-registration:** `PREREGISTRATION_v15_6.md` (A1, A2, D1).
**Integrity:** smoke gate 18/18 and grid gate 180/180 `ALL CHECKS PASSED`.
- Every job reproduced its stored v15.5 run bit-exactly: test accuracy, best and stop epoch, every test prediction, and every cross-eval prediction.
- A1 draw-0 predictions equal v15.5's. σ = 0 sanity holds at all layers.
- L-pix metrics are identical across models within each (cell, seed). The geometry pass took 4–11 s per job.
- D1 (a gate bug flagging `J0_full`) was fixed and logged before any number was read.

## 1. Answer in one paragraph

The trained quantum front-end does **not** make the noisy cloud smaller than having no front-end. At the front-end output (L0), M4's 2×2 averaging gives a cloud about 4× tighter relative to the data (ρ_W 0.23 vs 0.87). M3's advantage appears from the first conv block onward, and it comes from **where the cloud goes, not how big it is**. After conv (L1) and in the 2-D decision plane z (L2), M3's noisy cloud is *wider* than M4's (ρ_W +0.06, p < 1e-4) but *displaced less* (η −0.09, p = .0002, 10/10 seeds at K5). So fewer noisy points cross into another digit's region: at L1, bleed is 10% for M3 vs 23% for M4 (K5). In the z-plane, clean-trained M4's noisy digits are pulled toward the point where all the class regions meet, while M3's stay inside their own region. The Fisher ratio does not capture this: its retention differs little and inconsistently between M3 and M4 (K5 favours M3, K10 favours M4, neither significant after Holm). Training the circuit matters: the frozen M5 is worst on shift and bleed almost everywhere.

## 2. Accuracy check (L3 error on the 50-per-class subset, test σ = 0.3)

| | M3 | M4 | M5 |
|---|---|---|---|
| K5 deployed (clean σ = 0) | 7.0 | 6.6 | 9.2 |
| K5 deployed, σ = 0.3 | **17.0** | 26.8 | 30.8 |
| K5 matched, σ = 0.3 | **7.7** | 8.7 | 10.0 |
| K10 deployed, σ = 0.3 | **39.9** | 44.0 | 51.7 |
| K10 matched, σ = 0.3 | **19.8** | 23.6 | 24.0 |

This reproduces the v15.5 effect on the metric subset.

## 3. Layer profile (K5 pooled, deployed, test σ = 0.3; mean over 10 seeds)

| Layer | η shift M3 / M4 / M5 | ρ_W spread M3 / M4 / M5 | bleed % M3 / M4 / M5 | Fisher retention M3 / M4 / M5 |
|---|---|---|---|---|
| Lpix | 0.83 (all) | 0.66 (all) | 19.3 (all) | 0.47 (all) |
| L0 front-end | 0.88 / **0.82** / 1.15 | 0.87 / **0.23** / 1.14 | **20.5** / 28.5 / 31.3 | 0.43 / **0.72** / 0.41 |
| L1 conv | **0.50** / 0.59 / 0.65 | 0.15 / **0.09** / 0.22 | **10.1** / 23.3 / 24.3 | 0.78 / **0.81** / 0.69 |
| L2 z | **0.24** / 0.32 / 0.36 | 0.19 / **0.13** / 0.26 | **17.0** / 27.9 / 31.2 | **0.71** / 0.66 / 0.61 |
| L3 logits | **0.24** / 0.32 / 0.36 | 0.20 / **0.13** / 0.27 | **17.0** / 26.8 / 30.8 | — |

K10 shows the same pattern, except that at L0 M3's bleed equals M4's (40.0 vs 40.8%) and Fisher retention at z favours M4 (0.62 vs 0.68).

## 4. Primary contrasts (deployed, test σ = 0.3, Holm within K)

| K | ID | Contrast | Estimate [95% CI] | p (Holm) | MDE | Seeds < 0 |
|---|---|---|---|---|---|---|
| 5 | P-Fη | M3 − M4, η at L0 | +0.067 [−0.016, +0.150] | .13 | 0.115 | 3/10 |
| 5 | P-Tη | M3 − M5, η at L0 | **−0.262** [−0.433, −0.090] | **.029** | 0.238 | 8/10 |
| 5 | P-FJ | M3 − M4, J retention at z | +0.048 [+0.007, +0.088] | .079 | 0.056 | 2/10 |
| 5 | P-TJ | M3 − M5, J retention at z | +0.107 [−0.008, +0.221] | .13 | 0.159 | 2/10 |
| 10 | P-Fη | M3 − M4, η at L0 | **+0.149** [+0.105, +0.192] | **.00013** | 0.061 | 0/10 |
| 10 | P-Tη | M3 − M5, η at L0 | −0.150 [−0.302, +0.002] | .16 | 0.211 | 8/10 |
| 10 | P-FJ | M3 − M4, J retention at z | −0.057 [−0.131, +0.016] | .23 | 0.102 | 7/10 |
| 10 | P-TJ | M3 − M5, J retention at z | +0.079 [−0.056, +0.215] | .23 | 0.189 | 3/10 |

Two primaries are detected:
- **T on η at L0, K5:** the trained circuit shifts noise less than the frozen one, as predicted.
- **F on η at L0, K10:** M3's front-end output shifts *more* than M4's. This is the opposite of P1's direction.

## 5. Prediction verdicts

- **P1 (η at L0: M3 < M4 < M5): partly failed.** M3 < M5 holds (K5 detected; K10 same direction, n.s.). M3 < M4 fails: the sign is wrong, detected at K10.
- **P2 (Fisher retention at z: M3 > M4 > M5): not supported.** K5 is in the predicted direction but not significant after Holm. At K10, F has the wrong sign (n.s.).
- **P3 (the matched arm shrinks F and T toward 0): failed.** With noise training, M3's lead on shift and bleed *grows*: matched F on η is −0.14 at L0 and −0.08 at z (K5, p < 1e-4, 10/10 seeds).
- **P4 (at L0, model differences larger on η than on ρ_W): failed** at L0, where spread differs far more than shift. The idea behind P4 does show up downstream (finding L3, exploratory).
- **P5 (sanity): holds**, verified by the gate.
- **P6 (M5: |sin δ_b| vs η at L0): right direction, not significant.** ρ = +0.52, p = .13 at both K. The seed ranking of η is the same in every cell, because a frozen circuit's response to noise hardly depends on the digits.

## 6. Findings (L1–L6 from the pre-registered secondaries; the interpretation is exploratory)

- **L1. Averaging, not the circuit, makes the tightest front-end cloud.**
  - At L0, M4's ρ_W is 0.23 vs M3's 0.87 (K5), and F is detected at every K and arm (p < 1e-6).
  - Averaging 4 pixels cuts independent pixel noise 4× but cuts stroke structure much less. Circuit 4 keeps the 4 pixels as separate channels, so their noise stays.
- **L2. At L0, M3's noise lands in less class-confusing directions.**
  - Mahalanobis bleed at L0 is 20.5% vs 28.5% (K5, p = .0008, 10/10 seeds) despite M3's 4× larger cloud.
  - At K10 the two are equal (p = .47).
- **L3. The advantage forms in conv1/conv2 and is a shift effect, not a spread effect.** From L1 onward:
  - M3 has the *larger* spread (ρ_W F = +0.06, p < 1e-4).
  - M3 has the *smaller* shift (η F = −0.095 at L1, p = .005; −0.087 at z, p = .0002).
  - M3 has much less bleed (L1: −13 pp, p = .0005).
  - Accuracy tracks shift and bleed, not spread.
  - **z-plane (`fig3_zplane_k5_A_deployed.png`):** clean-trained M4's noisy clouds slide toward the junction where all the regions meet. M3's clouds stay inside their own regions.
  - This fits the clipped-noise mechanism: the noise is a one-sided brightening of the background (FRONTEND_MATH §3.4) that drags every digit the same way, and the trained cosine passes less of it into conv.
- **L4. The Fisher ratio is not the right summary of this robustness.** J = tr(S_W⁻¹ S_B) rewards a tight cloud. M3's cloud is wider, so its J retention is no better than M4's even though it bleeds far less. What matters is the *displacement relative to the decision boundaries*: η and bleed.
- **L5. Training the circuit matters.** M5 has the largest shift and bleed at nearly every layer. T on bleed at z: −14 pp (K5, p = .026).
- **L6. Noise training amplifies M3's lead on geometry.** In the matched arm, M3 is best on η and bleed at every layer and K (all F p < .005), and on accuracy (K10: 19.8% error vs 23.6% and 24.0%).

## 7. What it means for the thesis

- **Do not claim** "the quantum front-end filters noise" or "reduces the noise spread". It doesn't; plain 2×2 averaging gives a tighter cloud.
- **Do claim**, with v15.5 accuracy and this geometry: *with the trained cosine front-end, the network's internal representations are displaced less by input noise. From the first conv block onward, noisy digits stay inside their own class regions, while without the front-end the noise drags all digits toward a common point between the regions. The noise cloud is not smaller; it is better anchored.*
- The Fisher ratio goes in as a reported, pre-registered metric that did **not** separate the models, with the explanation in L4. That is itself a useful methodological point.
- The classical caveat from V15_5_MECHANISM still stood at the time of writing: a fixed cos(πx) map (M6) had not been run on Gadi. (Resolved by v15.8: `V15_8_RESULTS.md`.)

## 8. Caveats

- n = 10 seeds; 50 test images per class × 20 draws for the noisy metrics.
- Bleed at L0–L1 uses a shrunk Mahalanobis nearest-mean rule (λ = 0.1), not the network's own decision rule. Only L3 bleed is the actual error.
- Interpretations L2–L4 are post hoc.
- P6 uses the same seed ranking in every cell, so its K5 and K10 values are not independent tests.

## 9. Figure map (`v156_out/figures/`)

| Figure | Shows |
|---|---|
| `fig2_layer_profile_K5.png` / `_K10` | **Lead figure.** η, ρ_W, Fisher retention and bleed across Lpix → L0 → L1 → L2 → L3, both arms |
| `fig3_zplane_k5_A_deployed.png` (+ other cells and arms) | **The "regions" picture:** class regions in z with clean and σ = 0.3 noisy points, seed 42 |
| `fig1_metric_vs_sigma_K5.png` / `_K10` | Every metric vs test σ at every layer |
| `std_primaries_K5.png` / `_K10`, `std_effect_vs_mde.png` | Standard set for the primaries |
| `figP6_M5_slope_vs_eta.png` | P6 |
