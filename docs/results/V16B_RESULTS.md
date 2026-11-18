# v16b results: Design 1 (BFR head) vs M4 and M2-w15

**Status:** complete. 61 new runs: 7 smoke (6 BFR + 1 M4 reproduction) and 54 grid. Every integrity gate passed, and the reproduction is bit-exact. Pre-registration: `PREREGISTRATION_v16b.md` (A1, A2, D1; no amendment after the smoke).
**Date:** 2026-10-22. **Figures:** `figures_v16b/` (standard set, 200 dpi, `scripts/plot_v16b.py`).

> **Wording note (from `V16_SYNTHESIS.md` §4, after v16c):** where this file says "unbounded frequency", read "a fixed, high-gain periodic map" (v16c F2: M2 never uses the unbounded range). Where it says "the remaining cost comes from an active quantum branch", read "about half is the trimmed classical parameterisation (v16b-ext), the split unresolved at n = 10". The original text is kept below as written.

## 1. Answer in one paragraph

**Bounding the quantum head's encoding and adding it as a gated residual removes most of the ZZ head's cost, but not all of it, and gives no hybrid win.** BFR beats M2-w15 in every cell: +14.3 pp at N = 50 for both K, and +2.4 / +2.0 pp at N = 500 (P1 holds in all 3 pre-registered cells). That recovers 75–95% of the gap between the ZZ head and the classical head. BFR never beats M4: the point estimate is negative in 4/4 cells. Where data is scarce at 10 classes (K = 10, N = 50), BFR is **detectably worse than M4 by 4.7 pp** [−8.0, −1.3] (Holm p = .024), and its seed SD is twice M4's. The quantum branch was never switched off; the effective weight |g|·‖W_q‖ is 0.5–12 across runs. So the remaining cost comes from an active quantum branch, not a dormant one. Elsewhere BFR is within about 1 pp of M4, but n = 10 cannot show equivalence (P2 fails in 4/4 cells, three of them for power).

## 2. Integrity

- **Verify** (`verify_v16b.log`, Gadi):
  - M1–M5 and M2c are bit-identical to the v16a file at K = 5 and 10.
  - The BFR simulator equals an independent complex statevector (2.9e-15).
  - The 6 single-Y Paulis are identically 0, and Ry(u)Ry(θ) = Ry(u+θ).
  - Bounded frequency: no FFT content above degree 2.
  - Parameter table 1720/1719 (K = 5) and 1799/1799 (K = 10); the RNG and init are paired with M4; the trimmed classical path reproduces M4's logits.
  - Gradient regimes at g = 0 and g = 1.
  - Dry runs: M4 through the patched file equals the v16a file prediction by prediction, and BFR is deterministic.
  - The Qiskit cross-check was skipped on Gadi (mixed Qiskit versions in `qcqcnn`); it passed locally (7.8e-16).
- **Reproduction:** M4, K = 5 A, N = 500, seed 42 equals v14 exactly (test_acc, best 51 / stop 71, all 1000 predictions).
- **Gates:** smoke 7/7, grid 54/54.
  - Every run's train/val/test split equals the stored v14 M4 log of the same cell and seed (61/61), so BFR and M4 are paired on the images.
  - No continuations, no resubmissions, **0 runs at the epoch cap**.
  - Longest job 5 min 40 s (K10 N500).
- **Pre-registered implementation check** (triggered by BFR < M4 at K10 N50, done before reading):
  - Logs: no traceback, FATAL or inversion line.
  - g restored bit-exact after the gradient probe in every job.
  - The deficit is spread over 7/10 seeds (−4.2 to −10.3 pp), not one outlier.
  - Nothing points to an implementation fault.

## 3. Accuracy (%), mean ± seed SD, n = 10 per cell

| K | N | M2-w15 | M4 | **BFR** |
|---|---|---|---|---|
| 5 pooled | 50 | 67.95 ± 4.82 | **82.94 ± 1.51** | 82.23 ± 2.76 |
| | 500 | 90.68 ± 1.61 | **93.42 ± 0.65** | 93.12 ± 0.91 |
| 10 | 50 | 47.01 ± 6.45 | **65.89 ± 2.48** | 61.23 ± 5.17 |
| | 500 | 79.22 ± 2.18 | **81.61 ± 2.22** | 81.17 ± 1.32 |

M4 and M2 are the stored runs: M4 from v14; M2 from v14 at N = 50 and v13 at N = 500. Per subset (K = 5): A 80.9 / 93.0 and B 83.5 / 93.3 for BFR at N = 50 / 500.

**Share of the ZZ head's cost recovered by BFR**, computed as (BFR − M2) / (M4 − M2):

| | N = 50 | N = 500 |
|---|---|---|
| K = 5 | 95% (cost 15.0 → 0.7 pp) | 89% (2.7 → 0.3) |
| K = 10 | 75% (18.9 → 4.7) | 82% (2.4 → 0.4) |

## 4. Pre-registered primary contrasts (paired; Holm within K over the 2 N cells, per contrast)

| Cell | D4 = BFR − M4 | D2 = BFR − M2 |
|---|---|---|
| K5 N50 | −0.71 [−1.97, +0.55], p_holm .47, MDE 1.75; TOST p .31; non-inf p .31 | **+14.29** [+10.61, +17.96], p_holm 2e-5 |
| K5 N500 | −0.30 [−1.09, +0.48], p_holm .47, MDE 1.09; TOST p .038, **p_holm .076** | **+2.43** [+1.26, +3.60], p_holm .001 |
| K10 N50 | **−4.66** [−8.02, −1.30], **p_holm .024**, MDE 4.67; TOST p .98 | **+14.22** [+7.91, +20.53], p_holm .001 |
| K10 N500 | −0.43 [−2.50, +1.63], p_holm .65, MDE 2.87; TOST p_holm .55 | **+1.95** [+0.12, +3.77], p_holm .039 |

Per subset (secondary), D4 at N = 50 is −1.32 on A and −0.10 on B; at N = 500 it is −0.33 and −0.28. D2 is detected on each subset alone in all 4 subset cells.

## 5. Predictions: verdicts

| Prediction | Verdict |
|---|---|
| **P1** BFR − M2 ≥ +1.2 (K10 N500), ≥ +6 (N50) | **Holds** in all 3 cells (+1.95, +14.29, +14.22; all Holm p < .05). |
| **P2** BFR ≡ M4 ±1 at N500; ≥ −1 at N50 | **Fails, 0/4.** K10 N50: BFR detectably *worse* than M4. K5 N500: close (TOST p .038 uncorrected, .076 Holm). K5 N50 and K10 N500: underpowered (MDE 1.75 and 2.87), no equivalence claim. |
| **P5** seed SD(BFR) ≤ 1.5 × SD(M4) | **Fails, 2/4.** Holds at N = 500 (ratios 1.41, 0.60); fails at N = 50 (1.82 at K = 5, 2.08 at K = 10). |

Reading per pre-registration §7: P1 holds and P2 does not, and BFR < M4 is detected at K10 N50. So **bounding the encoding removes most of the head's cost (consistent with the diagnosis), and the gated quantum branch still costs accuracy against the classical head it starts from when data is scarce at 10 classes.** No cell meets the "BFR > M4 by ≥ 1 pp" bar. Causal attribution of the recovered 75–95% to removing the ZZ product term needed BFR-ZZ; v16b-ext (§11) ran it at K10 N50, and restoring the product term on bounded inputs did **not** bring the cost back.

## 6. Secondary and exploratory (uncorrected)

**V1. The quantum branch was used, not switched off (secondary).**
- |g|·‖W_q‖ at the best checkpoint is 0.45–2.0 at K = 5 and 3.3–12.5 at K = 10. No cell meets the pre-registered "switched off" rule (< 0.05 in more than half the runs).
- g's sign carries no meaning: g·W_q is invariant under (g, W_q) → (−g, −W_q), and both signs occur.
- The encoding scale s fell from π/2 in every run. Medians: 1.18 (K5 N50), 1.10 (K5 N500), 1.18 (K10 N50), 0.88 (K10 N500). Training narrowed the encoding further than the bound imposed.
- The branch's size does not track its help or harm: Spearman(|g|·‖W_q‖, per-run D4) is between −0.48 and +0.27.

**V2. At K10 N50 the cost is in fitting, not in the generalisation gap (exploratory; paired per seed with the stored v14 M4 `train_acc_at_best`, both eval mode at the best checkpoint).**

| K = 10 | D4 (test) = fit + gap | fit: BFR − M4 train acc | gap part: −(BFR gap − M4 gap) |
|---|---|---|---|
| N = 50 | −4.65 [−8.01, −1.29], p .012 | **−5.22** [−10.80, +0.36], p .064 | +0.57 [−4.76, +5.90], p .81 |
| N = 500 | −0.44 [−2.51, +1.63] | +0.37 [−1.44, +2.18] | −0.81 [−2.30, +0.68] |

- At N = 50 the whole deficit sits in training fit. The gap part is centred on zero.
- The fit difference is not detected on its own (p = .064). It is driven by seeds 43 (train 72.0% vs 95.6%) and 51 (79.2% vs 91.2%), and BFR's train-accuracy SD across seeds is 6.8 vs M4's 4.3.
- Fit and gap differences are strongly anti-correlated across seeds (r = −0.81): a BFR run that fits less also overfits less.
- **Reading:** BFR's failure mode is the *opposite* of the ZZ head's. v14 L3: equal fit, a 15.6 pp larger gap. Here: a gap no larger than M4's, and some seeds where the gated head fails to fit the training set at all. That points to optimisation instability of the narrowed or gated head at 10 classes with 500 training images. It does not point to a high-frequency inductive bias.
- At N = 500 neither part differs.
- Exploratory: n = 10, uncorrected, and found after the primary results were read.

**V3. BFR makes M4's mistakes (exploratory).** Error-set Jaccard with M4 at the same seed exceeds M4's own cross-seed agreement in every cell (K5 N50: 0.337 vs 0.282; K5 N500: 0.265 vs 0.234; K10: 0.383 vs 0.368 and 0.326 vs 0.325). BFR also overlaps less with M2 than with M4 in every cell. Behaviourally it is a member of the classical-head family.

**V4. Epochs.** Median best epoch, BFR vs M4: 91 vs 80, 46 vs 37, 96 vs 109, 59 vs 55. These are similar, with no warmup in either.

## 7. What it means for the thesis

1. **The redesign test works in one direction.** The cost of v14's quantum head is mostly a property of the ZZ encoding: unbounded frequency and the product term, ungated. Replacing it with a bounded, product-free, gated quantum branch recovers 75–95% of the cost at every N and K. This complements v16a, which showed the cost belongs to the classical feature map. Together they are the "diagnose → redesign" arc.
2. **But there is still no fair hybrid advantage.** Even when it starts *as* the classical head and can only add to it, the trainable quantum branch never improves on M4, and at K10 N50 it hurts (−4.7 pp, SD × 2). That loss is a fitting and stability failure on some seeds, not the ZZ head's generalisation failure (V2). The project-wide pattern holds: the classical head is the ceiling at this scale.
3. **Honest framing.** "Bounding removes most of the head's cost" is supported; "the product term causes it" is not yet tested. Report P2 as a failed prediction, with K10 N50 as a detected loss and the rest as underpowered.

### Supervisor summary

> We redesigned the quantum head so its encoding is bounded (u = s·tanh z), has no ZZ product term, and is added as a gated residual to the classical head, starting exactly at the classical head (gate g = 0). Over 60 runs paired with the stored M4 and ZZ-head runs, it recovers 75–95% of the ZZ head's accuracy cost: +14 pp at 50 images per class, +2 pp at 500. It never beats the classical head. With 50 images per class at 10 classes it is 4.7 pp worse and twice as seed-variable; elsewhere it is within about 1 pp, but 10 seeds can't prove equivalence. The quantum branch was genuinely used in every run (large learned gate), so this is a real cost of an active quantum branch, not a switched-off one.

## 8. Caveats

- The comparators are reused. Pairing is on images and seeds (60/60 split hashes equal v14 M4's), and one bit-exact reproduction validates the stored M4 runs. M2 at N = 500 comes from v13 (validated by v14's reproductions).
- **fc_mid trimmed** (15 → 5 at K = 5, → 3 at K = 10, A2). The function class equals M4's, but the parameterisation differs. BFR's starting point is a narrower-parameterised M4, not M4 exactly. This cannot be separated from the quantum branch's effect in v16b.
- 6 of the 15 Paulis are identically 0, so BFR reads 9 features. Effect attribution is to the whole design (bound + no product + gate + residual), not to any one part.
- V1–V4 are secondary or exploratory and are not substituted for the primary contrasts. MNIST only.

## 9. Next

- ~~Pair train fit per seed at K = 10~~: done 2026-10-22, see V2.
- ~~BFR-ZZ at K10 N50~~: run as v16b-ext (A3), together with M4-trim; see §11.
- **v16c diagnostics** and the thesis chapter "diagnose → redesign → test": lead with `paired_D2.png` (the recovery) next to `paired_D4.png` (no win, K10 N50 loss).

## 10. Figure map (`figures_v16b/`)

| Figure | Shows |
|---|---|
| `bars.png` | Mean ± SD per K, N: BFR sits next to M4, far above M2 at N = 50. |
| `strip.png` | Every seed per subset: BFR's wider spread at N = 50. |
| `paired_D4.png` | **Primary:** BFR − M4 per seed, 95% CI, ±1 pp band. Three CIs straddle 0; K10 N50 is wholly below −1. |
| `paired_D2.png` | **Primary:** BFR − M2 per seed with the P1 thresholds; all above. |
| `effect_vs_mde.png` | D2 resolvable everywhere; D4 resolvable only at K10 N50. |
| `seed_sd.png` | P5: BFR above the 1.5 × M4 line at N = 50, below it at N = 500. |
| `gate_scale.png` | Learned g (both signs, large) and s (below π/2) per run. |

## 11. v16b-ext (amendment A3): M4-trim and BFR-ZZ at K = 10, N = 50

**Status:** complete, 2026-10-22. 21 new runs: smoke 3 (M4-trim s42, BFR-ZZ s42, plus a BFR reproduction) and grid 18. A3 was written **after** v16b's results were read and **before** any ext job (honesty note in the pre-registration).
- **Integrity.**
  - Verify on Gadi passed: M1–M5, M2c and BFR are bit-identical to the v16b file; BFR-ZZ equals an independent statevector (2.4e-15) and equals BFR exactly with ζ = 0; M4-trim equals BFR's classical path tensor for tensor.
  - The BFR reproduction through the new file equals v16b exactly (best 96, stop 116, all 2000 predictions).
  - Gates: smoke 3/3, grid 18/18. Splits equal the stored v14 M4 and v16b BFR logs (6/6 and 36/36).
  - No caps hit, no continuations. Collect's decomposition check passed: C1 + C2 = BFR − M4 = −4.66 exactly.
- **Figures:** `figures_v16bx/`.

### 11.1 Accuracy (%), K = 10, N = 50, n = 10

| | M2-w15 | M4 | M4-trim | BFR | BFR-ZZ |
|---|---|---|---|---|---|
| test | 47.01 ± 6.45 | **65.89 ± 2.48** | 63.35 ± 3.07 | 61.23 ± 5.17 | 59.78 ± 5.23 |
| train @ best | 94.0 | 92.2 | 88.9 | 87.0 | 82.9 |
| gap @ best | 47.0 | 26.3 | 25.5 | 25.7 | 23.1 |
| params | 1766 | 1799 | 1643 | 1799 | 1799 |

### 11.2 Primary contrasts (paired, Holm over the 3)

| Contrast | Estimate [95% CI] | p / p_holm | MDE |
|---|---|---|---|
| C1 = BFR − M4-trim (the quantum branch on top of the trimmed head) | −2.12 [−4.72, +0.47] | .097 / .19 | 3.61 |
| C2 = M4-trim − M4 (the trim) | −2.53 [−5.07, +0.00] | .050 / .15 | 3.53 |
| C3 = BFR-ZZ − BFR (restoring the product term) | −1.45 [−4.85, +1.95] | .36 / .36 | 4.72 |

**None is detected after Holm.**

### 11.3 Predictions

| | Verdict |
|---|---|
| **E1** trim explains ≥ 50% of BFR − M4 (C2 ≤ −2.3, detected) | **Direction only, not confirmed.** The point estimate is −2.53 pp (54% of the −4.66 deficit), but it is not detected (Holm p = .15; uncorrected p = .050). |
| **E2** restoring the product term costs ≥ 5 pp (C3 ≤ −5, detected) | **Fails.** C3 = −1.45. The 95% CI lower bound is −4.85, so **a cost as large as the pre-registered 5 pp is excluded** at 95% confidence. |

### 11.4 Reading (pre-registered A3 readings, then exploratory)

1. **Product term (E2), the causal question v16b left open.**
   - Putting the ZZ product phase back into BFR's encoding, on bounded inputs with everything else identical, does not bring back the ZZ head's cost. BFR-ZZ still recovers 68% of the M4 − M2 gap (+12.8 pp over M2), against BFR's 75%, and a ≥ 5 pp loss is excluded.
   - Per the A3 reading, v16b's recovery is therefore attributed to **bounding the input, gating, and the residual form**, not to removing the product term itself.
   - These three can't be separated with the runs we have.
   - Caveat (stated in A3): this tests the product term on *bounded* inputs, which also activates the 6 Y-Pauli features. The *unbounded* product term of the original ZZ head was not reintroduced.
2. **Where BFR's −4.66 pp deficit against M4 comes from (E1).**
   - It splits roughly in half: the fc_mid trim (−2.53) and the quantum branch on top of it (−2.12).
   - Neither half is detected on its own at n = 10 (MDE ≈ 3.5 pp), although their sum is.
   - The honest statement: part of BFR's loss against M4 is a cost of the narrower classical parameterisation, not of the quantum branch, and the split is not resolved.
3. **Failure mode (exploratory, uncorrected).**
   - The bounded product term lowers training fit (C3 fit part −4.10 [−7.71, −0.49], p = .03) without widening the generalisation gap (BFR-ZZ has the smallest gap of all five models, 23.1 pp).
   - The original ZZ head (M2) does the opposite: train 94.0%, gap 47.0 pp.
   - So a bounded product term makes the head harder to fit, while M2's failure is overfitting. This is consistent with **unboundedness, rather than the product structure, driving M2's generalisation failure**. It is exploratory: not pre-registered, n = 10.
4. **One seed carries much of C3.** Seed 49 gives BFR-ZZ − BFR = −12.4 pp (BFR-ZZ 49.2%). Without it, C3 is −0.2. Its training fit fell, the same seed-level fitting failure seen for BFR at seeds 43 and 51 in V2. Reported, not removed.
5. **Gate and scale.** BFR-ZZ's quantum branch is as active as BFR's: median |g|·‖W_q‖ is 5.29 against 5.25. Its learned s stays closer to π/2 (median 1.43 against 1.18).

### 11.5 What it adds to the thesis

The "diagnose → redesign → test" arc now reads:
- **v16a:** the ZZ head's cost belongs to its classical feature map.
- **v16b:** bounding the encoding and making the branch a gated residual recovers 75–95% of that cost, but never beats M4.
- **v16b-ext:** at K10 N50, the recovery is not explained by removing the product term (restoring it on bounded inputs costs < 5 pp, point −1.45). About half of BFR's remaining deficit against M4 is plausibly the trimmed classical head rather than the quantum branch (not resolved at n = 10).

**The working conclusion:** the input bound matters more than the product structure. Testing it directly would need an *unbounded* BFR variant (u = s·z), which is not run and is future work.

### Supervisor summary (ext)

> To test whether removing the ZZ product term is what fixed the quantum head, we put it back (on the bounded inputs) and changed nothing else. At 10 classes and 50 images per class this cost only −1.5 pp, not significant, and a loss of 5 pp or more is ruled out. The redesigned head still recovered about two-thirds of the original head's deficit. So the improvement comes from bounding the encoding and gating it, not from dropping the product term. A second control, the classical head trimmed exactly as in the redesign, shows that about half of the redesign's remaining 4.7 pp shortfall against the classical baseline is plausibly due to the trim rather than the quantum branch. Neither half is significant on its own at 10 seeds.
