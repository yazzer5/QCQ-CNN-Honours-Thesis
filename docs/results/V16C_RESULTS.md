# v16c results: diagnostics (front-end conditioning, head frequency)

**Status:** complete. No training and no Gadi jobs. **Everything in this document is exploratory.** It tests the proposed mechanisms; it does not replace or modify any pre-registered primary of v13–v16b.
**Pre-registration:** `PREREGISTRATION_v16c.md`. Written before any input was opened; amendments A1–A4 were logged before the quantities they affect were computed.
**Date:** 2026-10-22. **Figures:** `figures_v16c/` (200 dpi). **Script:** `v16c_diagnostics.py` (`partA`, `partB`). **Numbers:** `v16c_partA.json`, `v16c_partB.json`.

> **Gate update (2026-10-22, after this file was written):** gate G-b **passed** on Gadi (torch 2.5.1+cu121). The 60 Circuit-4 init weights reproduce the local digest `5dba16aa2f1d12ab` bitwise, so Part A below is final, no longer provisional. See `V16_SYNTHESIS.md`.

## 1. Answer in one paragraph

**All five pre-registered predictions fail or are only directional. The two post-hoc explanations they turned up are sharper than the predictions were.**

**Seed 46.** Its frozen front-end is not badly *conditioned* in the graded sense C1/C2 assumed. It is **half dead**. Two of the four Circuit-4 output channels (⟨Z_b⟩ for the top and bottom pixel pairs) barely respond to the image: output std 0.017 against 0.22–0.63 for every other seed. The cause is exact and closed-form. Channel b equals A_b·cos(θ_b + w1 + φ), with amplitude **A_b = √(cos²w4 + sin²w4·cos²(w2/2))**. A_b is zero when w2 = π and w4 = 3π/2. Seed 46's random init lands 0.11 rad and 0.03 rad from that point, giving A_b = 0.063, against 0.76–0.98 for the other nine seeds.

**The QNN head.** The trained network never lets z drift. M2 keeps fc2's output inside |z| ≤ 2, and across 90% of test images the ZZ product phase covers **less than one period (0.80–0.94)**, at N = 50 and N = 500 alike. So the head's local frequency is essentially fixed at about 9 rad per unit z (2π√2 at z = 0). It does not grow with data scarcity (H1 fails), and it does not mark the errors (H2, H3 fail). The diagnosis "unbounded z → frequency grows → low-data collapse" is not supported. What the data show is a **high, fixed gain**: the network packs 10 classes into one period of a steep periodic map.

## 2. Integrity

**Part B (B0): all 12 runs pass.**
- Predictions, labels and accuracy match each result JSON.
- z hashes match the v16a run files.
- The test set is the same 2,000 images in all 12 runs (A1).
- `qnn_torch.py` feeds fc2's raw output to `P(2(π−x0)(π−x1))`, so f was computed on the right quantity (A2 resolved).

**Part A.** The pre-registered A0 gate (exact match to stored M5 parameters) could not run: **no stored front-end parameters exist** (A4.1). The replacement gate has three parts:

| Gate | Result |
|---|---|
| **G-a** code path | **Pass, 10/10 seeds.** `build_model("M5", s).front_end.weights` is bitwise equal to the direct init, bitwise unchanged under global seeds 0 and 999, and `requires_grad=False`. The init uses a private `torch.Generator().manual_seed(seed)`. |
| **G-c** forward | **Pass.** The analysis forward equals the code's `front_end.forward` to 3.9e-7. Autograd Jacobian vs central differences: 1.7e-9. |
| **G-b** training environment | Pending when written; **passed 2026-10-22** (see note above). Local digest of the `float.hex` strings: **`5dba16aa2f1d12ab`**. |

**Other checks:**
- Patches come from the code's own `load_dataset` (K = 10, N = 500, seed 42). Its train hash **`98a3182ce94eff49` equals the hash in the stored Gadi log** of the v16a z-job at that cell and seed.
- J's off-diagonal blocks are exactly 0 for every seed and patch (A4.3).
- The C2 cells are as in A4.4. The v15 `test_acc` equals the matched `acc_eval` column, and the v14 N = 500 rows equal v13.5 w15 (duplicates dropped).

## 3. Part A: front-end conditioning

10,000 patches, the same for every seed. **74% are blank (all four pixels 0).**

| Seed | median σ_min (primary) | median σ_min/σ_max | frac σ_min < 0.05 | min channel var | channel std [Za Zb Za Zb] | A_b (post hoc) | M5 score |
|---|---|---|---|---|---|---|---|
| 42 | 0.585 | 0.710 | 0.015 | 0.103 | .39 .32 .40 .32 | 0.81 | −0.58 |
| 43 | 0.308 | 0.335 | 0.016 | 0.050 | .22 .62 .22 .63 | 0.98 | −0.52 |
| 44 | 0.571 | 0.706 | 0.016 | 0.073 | .27 .51 .27 .52 | 0.93 | −0.17 |
| 45 | 0.381 | 0.711 | 0.015 | 0.242 | .49 .54 .49 .54 | 0.97 | +0.21 |
| **46** | **0.060** | **0.066** | **0.201** | **0.0003** | .28 **.017** .28 **.017** | **0.063** | **−2.07** |
| 47 | 0.311 | 0.748 | 0.024 | 0.272 | .52 .60 .52 .60 | 0.94 | +1.02 |
| 48 | 0.639 | 0.914 | 0.056 | 0.052 | .23 .51 .23 .52 | 0.95 | +1.18 |
| 49 | 0.079 | 0.149 | 0.087 | 0.148 | .59 .39 .59 .38 | 0.84 | +0.10 |
| 50 | 0.160 | 0.194 | 0.022 | 0.244 | .49 .54 .50 .54 | 0.84 | +0.45 |
| 51 | **0.025** | **0.034** | **0.824** | 0.048 | .37 .22 .37 .23 | 0.76 | +0.39 |

M5 score is the mean within-cell z of M5 test accuracy over the 10 unique K = 10 cells of v13.5, v14 and v15.

**C2, Spearman ρ vs M5 score (one-sided p):**

| Metric | All 10 seeds | Without 46 |
|---|---|---|
| median σ_min (primary) | +0.18 (p = .31) | −0.03 (.53) |
| median σ_min/σ_max | +0.43 (.11) | +0.32 (.20) |
| frac σ_min < 0.05 (sign flipped) | −0.22 (.73) | −0.58 (.95) |
| min channel variance | +0.47 (.087) | +0.27 (.24) |
| A_b (post hoc) | +0.32 | +0.07 |

Per cell, the primary ρ is positive in 9 of 10 cells. The cells share seeds, so this is not an independent count.

**Non-blank patches only (descriptive sensitivity):** seed 46 is the worst seed by a wide margin, with median σ_min 0.011 against the next lowest 0.079, and 78% of patches below 0.05.

## 4. Part B: head frequency (K = 10, seeds 42–44, 2,000 test images)

**z scale.**

| | M2 (feeds the ZZ map) | M3 (classical head, reference) |
|---|---|---|
| max \|z\| | 1.0–2.0 | 12.7–27.1 |
| IQR | 0.36–0.64 | 5.4–10.9 |
| f per image | 8.3–11.2 | 0–60 |
| ZZ product-phase periods across the middle 90% (post hoc) | **0.80–0.94** | **25–80** (if fed to the map) |

**H1. Median f, N = 50 vs N = 500.**

| | Seed 42 | Seed 43 | Seed 44 |
|---|---|---|---|
| M2 | 9.40 vs 9.64 | 8.64 vs 8.62 | 8.99 vs 9.06 |
| ratio | 0.98 | 1.00 | 0.99 |
| M3 pseudo-f ratio | 1.15 | 0.95 | 0.83 |

M2's median f is higher at N = 50 in 1 of 3 seeds.

**H2. AUC = P(f_error > f_correct), M2.**

| N | Mean AUC | Per seed | Stouffer |
|---|---|---|---|
| 50 | 0.507 | 0.568, 0.487, 0.466 | Z = 0.87, p = .19 |
| 500 | 0.477 | | Z = −2.51, p = .99 |

M3 reference: AUC 0.394 and 0.365. M3's errors have *lower* pseudo-f.

**H3. M2 error rate by f quartile, mean of 3 seeds (%).**

| N | Q1 (lowest f) | Q2 | Q3 | Q4 (highest f) | Q4 > Q1 | Trend (Stouffer) |
|---|---|---|---|---|---|---|
| 50 | 45.0 | 66.3 | 61.3 | 47.6 | 1 of 3 runs | p = .29 |
| 500 | 16.1 | 28.7 | 30.9 | 9.9 | 0 of 3 runs | p = 1.00 |

Overall, Q4 > Q1 in 1 of 6 runs. Every run is an inverted U, and M3's pseudo-f gives the same shape.

## 5. Verdicts

| Prediction | Verdict |
|---|---|
| **C1** seed 46 has the lowest median σ_min | **Fails.** Seed 46 ranks 2nd; seed 51 is lowest. On the secondaries, 46 is worst on min channel variance only (1/3; "partial" needed 2/3). |
| **C2** conditioning correlates positively with M5 accuracy | **Direction only.** ρ = +0.18, p = .31. Without 46, ρ = −0.03. |
| **H1** M2 median f higher at N = 50 | **Fails** (1/3 seeds; ratios 0.98–1.00). |
| **H2** M2 errors have higher f | **Fails** at both N (p = .19, .99; AUC ≈ 0.5). |
| **H3** M2 error rises across f quartiles | **Fails** (Q4 > Q1 in 1/6; inverted U). |

## 6. Findings and why

### C1/C2 failed because the primary metric measured one point, and seed 46's defect isn't graded

- **The primary was a single point.** 74% of patches are blank, so for every seed the median σ_min over all patches is *exactly* σ_min at the blank patch (checked: equal to 1e-12). The primary measured the circuit's sensitivity at one input. Seed 51 is poorly conditioned there but normal on real strokes (non-blank median 0.147). This is a flaw in how I specified the primary. Keeping blank patches was justified as "the distribution the front-end sees", but a median over that distribution reduces to the blank patch.
- **Graded conditioning doesn't predict accuracy.** Among the nine normal seeds, no conditioning metric tracks M5 accuracy (ρ between −0.58 and +0.32). Seed 46's failure is categorical: one channel pair switched off.

### F1. Seed 46's frozen front-end is half dead (post hoc, exact; `figA_dead_channel_posthoc.png`)

**What happens.** For qubit b of each 2-qubit block, the code's output is exactly

  ⟨Z_b⟩ = cos(w4)·cos(θ_b + w1) ± sin(w4)·cos(w2/2)·sin(θ_b + w1),

verified to ≤ 1e-15 on all 10,000 patches for all 10 seeds. It depends only on its own pixel, with amplitude A_b = √(cos²w4 + sin²w4·cos²(w2/2)).

**Why.** RY(w4) decides which component of qubit b the Z measurement reads.
- At w4 = 3π/2 it reads only the X component.
- The CRZ(w2) gate, controlled by qubit b, rotates qubit b's coherence by ±w2/2 depending on qubit a. At w2 = π that pushes the X component entirely into Y, which an RY-then-Z readout cannot see.
- Seed 46 has w2 = π + 0.11 and w4 = 3π/2 + 0.03, so A_b = 0.063.

**Counterfactual on seed 46's own weights.** The Z_b std over the patches is:

| Weights | Z_b std |
|---|---|
| actual | 0.017 |
| w2 = π, w4 = 3π/2 exactly | 9e-17 |
| w2 moved to π/2 | 0.18 |
| w4 moved to π | 0.63 |

No other seed has both factors small. The smallest products of |cos(w2/2)| and |cos(w4)| are seed 47 (0.10 × 0.94) and seed 51 (0.17 × 0.75).

**What it explains, one mechanism for four earlier observations:**
- **v13.5 R3, v14 L5, v15 L4: worst M5 everywhere (z ≈ −2.3).** The frozen model sees 2 informative channels instead of 4.
- **The rescue by training (M3 seed 46 unremarkable).** Moving either weight a modest distance restores the channel, as the counterfactual shows. *Not checked against the trained M3 weights, which aren't stored.*
- **v15 L4: seed 46 is the only clean-trained model that breaks at device noise.** The front-end noise contraction at α = 1 is about 0.04 on the [−1, 1] scale (v15 L1). That is comparable to channel b's entire signal range here (about 0.07). Noise therefore swamps half the input.
- **v15.5 L5: seed 46's M5 is the most fragile to pixel noise.** The same account applies, with the two remaining live channels carrying everything.

**What it does not explain:** v15.5's seed-46 **M4** cap hits. M4 has no front-end, so those come from seed 46's data draw or shuffle.

**Thesis wording.** "A random Circuit-4 initialisation can switch off half of the front-end's channels. Seed 46 is such a case, by an exact closed-form criterion. Training the front-end repairs it." This sharpens v13.5's "robustness to initialisation" into a specific, checkable failure mode. It also predicts something testable: front-ends initialised near (w2, w4) = (π, 3π/2), or (π, π/2) by the same algebra, will reproduce seed 46's behaviour.

### F2. The ZZ head runs at a fixed high gain, not a growing frequency (Part B; `fig3_f_histograms.png`)

**What happens.**
- M2's fc2 output stays roughly 10× tighter than M3's.
- All 10 classes sit inside one period of the ZZ product phase (0.80–0.94 periods across 90% of test images).
- The same holds at N = 50 and N = 500 and in every seed.
- The local frequency f is therefore pinned near 2π√2 ≈ 8.9 rad per unit z. Its spread within a run is about ±1.

**Why H2 and H3 fail.**
- With |z| ≪ π, f ≈ 2π√2 − √2·(z0 + z1) to first order. Sorting by f is really sorting by one direction in z-space.
- The extremes of that direction are the well-separated classes, and the middle is where classes overlap. That produces the inverted U.
- M3 shows the same shape, so it is geometry, not the ZZ map (the pre-registered specificity reading).

**What it means for the diagnosis.**
- The v16 plan's mechanism, "z unbounded, so frequency grows with |π − z|", is not what happens: the network never uses the unbounded range.
- v16a still stands: the cost belongs to the 12-feature map, not to anything quantum.
- A better description of the map's problem is **high, fixed gain on a periodic map**. A 0.1 change in z moves the phase by about 0.9 rad, so 10 classes must be packed into one steep period.
- This fits v14 L3 (fits fine, generalises poorly, worse with less data) and v15 L3 (amplifies front-end noise). It is a hypothesis generated post hoc, not a tested one.

### F3. Consequence for reading v16b (post hoc)

- BFR's features vary at most about 2s rad per unit z: degree ≤ 2 in u = s·tanh z, with |du/dz| ≤ s.
- BFR's learned s fell to 0.88–1.18 in every run, which gives about 1.8–2.4 rad per unit z, roughly 4–5× lower gain than the ZZ map's ~9.
- M2 already keeps itself bounded in practice, so BFR's recovery of 75–95% of the cost is better attributed to **lower gain (plus gating and the residual)** than to "bounding an unbounded frequency".
- **Suggested edit to V16B_RESULTS §7.1:** replace "unbounded frequency and the product term" with "a high fixed gain and the product term (v16c F2: M2 never uses the unbounded range)".
- BFR-ZZ remains the test that separates the product term from gain.

## 7. What it means for the thesis

1. **Seed 46 is solved.** It is a specific, exactly characterised dead-channel initialisation, not a vaguely "bad" one. This is the cleanest mechanistic result in v16 and costs no compute. It belongs in the front-end chapter next to v13.5 R3, as an explanation of why training the front-end helps (T > 0) mostly through rescue.
2. **The head mechanism needs rewording, not retracting.** v16a's causal claim (the cost is in the classical feature map) is unaffected. The *how* changes from "frequency grows with |z|" to "a fixed, high-gain periodic map that forces classes into one period". Report H1–H3 as failed predictions alongside the post-hoc replacement, labelled post hoc.
3. **Method lesson (record in findings).** A median over a distribution dominated by one value is that value. Conditioning metrics on image patches must be stratified by background, or computed on non-blank patches.

### Supervisor summary

> We ran two diagnostic analyses with no new training. First, the seed-46 front-end that is the worst frozen model in every experiment: it is not just poorly conditioned. Two of its four output channels are effectively switched off. We derived the channel's response in closed form and verified it exactly against the code. Its amplitude is √(cos²w4 + sin²w4·cos²(w2/2)), which vanishes when w2 = π and w4 = 3π/2, and seed 46's random initialisation lands within 0.11 and 0.03 rad of that point (amplitude 0.06, against 0.76–0.98 for every other seed). This one fact explains its poor frozen accuracy, why training rescues it, and why it alone breaks under hardware noise, since the noise is as large as that channel's whole signal. Our pre-registered conditioning predictions failed: the metric we chose was dominated by blank patches, and among normal seeds conditioning does not predict accuracy. Second, the quantum head: we predicted its effective frequency would rise with less data and mark the errors. It does neither. The network keeps the head's inputs within one period of the ZZ encoding at every data size, so the head works at a fixed, high gain (about 9 rad per unit input) rather than an exploding frequency. The v16a conclusion stands (the head's cost lies in its classical feature map), but the mechanism should be described as high gain, and the redesigned head's gains in v16b are better attributed to its roughly 4–5× lower gain than to bounding. All of this is exploratory.

## 8. Caveats

- **Everything here is exploratory.** F1–F3 are post hoc. No Holm correction; raw p-values with the number of tests stated.
- The replacement gate proves the analysis uses the weights the code generates, not weights read back from a finished run (A4.1).
- Part B has 3 seeds at K = 10 only. H1 could reach at most p = .125.
- Seed 46's head frequency is not observed: there are no z-dumps for seed 46.
- The rescue claim (training moves w2/w4 off the critical point) is inferred from the counterfactual. Trained M3 front-end weights are not stored. One `--save_z`-style dump of M3's trained `front_end.weights` at seed 46 would confirm it.
- F3's gain figure for BFR uses the learned s from V16B §V1 and the degree-2 bound. It is an upper bound on the slope, not a measured per-image quantity.

## 9. Figure map (`figures_v16c/`)

| Figure | Shows |
|---|---|
| `fig1_conditioning_by_seed.png` | Four conditioning metrics per seed, 46 in red. 46 is extreme on min channel variance (0.0003); 51 is lowest on the blank-dominated σ_min metrics. |
| `fig2_conditioning_vs_m5.png` | C2: primary metric vs M5 score. No relationship apart from 46 (ρ +0.18; −0.03 without 46). |
| `fig3_f_histograms.png` | M2's f is a narrow spike near 9 at both N; M3's pseudo-f is broad (0–60). The head works at fixed frequency. |
| `fig4_error_by_f_quartile.png` | M2 error by f quartile: inverted U at both N, same shape as M3's. Geometry, not frequency. |
| `figA_dead_channel_posthoc.png` | **Headline, post hoc.** Channel b's response to its pixel for each seed: seed 46 is flat. Amplitude A_b vs M5 score: 46 isolated at 0.06. |
