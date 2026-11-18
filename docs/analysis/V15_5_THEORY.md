# v15.5 theory: why the trained quantum front-end is robust to input noise

**Written:** 2026-10-27, after the supervisor meeting that made input noise the focus. **Exploratory and post hoc.** It builds on `V15_5_MECHANISM.md` and corrects it in one place (§4). Figure: `v155_theory.png`. Script: `v155_theory_fig.py` (closed forms, no training).

> **Later:** prediction P-T3 (§7) was run as v15.8 (`results/V15_8_RESULTS.md`): M6 ≥ M3 held at K = 5.

## 0. The answer in one paragraph

Each Circuit-4 output is a cosine of its pixel, cos(πx + δ), with a phase δ set by the weights. MNIST pixels sit almost all at 0 or 1, and clipped Gaussian noise moves them by a small, **one-sided** amount u ≥ 0. What hurts a clean-trained network is the **mean** shift this causes in the background (it adds up coherently over every conv kernel), not the jitter. For a linear input map that shift is σ/√(2π). For a cosine it has two parts: a **squash** term, second order in u, that is always smaller than the linear shift for σ ≲ 0.5, and a **tilt** term, first order in u, proportional to tan δ. The squash term alone (δ = 0) predicts M3's whole deployed curve at K = 5 from M4's curve **with no fitted parameter** (§3). The tilt term has a **sign**: it amplifies noise for one sign of δ and cancels it for the other. Clean training raises black–white contrast, which shrinks |tan δ| from either side, so it removes the worst (amplifying) circuits but does not pick the protective sign. That explains why training is needed (T > 0), why the frozen circuit is so seed-dependent (M5 SD 15–18 pp), and why a fixed classical cos(πx) matches or beats the trained circuit. Nothing in it is quantum: it is the inductive bias of angle encoding.

## 1. The front-end in closed form

Circuit 4 on one row pair (a, b), angles t_a = πx_a + w0, t_b = πx_b + w1; gate order RY RY · CRZ(w2, control b) · RY(w3) RY(w4) · CRZ(w5). The last CRZ is diagonal, so it never affects Z readout.

- **Channel b** (code-verified in v16c F1):
  ⟨Z_b⟩ = cos w4 · cos t_b − sin w4 · cos(w2/2) · sin t_b = A_b · cos(πx_b + δ_b)
  with A_b = √(cos²w4 + sin²w4·cos²(w2/2)) and δ_b = w1 + φ_b.
- **Channel a** (derived, and checked against a numpy statevector of the same gate order to 1e-15 over 2,000 random weight/pixel draws; verified against the code in `FRONTEND_MATH.md` §6):
  ⟨Z_a⟩ = cos w3 · cos t_a − sin w3 · sin t_a · [cos²(w2/2) + sin²(w2/2) · cos t_b]
  = a cosine of its own pixel **plus one product term** sin t_a · cos t_b, the only trace of the entangling gate.
- So every channel is a degree-1 trigonometric polynomial in each pixel, as Schuld, Sweke & Meyer (2021) predict for single-layer angle encoding. Everything below uses the single-pixel cosine; the product term is a caveat (§6).

## 2. Noise reaching the network

**Noise model.** Background pixel x = 0, corrupted to x̃ = clip(σε, 0, 1) = u. Half the time u = 0; otherwise u is half-normal. E[u] = σ/√(2π), E[u²] = σ²/2. Strokes at x = 1 are mirrored.

**Why the mean matters, not the variance.** conv1's 3×3 kernel over 4 channels sums 36 inputs. A per-pixel mean shift b adds up to 36·b; independent jitter of SD s adds up to only 6·s. At σ = 0.3 (b = 0.12, s = 0.175) that is 4.3 against 1.05. To a clean-trained network the noise is mostly a **uniform brightening of the background**, a covariate shift.

**Exact background mean shift, as a fraction of black–white contrast** (cosine channel cos(πx + δ); upper clip ignored, exact to < 10⁻³ for σ ≤ 0.3, Monte-Carlo verified):

  **β(σ, δ) = ¼ (1 − e^(−π²σ²/2))  +  (tan δ / 2√π) · D(πσ/√2)**
              squash (2nd order)        tilt (1st order)

with D the Dawson function. Linear map: **β_lin = σ/√(2π)**.

| σ | linear | cos, δ = 0 | δ = −20° | δ = +20° |
|---|---|---|---|---|
| 0.1 | 0.040 | 0.012 | −0.010 | 0.034 |
| 0.3 | 0.120 | 0.090 | 0.038 | 0.141 |
| 0.5 | 0.200 | 0.177 | 0.123 | 0.231 |

- **Squash.** For small u, (1 − cos πu)/2 ≈ π²u²/4: second order. It beats linear by 3.3× at σ = 0.1, 1.3× at σ = 0.3, and the advantage disappears near σ ≈ 0.6, when noise pushes pixels off the flat end.
- **Tilt.** First-order gain (π/2)·tan δ per unit u. It has a sign. With the convention above, **sin 2δ > 0 amplifies background noise and sin 2δ < 0 cancels it** (it puts the background just past the cosine's extremum, so noise moves it *away* from ink). Strokes see the opposite sign, but they are only ~20% of pixels.

## 3. Test: a zero-parameter prediction of M3

**Method.** Treat M3's front-end as δ = 0. For each σ, find the linear noise level σ_eff with the same background shift, β_lin(σ_eff) = β(σ, 0). Predict M3's loss = M4's measured loss at σ_eff (interpolated from v15.5's own deployed table). No parameter is fitted.

| test σ | 0.1 | 0.2 | 0.3 | 0.5 |
|---|---|---|---|---|
| σ_eff | 0.030 | 0.112 | 0.225 | 0.457 |
| K = 5 predicted | 92.5 | 90.8 | **81.7** | 53.3 |
| K = 5 measured M3 | 92.4 | 90.0 | **82.5** | 57.9 |
| K = 10 predicted | 80.6 | 77.4 | 66.9 | 38.3 |
| K = 10 measured M3 | 80.9 | 74.6 | 61.1 | 34.2 |

- **K = 5: within 1 pp up to σ = 0.3**, the primary F_dep test point. The pure squash term accounts for the whole +9.1 pp.
- **K = 10: M3 falls 3–6 pp short of the prediction**, so its trained circuits sit on average short of the ideal flat setting, consistent with F_dep being smaller and not detected after Holm at K = 10.
- The prediction is not valid beyond σ ≈ 0.5: σ_eff then exceeds the measured M4 range (panel B/C divergence at 0.7–1.0 is extrapolation).

## 4. Why training matters, and a correction to V15_5_MECHANISM

**What clean training does to δ.** A channel's black–white contrast is 2A|cos δ|, largest at δ ∈ {0, π}, where tan δ = 0. Clean training rewards contrast (weight decay penalises the downstream weights a low-contrast channel would need), so it shrinks |tan δ| **from either side**. It has no reason to prefer the protective sign: clean images have no noise to protect against.

**Consequences:**
1. **M5 (frozen, random phases)** has seeds with large positive tilt (much worse than linear) and seeds with negative tilt (better than δ = 0). That is a direct explanation of M5's huge deployed seed SD (15–18 pp) and of R = M5 − M4 ≈ 0 on average.
2. **T = M3 − M5 is a rescue of amplifying seeds.** Training helps a lot where the init tilt was amplifying and little, or slightly negatively, where it was already protective.
3. **A fixed cos(πx) (δ = 0) is a ceiling for what clean training reaches**, so COS4 ≥ M3 (local −8.7 pp, p = .07) is expected, not surprising.

**Correction.** V15_5_MECHANISM §1 argued from the *unsigned* slope |sin δ| (0.59 → 0.28 under training; ρ ≈ +0.78 with M5's loss). The unsigned measure is not enough. At |sin δ| = 0.28 the background shift at σ = 0.3 is **0.049 with one sign and 0.131 with the other**, i.e. better than δ = 0 or *worse than no front-end*. The direction of the conclusion stands; the evidence for it needs the signed phase (P-T1, P-T5).

## 5. The matched arm (trained on noise)

- In the matched arm there is no distribution shift, so the question becomes how much class information survives the noise.
- For near-binary pixels the Bayes-optimal per-pixel denoiser E[x | x̃] is a sigmoid of x̃: flat at the ends, steep in the middle, the same shape as (1 − cos πx)/2. The cosine is a **fixed, parameter-free approximation of the optimal denoiser**, which a 1.7k-parameter network would otherwise have to learn.
- Per-pixel MSE to the clean image after the best affine rescale (binary pixels, 20% ink): σ = 0.1: linear 0.0039, cos 0.0007, optimal 0.0000; σ = 0.2: 0.0173, 0.0098, 0.0037; σ = 0.3: 0.0397, 0.0345, 0.0269; σ = 0.5: 0.0901, 0.0913, 0.0827.
- This predicts a small F_mat at σ ≤ 0.3 (measured +1.0 pp at K = 5) but **no per-pixel gain at σ = 0.5**, where F_mat is largest at K = 10 (+7.3 pp). **The K = 10 matched effect at high σ is not explained by per-pixel theory.** One candidate: with noisy training data the gradient *does* reward the protective tilt sign, which a trainable circuit can find and M4 cannot (P-T5).

## 6. What the theory does not cover

- **Salt-and-pepper (L6).** A flip changes a pixel by the full contrast through any map, so neither squash nor tilt applies. The remaining ~+10 pp for cosine maps is unexplained (V15_5_MECHANISM §3).
- **Strokes and the product term.** §3 uses background pixels and one cosine. The stroke side (20% of pixels, opposite tilt sign) and channel a's sin t_a·cos t_b term are ignored. That §3 fits K = 5 this well is partly fortunate.
- **The QNN head** (BE = +18 pp) is a separate mechanism: v16c F2's fixed ~9 rad/unit gain multiplies whatever shift reaches fc2.

## 7. Predictions that would make this citable

None of these needs new quantum training except P-T4.

| ID | Test | Cost | Prediction |
|---|---|---|---|
| **P-T1** | Recompute M5's four init phases per seed (G-a gives the exact init weights) and the signed tilt Σ tan δ weighted by A. Correlate with M5's measured deployed loss at σ = 0.3. | no jobs | Signed tilt predicts loss better than unsigned \|sin δ\| (ρ_signed > ρ_unsigned = 0.78). |
| **P-T2** | Per-seed T_dep against the init tilt sign. | no jobs | T_dep large where sin 2δ > 0, near 0 where sin 2δ < 0. |
| **P-T3** | M6: M4 fed cos(πx), zero parameters (V16_SYNTHESIS §6). | ~60 jobs | Deployed curve within ~1–2 pp of the §3 prediction at K = 5, σ ≤ 0.3; M6 ≥ M3. **Run as v15.8.** |
| **P-T4** | **Phase sweep (the decisive test):** M4 fed cos(πx + δ), δ ∈ {−20°, 0, +20°}, trained clean, cross-evaluated σ ∈ {0.1, 0.2, 0.3, 0.5}. | ~60 jobs | Strict order −20° > 0° > +20° at σ = 0.3 at both K. Background-only theory gives 92 / 82 / 67 at K = 5; the stroke side will shrink the spread, so pre-register the **order** and a minimum gap, not the numbers. |
| **P-T5** | Dump trained `front_end.weights` for M3 clean vs matched σ = 0.3 (one flag). | ~20 jobs | Matched-trained circuits have sin 2δ < 0 more often than clean-trained ones. |

P-T4 is the one to prioritise: it is fully classical, it tests the signed-tilt mechanism by dose-response, and a +20° result *worse than M4* would be a clean, counter-intuitive confirmation that the effect is the encoding, not "having a front-end".

## 8. Thesis wording

> "Angle encoding followed by Z readout maps each pixel to cos(πx + δ). For near-binary images under small one-sided pixel noise, this map transmits a background shift that is second order in the noise when δ = 0, against first order for a linear map; a non-zero phase adds a first-order term whose sign decides whether noise is amplified or cancelled. A zero-parameter calculation from this alone reproduces the trained front-end's deployed accuracy at K = 5 to within 1 pp for σ ≤ 0.3. Clean training increases black–white contrast and therefore drives the phase toward zero, which is why the trained circuit is robust and the frozen random circuit is not. The effect is an inductive bias of the encoding and is reproduced by a fixed classical cosine; it is not a quantum advantage."

## Sources

- Schuld, Sweke & Meyer 2021, *The effect of data encoding on the expressive power of variational quantum machine learning models*, arXiv:2008.08605.
- Berberich et al. 2024, *Training robust and generalizable quantum models*, arXiv:2311.11871 (encoding-controlled Lipschitz bounds; note the global bound is the wrong quantity here, the local slope at the data is what matters).
- Bishop 1995, *Training with noise is equivalent to Tikhonov regularization*, Neural Computation.
- Xu, Evans & Qi 2018, *Feature Squeezing*, NDSS.
- Geirhos et al. 2018, *Generalisation in humans and deep neural networks*, arXiv:1808.08750.
