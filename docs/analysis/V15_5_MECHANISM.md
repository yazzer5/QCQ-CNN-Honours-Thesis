# v15.5 mechanism: why D1 and D7 look the way they do, and what the front-end claim can be

**Written:** 2026-10-26, after V15_5_RESULTS and V16_SYNTHESIS. **Everything here is exploratory and post hoc.** It explains the v15.5 results; it does not change any pre-registered verdict.

> **Later:** the classical control proposed here was run on Gadi as v15.8 (`results/V15_8_RESULTS.md`) and confirmed the local prediction M6 ≥ M3 > M4 at K = 5.

**Two kinds of evidence:**
- **(R) Real v15.5 data:** per-seed analysis of the stored runs.
- **(L) A local mechanism check:**
  - CPU only; not the Gadi protocol and not pre-registered.
  - mlxtend's bundled 5,000-image MNIST sample: K = 5 subset A, 300 train / 75 val / 125 test per class, 10 seeds.
  - The real `ClassicalBackbone` and the real Circuit-4 code (`quanv_torch.py`), with a w15-shaped linear head.
  - Test noise uses v15.5's seeds.

  (L) cannot be pooled with v15.5. It is the M6 control from V16_SYNTHESIS §6 in miniature, and it reproduces v15.5's pattern (deployed σ = 0.3: M3 77.0 vs M4 66.1, T = +8.8).

**Correction to V15_5_RESULTS L4.** Circuit 4 does not output cos(πx). Its channel b is exactly A_b·cos(πx + w1 + φ) (v16c F1): a cosine of the pixel with a **phase set by the weights**. Channel a is a similar function of its own pixel and its row partner. "Flat at black and white" holds only when the phase δ = w1 + φ is near 0 or π. Everything below uses the exact form.

---

## 1. The front-end claim (the "why" under everything else)

**Claim.** The trained front-end is robust because the angle encoding turns each pixel into a cosine whose slope is near zero at the pixel values MNIST actually uses. Training moves the circuit's phase toward that flat setting. The property is classical; a fixed cos(πx) map has it too.

**Step 1: where the pixels are.**
- In 5,000 real MNIST digits, **91.2% of pixels are within 0.1 of black or white** (80.7% exactly 0).
- Clipped Gaussian noise on a black pixel is one-sided. At σ = 0.3 it adds a **mean of +0.12** (σ/√(2π)) and an SD of 0.175. Strokes near 1 are dimmed by the same amount.
- So pixel noise is mostly a small, **one-sided shift away from 0 and 1**, not symmetric jitter.

**Step 2: what each front-end does with a small shift u away from the endpoints.**

| Front-end | Output change for a small shift u | At σ = 0.3, mean background shift |
|---|---|---|
| Linear (M4: 2×2 average, then replicated) | first order: u (the average keeps the mean, halves the SD) | 0.12 |
| Fixed cos(πx), flat at 0 and 1 | **second order:** (π²/2)·u² | ≈ 0.11, falling to ≈ 0.012 at σ = 0.1 (rescaled to [0, 1]) |
| Circuit channel, phase δ | first order: A·π·\|sin δ\|·u | grows with \|sin δ\| |

A cosine with δ = 0 has zero slope at x = 0 and x = 1, so small perturbations there are suppressed quadratically. A random-phase circuit (M5) has a first-order slope ∝ |sin δ|, so it passes noise through almost like a linear map, and it also has less black-to-white contrast (2A|cos δ|).

**Step 3: why training finds the flat setting, even with clean data.** For a cosine over [0, π], the contrast between a black and a white pixel is 2A|cos δ|. That contrast is largest exactly when δ = 0 or π, and that is also where the slope at both endpoints is zero. **Making black and white maximally distinguishable and making them insensitive to small perturbations are the same condition.** So ordinary clean training, which rewards separating ink from background, pushes the circuit toward the noise-insensitive setting without ever seeing noise.

**Evidence:**
- **(L) Training flattens the circuit.** Channel b's relative slope at black pixels, |sin δ|, falls from 0.59 at init (= M5) to 0.28 after clean training (= M3). It falls in 9 of 10 seeds; Wilcoxon p = .006. Seed 46 goes from 0.99 to 0.34, and its dead amplitude recovers from A_b = 0.06 to 0.65. That is the first direct observation of the v16c "rescue", in the local runs.
- **(R) The flatter a frozen circuit, the more robust.** Across the 10 real M5 seeds, the frozen circuit's slope at black pixels (relative to its contrast) predicts its measured deployed loss at σ = 0.3:
  - K = 5: Spearman ρ = +0.78, p = .008
  - K = 10: ρ = +0.75, p = .013
  - without seed 46: +0.70 and +0.65

  The front-end signal-to-noise ratio at σ = 0.3 gives ρ = +0.75 / +0.89. This uses the real init weights (G-b-verified) and the real accuracies.
- **(L) It is the cosine map, not resolution.** M3 keeps all four pixels of each 2×2 patch as channels, while M4 averages them. A 2 × 2 factorial separates the two (deployed, σ = 0.3, paired over 10 seeds):

| Contrast | Effect at σ = 0.3 | σ = 0.5 |
|---|---|---|
| cos vs linear, 4 pixels kept (COS4 − LIN4) | **+18.5 pp** [+11.7, +25.3], p = .0002 | +19.1, p = .001 |
| cos vs linear, averaged (COSAVG − M4) | **+20.0 pp** [+14.0, +25.9], p < .001 | +19.5 |
| 4 pixels vs averaged, linear (LIN4 − M4) | +1.0 [−8.1, +10.1], p = .81 | −1.0 |
| 4 pixels vs averaged, cos (COS4 − COSAVG) | −0.5 [−3.3, +2.4], p = .72 | −1.4 |
| trained circuit vs fixed cos (M3 − COS4) | **−8.7** [−18.1, +0.8], p = .07 | −6.9 |

**What follows:**
- The whole robustness effect is the cosine pixel map. Keeping resolution contributes nothing.
- A **fixed, parameter-free cos(πx) is at least as robust as the trained circuit**, and probably more (−8.7 pp, p = .07). Training gets the circuit only part of the way to the flat setting (|sin δ| = 0.28, not 0).
- With noise in training (matched, σ = 0.3), everything converges: M3 − M4 = +1.0 (n.s.), M3 − COS4 = 0.0. That matches v15.5's small matched effect at K = 5 (+0.98).

**Literature fit:**
- Models built from angle encoding plus a measurement are truncated Fourier series of the input (Schuld, Sweke & Meyer 2021). The front-end here is degree 1 per pixel, which is exactly the closed form of v16c F1.
- Berberich et al. (2024, *Phys. Rev. Research*) show a quantum model's robustness to input perturbations is controlled by its data encoding, through a Lipschitz bound.
- There is a useful twist. cos(πx) has a *global* Lipschitz constant of π, three times the linear map's, so a global bound would predict *worse* robustness. What matters is the **local slope at the operating points of the data**, which is zero here.
- The effect is a soft version of "feature squeezing" by bit-depth reduction (Xu, Evans & Qi 2018). Squashing near-binary inputs toward their extremes removes small perturbations before the network sees them.

**Thesis wording:** "Angle encoding with Z readout acts as a soft binariser: each pixel enters as a cosine that is flat at black and white. Clean training moves the trainable circuit toward this flat setting, which explains its robustness to pixel noise and why the frozen random circuit lacks it. A fixed classical cos(πx) map reproduces the effect locally, so it is an inductive-bias effect of the encoding, not a quantum advantage."

---

## 2. D1: why each curve does what it does

**(a) Why models trained on clean images collapse under noise.**
- The models have only ever seen backgrounds at exactly 0. At σ = 0.3 the background brightens by 0.12 on average over ~80% of the image, and strokes dim by the same amount.
- Every 3×3×4 kernel of conv1 integrates that shift over 36 inputs. Units that were silent on the background switch on, and the noise reaches the head as a systematic feature shift, not as averaged-out jitter.
- This is the classic covariate-shift failure: networks trained on clean images are fragile to noise they never saw (Geirhos et al. 2018; Hendrycks & Dietterich 2019).
- The curves drop steeply between σ = 0.2 and 0.5 because that is where the shift becomes comparable to the stroke contrast.

**(b) Why training on the noise recovers almost everything.** The matched model's training distribution equals its test distribution, so the shift is no longer a shift; the network simply learns the brightened background. Training with input noise is also equivalent, to first order, to Tikhonov regularisation: it penalises the squared input gradient of the network (Bishop 1995), which makes the learned function smoother around the data.

**(c) Why the QNN-head models (M1, M2) lose the most when trained clean, and gain from mild noise.**
- v16c F2 shows the ZZ head runs at a **fixed high gain of about 9 rad per unit of fc2 output**, with all classes packed into one period. Any shift in fc2's output is multiplied by that gain. The same front-end therefore loses 26 pp through the QNN head and 10 pp through the classical head at σ = 0.3 (BE = +18 pp).
- In the matched arm, Bishop's equivalence says noise training penalises exactly that input gain. That is the component that over-fits (v14 L3, the generalisation gap). Mild noise acts as the regulariser the head was missing: M1 and M2 gain 1.5–4 pp at σ = 0.1–0.2, and the head's clean-data cost disappears (BE ≈ 0).

**(d) Why M3 loses the least:** §1. Its trained circuit is partly flat at black and white (|sin δ| ≈ 0.28), so the one-sided background shift is suppressed before conv1.

**(e) Why M5 is fragile, and so spread out across seeds:** its frozen phases are random. Seeds whose circuit is steep at black pixels lose the most (ρ ≈ +0.75 to +0.89, §1). Seed 46 adds a dead channel on top (v16c F1).

**(f) Why M4 sits in the middle:** linear transfer of the shift, partly offset by the 2×2 average halving the zero-mean part. It has no flat region.

---

## 3. D7: why salt-and-pepper gives the same ranking, and how far the mechanism explains it

**What is different.** Salt-and-pepper sets pixels **exactly** to 0 or 1. Most hits change nothing: salt on a white stroke, pepper on the black background. The rest are **full flips** (background → white, stroke → black). A flip is a full-contrast change in every pixel map: linear, cos, or any phase. **Per-pixel flatness gives no protection against flips.** The small-perturbation argument in §1 does not apply.

**What the data show:**
- **(R)** M3 still leads M4 (+8.2 pp at p = 0.2, K = 5), but the margin is smaller than for Gaussian noise at a similar accuracy loss.
- **(R)** The frozen M5 does unusually well: 73.7 vs M4's 70.3 at K = 5, p = 0.2, although M5 is worse than M4 under Gaussian noise. Training the circuit no longer matters: T_sp is not detected.
- **(L)** The cos map still helps (COS4 − LIN4 = +10.4 pp, p = .009, about half its Gaussian benefit). Resolution again does nothing (−2.6, n.s.). Trained vs fixed cos is equal (−0.7), M5 ≈ COS4 (72.8 vs 72.4), and T = −1.0, n.s.

**Reading:**
- The part of the robustness that depends on the circuit's *phase* (flatness, and therefore training) disappears for salt-and-pepper, exactly as the mechanism predicts. That is why M5 catches up and T vanishes.
- What remains, about +10 pp for any cosine map whatever its phase, cannot come from per-pixel transfer, which is identical for flips. It must come from how the network downstream of a cosine map responds to isolated full-contrast spots. **This part is not explained here.**
- One hypothesis to test: the cosine map compresses the gray stroke-edge pixels, so the trained backbone relies on coarser stroke structure and is less moved by isolated spots. A check would compare the input-gradient maps of COS4 and LIN4 networks.
- Note that the models here were never trained on noise. Geirhos et al. (2018) show the related fact that networks trained on one noise type gain little robustness to another, so the Gaussian and salt-and-pepper results are best read as two separate robustness properties.

---

## 4. Caveats

- §1–§3 are post hoc. The local check (L) uses a 5,000-image MNIST sample, K = 5 subset A only, a w15-shaped head whose exact activation may differ from the Gadi code, and 10 seeds. It is not pre-registered and not pooled with v15.5.
- The per-seed correlation (R) uses 10 seeds. It is strong and survives dropping seed 46, but it is one exploratory test.
- The flattening analysis looks at channel b only, where a closed form exists. Channel a mixes two pixels.

## Sources

- [Schuld, Sweke & Meyer 2021, The effect of data encoding on the expressive power of variational quantum machine learning models](https://arxiv.org/abs/2008.08605v2)
- [Berberich et al. 2024, Training robust and generalizable quantum models](https://arxiv.org/abs/2311.11871v1)
- [Bishop 1995, Training with noise is equivalent to Tikhonov regularization](https://mlanthology.org/neco/1995/bishop1995neco-training)
- [Geirhos et al. 2018, Generalisation in humans and deep neural networks](https://arxiv.org/abs/1808.08750v3)
- [Xu, Evans & Qi 2018, Feature Squeezing (NDSS)](https://www.cs.virginia.edu/~evans/pubs/ndss2018)
- Hendrycks & Dietterich 2019, Benchmarking neural network robustness to common corruptions and perturbations (ICLR).
