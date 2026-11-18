# Pre-registration v15.7: does the noise bubble tighten earlier in training for M3?

**Written:** 2026-11-06, before any v15.7 code, job or result exists.
**Follows:** v15.6 (`PREREGISTRATION_v15_6.md`, `V15_6_RESULTS.md`).
**Motivation:** the supervisor's criterion: a good model forms a tighter, more consistent bubble around the true value, *faster* than the others. v15.6 measured only the final trained model. The best-epoch check (2026-11-06, exploratory) found no detected difference in epochs-to-best-accuracy (all 12 paired comparisons p ≥ .16). This experiment measures the bubble itself during training.

## 1. Question

**Q-v15.7.** During training, does M3's noise bubble around the true value become tight and centred in fewer epochs than M4's (no front-end) and M5's (frozen front-end)?

## 2. Models and protocol

M3, M4 and M5, exactly as in v15.5/v15.6: same code, same protocol, same seeds, same data. Training is not changed in any way.

## 3. Measurement

**Bubble radius.** For test image *i*, the R noisy versions land at points f(x̃ᵢᵣ), and the clean image lands at f(xᵢ).
- radius² = [ mean over i and r of ‖f(x̃ᵢᵣ) − f(xᵢ)‖² ] / (2 tr S_B)
- Equivalently, in v15.6 notation, radius = √((ηD)² + tr S_N) / D: shift and spread combined, in units of the distance between class centres.
- Smaller means a tighter bubble around the right point.

**Also recorded at each checkpoint:** η (shift), ρ_W (spread), bleed and error, using the v15.6 definitions and v15.6 code (`geom_v156.layer_metrics`).

**Where:** L0, L1, L2 (z) and L3. **Primary layer: z (L2).**

**Noise:** test σ = 0.3, plus σ = 0 for sanity. The same 50-per-class subset as v15.6. R = 10 draws: draw 0 is v15.5's fixed noisy test image (A1 of v15.6), and draws 1–9 are the v15.6 fresh draws. The class statistics come from the full clean test set.

**When:** at the start of training epochs, after c completed epochs, for c ∈ {0, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30, …, every 5 until training stops}.
- c = 0 is the initial weights.
- Epoch boundaries are detected from the trainer's own switch from evaluation back to training mode, so labels may be offset by at most 1 epoch. The offset is identical across models. (The actual offset was 2: see A1.)

**No influence on training:**
- The test set is only observed.
- Early stopping and model selection use the validation set, exactly as before.
- Measurements run under `no_grad` in eval mode and draw only from private RNGs.
- The bit-exact reproduction gate (§6) proves training is unchanged.

## 4. Design

| Item | Setting |
|---|---|
| Models | M3, M4, M5 |
| Arms | clean-trained (σ_train = 0), noise-trained (σ_train = 0.3) |
| Cell | K = 5, subset A {1,3,5,8,9} only (fastest cell; walltime safety) |
| Seeds | 42–51, paired |
| Jobs | 3 × 2 × 10 = **60**; smoke = seed 42 = 6 jobs |

## 5. Statistics

**Summary per run: early-training radius** = mean bubble radius at z over checkpoints c ∈ {1, 2, 3, 5, 7, 10, 15, 20, 25, 30}, at test σ = 0.3. Every run trains past epoch 30: M3 and M5 have a 30-epoch warmup, and M4's earliest v15.6 stop epoch at K5 A is later than 30. A run that stops earlier is reported, and its missing checkpoints are filled with its last value.

**Primary** (paired over 10 seeds, Holm over 4 tests):

| ID | Contrast | Arm |
|---|---|---|
| P-Fc | F = M3 − M4, early radius | clean-trained |
| P-Tc | T = M3 − M5, early radius | clean-trained |
| P-Fn | F = M3 − M4, early radius | noise-trained |
| P-Tn | T = M3 − M5, early radius | noise-trained |

**Secondary** (uncorrected, with MDE):
- **Epochs-to-tight:** the first checkpoint at which the radius at z falls to within 10% of that run's own final radius, i.e. ≤ 1.1 × the radius at the last checkpoint. "How fast does it settle?"
- **Epochs-to-threshold:** the first checkpoint with radius ≤ 0.35 at z, the K5 clean-trained mean final radius of M4 in v15.6 (0.343, rounded up). "How fast does it get as tight as M4 ever gets?" A run that never reaches it is assigned its stop epoch.
- Early-training η, ρ_W and bleed at z; early radius at L0 and L1.
- The full curves.

Every null is reported with its MDE.

## 6. Integrity gate

1. **Patch verify:**
   - The v15.7 trainer minus the two v15.7 blocks is byte-identical to the v15.6 trainer.
   - In a simulated training loop on the real models, parameters after training and global RNG states are bit-identical with tracking on and off.
2. **Smoke and grid reproduction:** every job equals the stored v15.5 run bit-exactly (test_acc, best/stop epoch, every test and cross-eval prediction).
3. **Final geometry** (the v15.6 pass at the end of each job) equals the stored v15.6 geometry file for the same job, exactly.
4. **Tracking:** at least 8 checkpoints; the first is at c = 0; epochs strictly increasing; the epoch count is within ±2 of stop_epoch; no NaN; σ = 0 sanity (shift and spread = 0).
5. No accuracy or metric is read before the gate prints `ALL CHECKS PASSED`.

## 7. Predictions (written before any v15.7 result)

- **P1.** F early radius < 0 in both arms: M3's bubble is tighter during early training. Basis: v15.6 shows M3's final bubble is tighter at z (clean-trained K5, 10/10 seeds), and the mechanism (flattening of the cosine by black/white contrast learning) acts from the first epochs.
- **P2.** T early radius < 0 in both arms. A frozen circuit cannot flatten.
- **P3.** Epochs-to-threshold is smaller for M3 than for M4 (clean-trained). No prediction for epochs-to-tight, because each run settles relative to its own final value.

## 8. Stop rules

- A reproduction mismatch or a final-geometry mismatch → stop.
- Any continuation (time budget hit) → that job fails the gate; report and amend before the grid.
- Inodes: about 5 files per job.

## Amendments and defects

- **D1 (2026-11-06, setup verify on Gadi, before any job; verify-test defect).** V4b ("parameters after training identical, tracker on vs off") failed for M3, M4 and M5 on the real trainer, while every other check passed, including V4c (global RNG identical).
  - **Cause:** the test built the tracker-on and tracker-off models one after the other. The real `build_model` does not reseed (the stand-in did), so the second model started from different random weights, and the comparison was not like-for-like.
  - **Reproduced locally:** with a non-reseeding stand-in, the same three checks fail.
  - **Fix:** the tracker-off model is loaded with the tracker-on model's initial weights (`b.load_state_dict(a.state_dict())`) before training. With that fix all checks pass locally.
  - **Unchanged:** the tracker and the trainer. Only the verify script changed. The fixed verify passed on Gadi before the smoke.
- **A1 (2026-11-06, found AFTER the grid results were read; epoch-label offset).** In all 60 runs, the records labelled c = 0, 1 and 2 are identical: no training happens before the third detected epoch start. The real trainer switches between evaluation and training twice before its first optimiser step (likely an initial validation pass and the gradient-flow probe).
  - **Consequence:** the true number of completed training epochs is label − 2 (label 2 = initial weights). The offset is identical in every run and model, consistent with the gate's epoch_starts ≈ stop_epoch + 2.
  - **Effect on the primary:** the pre-registered early window (labels 1–30) contains two untrained checkpoints, where the circuit models' radius is 2.03 against M4's 1.07. These dominate the pre-registered statistic.
  - **Reported both ways:**
    - As pre-registered (none detected; F positive).
    - Corrected: labels 3–30 = true epochs 1–28. All four contrasts favour M3 on 8–9/10 seeds; none detected after Holm.
  - The corrected analysis is post hoc and labelled as such. Neither verdict is upgraded.
  - **Unchanged:** the data. Only the epoch labelling and the window.
