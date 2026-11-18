# Pre-registration v15.6: Fisher ratio and noise geometry of M3, M4, M5

**Written:** 2026-11-04, before any v15.6 code, job or result exists.
**Follows:** v15.5 (`PREREGISTRATION_v15_5.md`, `V15_5_RESULTS.md`, `V15_5_MECHANISM.md`, `FRONTEND_MATH.md`).
**Protocol and package format:** identical to v15.5.

## 1. Question

v15.5 found M3 − M4 = +9.1 pp at σ = 0.3 (deployed, K = 5, Holm p = .001). Accuracy says *that* the trained front-end is more robust, not *where* or *how*.

**Q-v15.6.** Does the trained quantum front-end (M3) reduce pixel noise relative to the structure of the data, compared with no front-end (M4) and a frozen front-end (M5)? And at which layer does the difference appear, grow or vanish?

Operationally: is the spread and the shift of a noisy digit small compared with the spread of the clean data and the distance between digit classes, so that the class regions do not bleed into each other?

## 2. Models (verified 2026-11-03 from `qcq_cnn_v155/train_gadi.py` and all 750 v15.5 result JSONs)

| | Front-end (1×28×28 → 4×14×14) | Params K=5 / K=10 |
|---|---|---|
| M3 | Circuit 4, trained | 1725 / 1805 |
| M4 | bilinear 28→14 (= 2×2 mean), repeated to 4 channels | 1719 / 1799 |
| M5 | Circuit 4, frozen at seed init | 1725 / 1805 (1719 trainable) |

**Shared backend (identical code and shapes in all three):**
- conv1(4→8, 3×3) → ReLU → pool → conv2(8→16, 3×3) → ReLU → pool → flatten 64 (1464 params)
- → fc2(64→2) = z → fc_mid(2→15) → fc3(15→K)
- Linear head: `head_variant = linear` in 750/750 runs.

The only differences between the models are the front-end map and the warmup (30 epochs for M3/M5, 0 for M4), kept as in v15.5.

## 3. Measurement points

| ID | Dim | Features |
|---|---|---|
| L-pix | 784 | the noisy, clipped image (identical across models; baseline) |
| L0 | 784 | front-end output, flattened 4×14×14 |
| L1 | 64 | conv2 → ReLU → pool → flatten |
| L2 | 2 | z = fc2 output |
| L3 | K | logits |

Structural fact: everything after z is linear, so the logits are an affine, rank-2 image of z, and L3's Fisher ratio is identical to L2's by construction. **J is therefore not computed at L3. At L3, bleed equals the model's own error rate.**

## 4. Metrics (fixed definitions)

**Notation.**
- For test image *i* of class *c* (n_c images), let f = f_L(·) at a given layer.
- Clean features: fᵢ = f(xᵢ).
- Noisy features: fᵢᵣ = f(clip(xᵢ + nᵢᵣ, 0, 1)), for r = 1…R.

**Scatter matrices.**
- μ_c = mean over i∈c of fᵢ, and μ = class-balanced mean of the μ_c.
- **S_W** (clean within-class) = (1/K) Σ_c cov_{i∈c}(fᵢ)
- **S_B** (between-class) = (1/K) Σ_c (μ_c − μ)(μ_c − μ)ᵀ
- **S_N** (noise) = mean over i of cov_r(fᵢᵣ)

**M1. Spread ratios (lower = noise cloud smaller than the data).**
- ρ_W = tr S_N / tr S_W
- ρ_B = tr S_N / tr S_B

**M2. Noise shift (lower = better).**
- η = mean over i of ‖ f̄ᵢ − fᵢ ‖ / D, where f̄ᵢ = mean_r fᵢᵣ and D = √(2 · tr S_B), the RMS distance between class means.

**M3. Fisher ratio.**
- Clean: J₀ = tr(S_W⁻¹ S_B) at L2.
- Noisy: J_σ = the same formula with S_W and S_B recomputed on all noisy points {fᵢᵣ}, class-labelled.
- **Reported: retention J_σ / J₀** (denominator amended by A2).
- At L-pix, L0 and L1 (S_W ill-conditioned at d = 784, 64), the trace form J_tr = tr S_B / tr S_W is used instead, with the same retention ratio.

**M4. Boundary bleed.**
- The fraction of noisy points fᵢᵣ whose nearest clean class mean, under Mahalanobis distance with S̃_W, is not their own class.
- S̃_W = (1 − λ)S_W + λ(tr S_W / d)I, with **λ = 0.1 fixed at L-pix, L0 and L1**, and λ = 0 at L2.
- At L3, bleed = the error rate of argmax logits on the same noisy points.

All metrics are invariant to rescaling a layer, so M3 (outputs in [−1, 1]) and M4 (outputs in [0, 1]) compare fairly.

**Data used.**
- S_W, S_B, μ_c and J₀ come from the full clean test set (200 per class).
- Noisy metrics come from a fixed subset of **50 test images per class**, drawn once with seed 20261104, the same for every model, seed and σ.
- **R = 20** noise draws per image.

## 5. Design

| Item | Setting |
|---|---|
| Models | M3, M4, M5 |
| Training arms | **deployed**: trained at σ = 0; **matched**: trained at σ = 0.3 (fresh noise per batch, as v15.5) |
| Cells | K = 5 A {1,3,5,8,9}, K = 5 B {0,2,6,8,9}, K = 10 all |
| Seeds | 42–51, paired across models |
| Test σ | 0, 0.1, 0.2, 0.3, 0.5 |
| Data | N = 500/class train, 125 val, 200 test |
| Protocol | v15.5 exactly: CE, linear head, `--deterministic`, batch 32, AdamW lr 1e-3, wd 1e-4, dropout 0, max 200 epochs, patience 20, warmup 30 (M3/M5) / 0 (M4), `--noise_alpha 0.0` |

**Noise.**
- Test noise for the metrics: draw r of σ uses one fixed generator seeded by (20261104, σ, r), identical across models and seeds, so every contrast is paired on the same noisy images. (Draw 0 amended by A1.)
- Training noise: exactly as v15.5.

**Size.**
- 3 models × 2 arms × 3 cells × 10 seeds = **180 jobs**.
- Smoke test = seed 42 at every (model, arm, cell) = 18 jobs, kept and reused by the grid.

**Outputs.**
- ≤ 4 files per job: log, result JSON, `geom_<M>_seed<S>.npz` (per-layer traces, η per image, J, bleed, the L2 clean z and noisy z, L3 logits), and the PBS stdout.
- Checkpoint dirs are deleted at job end (v15 A1).

## 6. Statistics

- Paired over seeds.
- The K = 5 primary is pooled per seed over subsets A and B (df = 9), with each subset reported alongside. K = 10 is analysed separately.
- Every null is reported with its MDE (80% power, α = .05).

**Contrasts** (defined on any metric X):
- **F** = X(M3) − X(M4)
- **T** = X(M3) − X(M5)
- **R** = X(M5) − X(M4)

**Primary** (deployed arm, test σ = 0.3, Holm within each K, 4 tests):

| ID | Contrast | Layer | Metric |
|---|---|---|---|
| P-Fη | F | L0 | η |
| P-Tη | T | L0 | η |
| P-FJ | F | L2 | J_σ / J₀ |
| P-TJ | T | L2 | J_σ / J₀ |

**Secondary** (uncorrected, with MDE):
- every metric at every layer, σ and arm
- R
- the per-layer profile (how F changes from L0 to L1 to L2)
- per-subset results
- the M5 per-seed |sin δ_b| vs η(L0) Spearman correlation

## 7. Predictions (written before any v15.6 result)

- **P1.** Deployed, L0, σ = 0.3: η(M3) < η(M4) < η(M5), so F_η < 0 and T_η < 0.
  - Basis: FRONTEND_MATH (C1). The trained circuit is flatter at x = 0 and 1 (|sin δ| 0.59 → 0.28, V15_5_MECHANISM), so it passes less of the clipped one-sided shift. M5 is steeper.
- **P2.** Deployed, L2, σ = 0.3: retention J_σ/J₀ ranks M3 > M4 > M5, mirroring v15.5's accuracy ranking.
- **P3.** The matched arm shrinks F and T on η and J toward 0, because noise training removes the covariate shift (v15.5: matched F = +0.98 pp at K = 5).
- **P4.** At L0, the model differences are larger on η than on ρ_W. The shift, not the spread, carries the effect (FRONTEND_MATH §5.3).
- **P5.** At L-pix all metrics are identical across models (sanity), and at σ = 0, S_N = 0, η = 0 and bleed equals the clean error rate.
- **P6.** Within M5, a seed's |sin δ_b| correlates positively with its η at L0, deployed, σ = 0.3 (Spearman ρ > 0; exploratory).

## 8. Reproduction and integrity gate

1. **Patch verify:** with the geometry dump off, the patched trainer is bit-identical to the v15.5 trainer (initial weights, outputs, gradients, predictions). The verify script must end in `ALL CHECKS PASSED`.
2. **Smoke reproduction:** every deployed-arm (σ_train = 0) and matched-arm (σ_train = 0.3) smoke job must equal the stored v15.5 run of the same model, cell and seed bit-exactly (test_acc, best/stop epoch, every test prediction). The geometry pass runs after training and must not touch the global RNG.
3. **Gate checks:**
   - all v15.5 gate items
   - the metric-noise subset hash is identical across all jobs
   - the noisy-image hash per σ is identical across all jobs
   - L-pix metrics are identical across models within a (cell, seed)
   - P5 sanity holds exactly
   - L3 bleed equals the argmax error recomputed from saved logits
4. Accuracies and metrics are not read until the gate prints `ALL CHECKS PASSED`.

## 9. Stop rules

- A reproduction mismatch → stop; nothing is read until the cause is found.
- Any smoke cap hit or continuation → stop and amend.
- Inodes: check `lquota` before the smoke and before the grid. If headroom is under 1,500 inodes, free space first; do not shrink the design.
- No grid is re-run to improve a result. Any change after the smoke is logged below, dated, before it is acted on.

## Amendments and defects

- **A1 (2026-11-04, before any v15.6 code; implementation of the test-noise draws).** At each σ > 0, draw r = 0 is v15.5's own fixed noisy test image: `_v155_fixed(X_test, "gauss", σ, V155_TEST_GAUSS_BASE)`, generator seed 15500000 + round(1000σ), restricted to the 50-per-class metric subset. Draws r = 1…19 are fresh, from `torch.Generator().manual_seed(15600000 + 1000·r + round(1000σ))`, unique for r ≤ 19 and σ ≤ 0.999. R stays 20.
  - **Why:** it adds a hard gate check tying the geometry pass to v15.5. L3 argmax on draw 0 must equal v15.5's saved cross-eval predictions (`v155_xeval_<M>_seed<S>.json`, key `g<σ>`) on the subset images.
  - **Unchanged:** every metric, contrast, prediction and the design.
- **Implementation notes (2026-11-04, from the real code; not design changes).**
  - Measurement hooks: L0 = `model.front_end` output, L1 = `model.backbone` output, L2 = `model.back_end.fc2` output, L3 = `model` output. `QCQCNN` is front_end → backbone → back_end (lines 611–630).
  - The geometry pass runs inside `_v155_xeval` after the cross-eval and before `_v155_rng_restore(st)`. It runs on the model that passed `selfcheck=MATCH`, only when `V156_GEOM=1`, and draws only from private `torch.Generator`s.
- **A2 (2026-11-04, before any v15.6 job; implementation details found while writing and testing the code).**
  - **Fisher retention denominator:** retention = J(noisy subset points) / J(clean subset points), not J divided by J₀ from the full test set. Numerator and denominator then use the same images, so retention at σ = 0 is exactly 1 (gate G5). J₀ on the full test set is still stored. S_W, S_B and μ_c for ρ, η and bleed still come from the full clean test set, as pre-registered.
  - **Draw 0 batching:** the features for draw 0 are extracted from v15.5's full fixed noisy test set, batched with 32 images per batch in test order exactly as `_v155_predict` does, and then restricted to the subset. This makes A1's check bit-exact rather than subject to batch-composition rounding.
  - **σ = 0:** a single clean pass (R = 1), so S_N = 0 and η = 0 hold by construction.
  - **Enabling:** the hook runs when `V156_GEOM=1` or a `V156_GEOM_ON` marker file sits beside `train_gadi.py`. The setup script creates the marker in `qcq_cnn_v156` only.
  - **Gate scope:** hashes and L-pix equality are checked within each (cell, seed), because v15.5's test split may depend on the seed.
  - **Unchanged:** every metric definition, contrast, primary, prediction and the design.
- **D1 (2026-11-04, after the smoke gate failed, before any metric or accuracy was read; gate defect).** The smoke gate reported 360 failures, all `G5 NaN J0_full`, which is 18 jobs × 4 layers × 5 σ. `J0_full` (the full-test J₀ stored under A2) is NaN by design except at L2, but the gate's defined-value rule exempted only names starting with `J_`. My local fixtures had been generated before `J0_full` was added, so the pre-delivery test missed it.
  - **Fix:** a one-line change to `gate_v156.py`, exempting `J0_full` at layers other than L2. Tested on regenerated fixtures: the gate passes.
  - **Unchanged:** no job, result or metric. The gate is re-run on the same 18 smoke runs; nothing is re-submitted.
  - **Lesson:** regenerate test fixtures after every code change.
