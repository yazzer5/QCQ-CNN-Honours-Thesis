# v15.7 results: the noise bubble during training

**Run:** 2026-11-06, 60 jobs: M3, M4 and M5, trained at σ = 0 or 0.3, K5 A, seeds 42–51.
**Pre-registration:** `PREREGISTRATION_v15_7.md` (D1, A1).

**Integrity:** smoke gate 6/6 and grid gate 60/60 `ALL CHECKS PASSED`.
- Every job reproduced its v15.5 run bit-exactly, so the tracker did not change training.
- Every job's final geometry equals v15.6 exactly.
- 14–45 checkpoints per job; tracking cost 8–35 s per job.

## 1. Answer in one paragraph

After one epoch of training, M4 (no front-end) has the tightest bubble, because averaging gives it a head start. By about **epoch 3, M3's bubble is tighter**, and from then on the two models go in opposite directions.
- **M3 keeps its bubble tight** (radius ≈ 0.31 clean-trained, falling to 0.21 noise-trained).
- **M4's bubble steadily loosens as clean training continues** (0.33 at epoch 1 → 0.44 by epoch 40).
- **M5's bubble never gets tight** (≈ 0.48).

So the answer to "who gets the tight bubble faster" is: M3 overtakes within a few epochs (median crossover epoch 3) and is the only model that holds it. The defining difference is that **M4 loses robustness as it fits the clean data, and M3 doesn't**. The pre-registered early-window primary was not detected. Its window contained two untrained checkpoints, because of an epoch-label offset found after the run (A1). Corrected, all four contrasts point M3's way on 8–9 of 10 seeds, but none survives Holm correction.

## 2. Bubble radius at z (mean of 10 seeds, test σ = 0.3; epochs corrected per A1)

| Epochs trained | 0 | 1 | 3 | 5 | 8 | 13 | 18 | 28 | 38 | 58 |
|---|---|---|---|---|---|---|---|---|---|---|
| **Clean-trained** M3 | 2.03 | 0.43 | **0.33** | **0.31** | **0.31** | **0.31** | **0.31** | **0.32** | **0.32** | **0.32** |
| M4 | 1.07 | **0.33** | 0.35 | 0.37 | 0.40 | 0.41 | 0.42 | 0.43 | 0.44 | 0.44 |
| M5 | 2.03 | 0.49 | 0.47 | 0.50 | 0.50 | 0.47 | 0.47 | 0.48 | 0.48 | 0.49 |
| **Noise-trained** M3 | 2.03 | 0.48 | **0.26** | **0.25** | **0.24** | **0.22** | **0.22** | **0.21** | **0.21** | **0.21** |
| M4 | 1.07 | **0.30** | 0.29 | 0.30 | 0.31 | 0.32 | 0.31 | 0.31 | 0.30 | 0.30 |
| M5 | 2.03 | 0.55 | 0.34 | 0.29 | 0.30 | 0.29 | 0.29 | 0.29 | 0.29 | 0.28 |

**Crossover** (first epoch at which M3's radius < M4's, per seed):
- Clean-trained: 1, 5, 3, 3, 23, 1, 3, 1, 3, 13 (median 3).
- Noise-trained: 3, 3, 3, 8, 13, 1, 3, 1, 1, 13 (median 3).

Test error at σ = 0.3 (clean-trained) separates from about epoch 5 onward. It settles near 19% for M3 and 33–35% for M4 and M5.

## 3. Primary (early radius at z, Holm over 4)

| Contrast | As pre-registered (labels 1–30) | Corrected (true epochs 1–28) |
|---|---|---|
| F clean-trained | +0.141, p = .055, p_holm = .17 (M3 smaller 3/10) | −0.063, p = .071, p_holm = .28 (9/10) |
| T clean-trained | −0.121, p = .10, p_holm = .17 (9/10) | −0.151, p = .10, p_holm = .28 (9/10) |
| F noise-trained | +0.155, p = .029, p_holm = .12 (2/10) | −0.045, p = .11, p_holm = .28 (8/10) |
| T noise-trained | −0.055, p = .079, p_holm = .17 (9/10) | −0.069, p = .079, p_holm = .28 (9/10) |

- **Pre-registered verdict: none detected.** The positive F is an artefact of the two untrained checkpoints in the window: at initialisation M4's radius is 1.07 against 2.03 for the circuit models (A1).
- **Corrected (post hoc):** all four contrasts favour M3 on 8–9 of 10 seeds, but none is detected. **P1 and P2 are not supported** at n = 10.

## 4. Secondary (uncorrected; as computed by the pre-registered script)

- **Last-checkpoint radius:**
  - F clean-trained −0.121 (p = .0003, 10/10); F noise-trained −0.074 (p = 5e-6, 10/10).
  - T clean-trained −0.167 (p = .074); T noise-trained −0.054 (p = .032).
- **Early bleed at z:** F −0.068 (p = .011, 9/10) clean-trained; −0.045 (p = .035) noise-trained.
- **Early test error at σ = 0.3:** F −7.9 pp (p = .017) clean-trained; −2.0 pp (p = .022) noise-trained. T −8.5 pp (p = .007) clean-trained.
- **Epochs to radius ≤ 0.35:** M3 13.0 vs M4 23.1 vs M5 18.5 (clean-trained); n.s., with very large seed spread (runs that never reach it are assigned their stop epoch). **P3 not supported.**
- **Epochs to settle** (within 10% of own final radius): M3 7.4 vs M4 3.0 (clean-trained), n.s. M4 "settles" sooner because its radius barely improves and then drifts upward. This metric rewards standing still.

## 5. Findings

- **V1. M4 starts tightest but loosens; M3 tightens and holds.** With clean training, M4's bubble grows by a third between epoch 1 and epoch 40, while its clean accuracy improves. This is classic fit-versus-robustness drift: the backbone learns detail that the background brightening disturbs. M3's bubble does not drift.
- **V2. The overtake happens within about 3 epochs** (median crossover, both arms). That fits the mechanism: flattening the cosine at black and white is driven by black-white contrast, which is the first thing learned.
- **V3. A frozen circuit never gets a tight bubble when clean-trained** (≈ 0.48 throughout). It does tighten with noise training (0.28), because the backbone then learns the shift.
- **V4. Noise training makes everyone tighter, and M3 tightest** (0.21 vs 0.30 for M4 at the end).
- **V5. The ranking that matters for accuracy is stable from about epoch 5 onward.** Error at σ = 0.3 separates in the same order as the radius.

## 6. What it means for the thesis (supervisor's criterion)

"A good model forms a consistent bubble around the true value faster than the others":
- **M3 meets it.** It overtakes M4 within a few epochs and is the only model that keeps a tight bubble for the rest of training.
- **Do not claim "M3 is tighter from the start".** After one epoch M4 is tighter, and the pre-registered early-window test did not detect an M3 advantage.
- **The stronger, defensible claim:** *the trained front-end prevents the loss of robustness that a classical network shows as it trains on clean data.*

## 7. Caveats

- K5 A only; 10 seeds.
- Epoch labels were offset by 2 (A1). The correction is a constant shift applied identically to all models, and it was identified after the results were read, so it is reported as post hoc.
- R = 10 noise draws per checkpoint, against 20 in v15.6. The last checkpoint is the final-epoch model, not the best-validation model, so final radii differ slightly from v15.6 (e.g. M3 0.32 here vs 0.28 there).

## 8. Figures (`v157_out/figures/`)

| Figure | Shows |
|---|---|
| `E1c_radius_error_vs_true_epoch.png` | **Lead figure.** Radius and σ = 0.3 error vs true epoch, both arms |
| `E1_radius_vs_epoch.png` | As pre-registered (uncorrected labels) |
| `E2_components_vs_epoch.png` | Shift, size, bleed and error vs epoch (uncorrected labels) |
| `E3_per_seed.png` | Every seed's curve |
| `E4_primary_standard.png`, `E5_epochs_to_tight.png` | Standard set |
