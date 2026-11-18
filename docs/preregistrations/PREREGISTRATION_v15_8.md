# Pre-registration v15.8: the fixed classical cos(πx) front-end (M6) as the control for v15.5

**Written:** 2026-11-09, before any v15.8 code ran on Gadi and before any v15.8 result existed.
**Follows:** v15.5 (`PREREGISTRATION_v15_5.md`, `V15_5_RESULTS.md`), V16_SYNTHESIS §6 (the open item), v15.6/v15.7.
**Why:** v15.5's F = M3 − M4 > 0 under pixel noise is the thesis's only positive front-end result and has no classical control. The front-end solves in closed form to A·cos(πx + δ) per pixel (FRONTEND_MATH). This run asks whether the fixed, untrainable encoding alone reproduces the robustness.

## 1. Question

**Q-v15.8.** Under Gaussian pixel noise, is the trained Circuit-4 front-end (M3) more robust than a fixed classical per-pixel cos(πx) front-end with zero parameters (M6), and is M6 more robust than no front-end (M4)?

## 2. M6 (new model)

- M6 = M3's architecture with the Circuit-4 front-end replaced by the fixed map: each 2×2 patch (p0, p1, p2, p3) → (cos πp0, cos πp1, cos πp2, cos πp3), 28×28 → 4×14×14, channel order identical to the circuit's (block A = top row, block B = bottom row).
- This is exactly Circuit 4 with every trainable gate set to the identity (w = 0): RY(πx)|0⟩ measured in Z gives cos(πx). Verified numerically against `TrainableQuanvFrontEndTorch` with zero weights (verify §3; also a run-time self-check in every M6 job).
- 0 front-end parameters; total = M4's (1,719 at K = 5, 1,799 at K = 10). Same backbone, same classical head (w15, linear), same protocol.
- Warmup: 0 epochs, like M4 (the trainer's own rule: warmup is for trainable quantum parts; M6 has none). Stated here because it differs from M3.
- Initial classical weights: the front-end draws no global RNG, so at a given seed M6's backbone and head start bit-identical to M3's and M4's (verify §3).

## 3. Design (cut for time)

| Item | Setting |
|---|---|
| New runs | M6 |
| Reused (not re-run) | M3, M4, M5 from the stored v15.5 runs, same seeds, same noisy test sets |
| Cells | K = 5 A {1,3,5,8,9}, K = 5 B {0,2,6,8,9}, K = 10 |
| σ_train | 0 (deployed arm: train clean, cross-evaluated at every σ) and 0.3 (matched arm) |
| Eval | v15.5's σ ∈ {0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0} and salt-and-pepper p ∈ {0.05, 0.1, 0.2}, v15.5's fixed noise fields |
| Seeds | 42–46 (5, paired). Fewer than v15.5's 10 to finish in a day; every null is reported with its MDE |
| Jobs | Smoke 9 = M6 seed 42 at all 6 (cell, σ_train) + M3/M4/M5 seed 42 at K = 5 A, σ_train = 0.3 (bit-exact reproductions of v15.5). Grid 24 = M6 seeds 43–46 × 6 |

Extending to seeds 47–51 later is allowed only as a dated amendment, with the same code, and reported as such.

## 4. Contrasts and statistics

Primary (σ_test = 0.3, both arms, per K; K = 5 is the per-seed mean of subsets A and B, as in v15.5):
- **G = M3 − M6** (does the trained circuit add anything beyond the fixed encoding?)
- **H = M6 − M4** (does the fixed encoding alone beat no front-end?)

Paired over seeds 42–46: mean, 95% t-CI, paired t-test, Holm over {G, H} within each (K, arm). MDE at 80% power (two-sided α = .05) for every null. Equivalence by TOST ±1 pp only if the CI allows it.

Secondary (exploratory): F = M3 − M4 on seeds 42–46 (v15.5 subset, for reference), M6 − M5, per-subset results, all contrasts at every σ_test, clean accuracy (σ_test = 0), salt-and-pepper.

## 5. Readings (fixed now)

| Result | Reading |
|---|---|
| H > 0, G not detected | The robustness is the fixed cos encoding: classical, no training or circuit needed |
| H > 0, G > 0 | The encoding explains part; the trained map adds more |
| H not detected, G > 0 | Needs the trained mixing (still classical in closed form) |
| G < 0 | The fixed δ = 0 encoding is more robust than the trained circuit (trained \|δ\| > 0) |

## 6. Predictions (before any result)

- **P1.** H > 0 in the deployed arm at σ_test = 0.3 at K = 5 and K = 10. Basis: cos(πx) has zero slope at x = 0, 1 (FRONTEND_MATH; V15_5_THEORY).
- **P2.** G ≤ 0 in the deployed arm (M6 at least as robust as M3). Basis: M6 has δ = 0 exactly; trained M3 keeps |sin δ| ≈ 0.28 (v15.5 local check).
- **P3.** At σ_test = 0 (clean), M6 − M4 within ±1.5 pp (neutral on clean data, like M3).
- No directional prediction for the matched arm (noise training shrank every front-end difference in v15.5).

## 7. Known limitation (stated before results)

H changes two things at once versus M4: the cosine nonlinearity and keeping the four pixels separate (M4 averages the patch and replicates it). A raw-pixel control (four pixels, no cosine) would split them; it is not in this run.

## 8. Integrity gate

1. Verify (before any job): the v15.8 trainer minus the v15.8 block and the one argparse edit is byte-identical to the v15.5 trainer; M1–M5 build bit-identically (weights, outputs, gradients, RNG) to the v15.5 file; M6 parameter count, cos(πx) = circuit-at-w=0, M6 classical init = M4's, gradient check, warmup = 0; dry runs: M3 on the v15.8 and v15.5 trainers identical, M6 reproducible with cross-eval MATCH.
2. Smoke: all v15.5 job checks (rc, protocol, params, splits, noise seeds, RNG self-checks, cross-eval MATCH) plus: M3/M4/M5 reproduce the stored v15.5 runs **bit-exactly** (test_acc, best/stop epoch, every test prediction, every cross-eval prediction); every sidecar's test-set hashes equal v15.5's at every level (same noisy images); clean splits equal v15.5's M4 logs; M6 front-end self-check PASS. A mismatch is a stop rule.
3. Grid: the same per-job checks; all M6 runs present; the stored v15.5 M3/M4/M5 runs (result + MATCH sidecar) for seeds 42–46 present for every (cell, σ_train); no continuations. Cap hits reported.
4. No accuracy is read before the gate prints `ALL CHECKS PASSED`.

## Amendments and defects

(none)
