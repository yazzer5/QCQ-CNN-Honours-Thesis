# v15.8 results: the fixed cos(πx) front-end (M6) as the classical control for v15.5

**Written:** 2026-11-09, after the v15.8 smoke and grid gates passed. Pre-registration: `PREREGISTRATION_v15_8.md` (no amendments).
**Runs:** 33 new (9 smoke + 24 grid), all through the gate. M6 new; M3/M4/M5 are the stored v15.5 runs, seeds 42–46. The smoke rebuilt M3/M4/M5 at seed 42 (K = 5 A, σ = 0.3) on the v15.8 trainer and reproduced v15.5 bit-exactly (test_acc, best/stop epoch, 1,000 test predictions, all 10 cross-eval levels). Every run used v15.5's clean splits and noisy test images (hash-checked).

## 1. Headline

**A fixed, zero-parameter cos(πx) per-pixel encoding is at least as robust to pixel noise as the trained quantum front-end, and at K = 5 it is more robust.** v15.5's robustness is an encoding effect. Training the circuit does not add to it; at K = 5 it costs some.

## 2. Primary contrasts (σ_test = 0.3, n = 5 seeds, Holm over {G, H} within K × arm)

G = M3 − M6 (trained circuit vs fixed cos); H = M6 − M4 (fixed cos vs no front-end).

| K | Arm | Means M3 / M4 / M5 / M6 (%) | G | H | Reading |
|---|---|---|---|---|---|
| 5 | deployed (trained clean) | 82.7 / 76.4 / 64.2 / 87.0 | **−4.22 pp [−6.96, −1.48], Holm p = .026**, 0/5 seeds > 0 | **+10.58 pp [+3.33, +17.83], Holm p = .026**, 5/5 | G < 0: the fixed δ = 0 encoding is more robust than the trained circuit |
| 5 | matched (trained at 0.3) | 92.7 / 91.8 / 89.1 / 92.9 | −0.23 [−0.86, +0.40]; **TOST ±1 p = .014 (equivalent)** | +1.08 [−0.00, +2.16], Holm p = .10, MDE 1.45 | neither detected; G equivalent within ±1 pp |
| 10 | deployed | 58.6 / 58.0 / 43.9 / 61.7 | −3.17 [−14.15, +7.81], MDE 14.7 | +3.77 [−4.36, +11.90], MDE 10.9 | neither detected |
| 10 | matched | 82.1 / 77.9 / 74.8 / 80.3 | +1.77 [−2.21, +5.75], MDE 5.3 | +2.40 [−0.75, +5.55], MDE 4.2 | neither detected |

Per seed (deployed, σ_test = 0.3): M6 ≥ M3 in 10/10 K = 5 subset-seeds and 4/5 at K = 10 (seed 46 is the exception: M6 54.5 vs M3 66.3), which is what widens the K = 10 CI.

## 3. Predictions

- **P1 (H > 0 deployed):** held at K = 5 (+10.58, Holm p = .026); not shown at K = 10 (+3.77, MDE 10.9).
- **P2 (G ≤ 0 deployed):** held at K = 5 (CI entirely below 0); not decided at K = 10.
- **P3 (clean M6 − M4 within ±1.5 pp):** not shown, but only because the CI is too wide: K = 5 −0.76 [−2.52, +1.00], K = 10 −0.10 [−1.97, +1.77]. No clean-data difference was detected.

## 4. Secondary (uncorrected)

- **Deployed curves (fig5):** at K = 5, M6 is the top curve at every σ_test ≥ 0.2; G is negative and its CI excludes 0 at σ = 0.3, 0.5 and 0.7. At K = 10, M6 is top or tied at every σ but no contrast is detected.
- **F = M3 − M4 on seeds 42–46** (reference to v15.5): K = 5 deployed @0.3 +6.36 [−0.56, +13.28]; K = 10 matched +4.17 [+0.83, +7.51]. Same direction as v15.5's 10-seed result.
- **M6 − M5:** M6 beats the frozen random-phase circuit wherever detected (e.g. K = 5 deployed σ = 0.5: +23.47 [+6.77, +40.17]). M5's random phase is the worst setting, consistent with the |sin δ| mechanism.
- **Clean data (σ_test = 0):** no contrast detected; all four models within ~1–5 pp CIs of each other except M5.

## 5. Exploratory: salt-and-pepper (deployed, not pre-registered as primary)

At K = 10, H is negative (−3.18, −3.00 at p = 0.1, 0.2) and G positive (+5.70, +6.59): the cosine encoding does not help against impulse noise, which pushes pixels fully to 0 or 1 where the cosine's flatness offers nothing. At K = 5 differences are small. Exploratory, no CIs reported here.

## 6. What this means for the thesis

- The v15.5 robustness is reproduced by a fixed classical cos(πx) with 0 parameters and no training of the front-end. It is not a quantum effect and does not need the circuit.
- The trained circuit is less robust than the fixed map at K = 5 (deployed). This fits the mechanism: the encoding is flattest at δ = 0, and the trained circuit only partly moves toward it (|sin δ| 0.59 → 0.28), while M6 sits at δ = 0 exactly.
- **Limitation (pre-registered):** H mixes two changes versus M4 (the cosine, and keeping four pixels instead of their mean). G does not have this confound: M3 and M6 both keep per-pixel information, so G isolates training the circuit vs fixing δ = 0.
- **Power:** 5 seeds. K = 10 nulls carry MDEs of 11–15 pp and say little. Extension to seeds 47–51 would be a dated amendment.

Suggested wording: "A fixed classical cos(πx) per-pixel encoding with no trainable parameters reproduces the trained front-end's robustness to pixel noise. At K = 5 it beats no front-end by 10.6 pp and the trained circuit by 4.2 pp (σ = 0.3, Holm p = .026 for both, 5 seeds). The robustness is a property of the angle encoding, not of the trained quantum circuit."

## Figures (`v158_out/`, 200 dpi, script `analyse_v158.py`)
fig1_strip (per-seed), fig2_bar (mean ± 95% CI), fig3_paired (paired differences), fig4_effect_vs_mde, fig5_deployed_curves (accuracy vs σ_test, trained clean).
