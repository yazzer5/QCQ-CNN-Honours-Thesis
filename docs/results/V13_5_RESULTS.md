# v13.5 results: M5, the untrained-front-end control

**Status:** complete. 130/130 M5 runs, every integrity gate passed, M3 reproduction
bit-exact against v12. Pre-registration: `PREREGISTRATION_v13_5.md` (amendments
A1, A2 and defect D1 were all logged before any grid result was read).
**Date:** 2026-10-14.

M5 = M3 (trained Circuit-4 front-end, classical head) with the front-end frozen at
its seed initialisation. Same initial weights as M3, same 30-epoch warmup, same
parameter count. **M3 − M5 changes exactly one thing: whether the front-end is trained.**

## 1. Headline

| K | width | M3 − M5 (pp) | 95% CI | p (Holm) | TOST ±1 | M5 − M4 (pp) |
|---|---|---|---|---|---|---|
| 3 | 3 | +0.61 | [−0.03, +1.24] | 0.12 | no | −0.51 (TOST **yes**) |
| 4 | 3 | +0.67 | [−0.22, +1.56] | 0.12 | no | −1.10 |
| 5 | 3 | +1.45 | [−0.28, +3.18] | 0.24 | no | −1.56 |
| 5 | 6 | +1.61 | [−0.40, +3.62] | 0.24 | no | |
| 5 | 15 | +1.59 | [+0.04, +3.13] | 0.18 | no | |
| 10 | 3 | +2.96 | [+0.17, +5.74] | 0.079 | no | −2.63 |
| 10 | 6 | **+2.63** | [+1.12, +4.13] | **0.014** | no | |
| 10 | 15 | **+3.08** | [+0.75, +5.42] | **0.046** | no | |

K = 3, 4, 5 are pooled over subsets A and B. Per-subset results are in `v135_contrasts.csv`.

## 2. Findings

**R1. Training the front-end helps, and the gain grows with class count.** M3 − M5 is
positive in **all 13 cells** (every K, subset and width). Its size goes ~0.6 pp (K=3)
→ 0.7 (K=4) → 1.5 (K=5) → ~3 pp (K=10). It is **detected after Holm at K=10, widths 6
and 15**. At K ≤ 5 each cell alone is a null (MDE 0.9–2.8 pp), but no cell passes
TOST, so "no difference" cannot be claimed either.

**R2. An untrained Circuit-4 front-end is slightly *worse* than no front-end.**
M5 − M4 is negative at every K (−0.5 → −2.6 pp). It is detected only at K=4 subset A
(−1.61, Holm p = 0.009) and is equivalent within ±1 pp only at K=3. Combined with
v12's M3 ≈ M4: **training repairs what a random front-end loses. It does not add
anything beyond having no front-end at all.**

**R3. The training effect is partly a rescue of bad random initialisations
(exploratory).** Seed 46's frozen front-end is the worst M5 in **all 13 cells**
(mean within-cell z = −2.3). M3 at seed 46 is unremarkable (z = −0.5). Its
per-seed M3 − M5 is 3.8–11.2 pp, against a median of about 0.5–2 pp elsewhere.
Seed 42 is also weak at K=10. The same seed gives the same 6 initial circuit
parameters at every K and width, so this is one bad front-end, reproduced everywhere.

Sensitivity without seed 46 (exploratory, not pre-registered):

| K | width | M3 − M5 | 95% CI | TOST ±1 |
|---|---|---|---|---|
| 3 | 3 | +0.34 | [+0.12, +0.57] | **yes** |
| 4 | 3 | +0.32 | [−0.15, +0.79] | **yes** |
| 5 | 3 / 6 / 15 | +0.76 / +0.78 / +0.98 | lower bounds −0.10 / −0.05 / +0.14 | no |
| 10 | 3 / 6 / 15 | +2.04 / +2.28 / +2.19 | [−0.09, +4.18] / [+0.81, +3.76] / [+0.86, +3.51] | no |

So at K = 3–4 the whole effect is the seed-46 rescue: without it, training is
equivalent to freezing within ±1 pp. **At K=10 a ~2 pp gain remains with seed 46
removed**, so there training does more than repair one bad initialisation.

**R4. Freezing the front-end makes results seed-dependent (P4 refuted).** M5's
seed SD is 4–6× M3's at K=5 (2.4–3.1 vs 0.5–0.6 pp) and 2–3× at K=10. Without seed
46, M5's K=5 SD falls to 1.0–1.3 pp. Most of the instability is the
initialisation lottery that training removes.

**R5. Width control holds (P3).** M5(w15) − M5(w3) at K=5 pooled is −0.33 pp, TOST
equivalent (p = 0.008). At K=10 it is a null with a wide MDE (4.0 pp). As for M3 in
v13, fc2 caps the classical head at rank 2.

**Secondary (descriptive).**
- M5 − M1 is +3.4 to +5.6 pp at K ≥ 4. That mixes head type and circuit type, and is
  consistent with v12's back-end cost.
- The interaction (M3−M5) − (M2−M1) is null everywhere, with MDE 1.5–7.7 pp. There is
  no evidence that the head changes the training effect, but no power to rule it out.

## 3. Against the predictions

| Prediction | Result |
|---|---|
| P1: M3 − M5 ≈ 0 (TOST) at every K, width | **Not supported.** Positive in 13/13 cells, detected at K=10 w6/w15. Holds at K=3–4 only once seed 46 is excluded (exploratory). |
| P2: M5 − M4 ≈ 0 at K ≤ 5 | **Partly.** Equivalent at K=3. Negative at K=4 (detected in subset A) and K=5. |
| P3: M5 width-insensitive | **Supported** at K=5 (TOST). Underpowered at K=10. |
| P4: M5 SD ≤ M3 SD | **Refuted.** M5 is 2–6× less stable. |

## 4. What it means for the thesis

- v12's "training the front-end gives nothing" (M2 − M1 null) does **not** carry
  over to a controlled comparison. With the circuit held fixed and a classical head,
  training helps, by roughly 0.5–3 pp depending on K.
- The benefit is best described as **robustness to initialisation**: a randomly
  initialised Circuit-4 layer can be a poor feature map (seed 46), and training fixes
  that. Beyond the rescue, a ~2 pp gain appears only at 10 classes.
- None of this is a quantum advantage. The trained quantum front-end still only
  matches no front-end (v12 F2, M3 ≈ M4). The classical-matched front-end
  comparison (now M6, v14) is still needed.

## 5. Caveats

- R3 and the without-46 table are exploratory, found after reading results. They are
  reported as such and are not substituted for the pre-registered contrasts.
- Holm is applied within each K. The 13/13 sign consistency is suggestive, but the
  cells share seeds and data, so it isn't an independent sign test.
- K=10 w3 comparators for M3 pool v10 (seeds 42–46) and the v13 top-up (47–51),
  valid because of the v13 bit-exact reproduction.

## 6. Integrity record

- Setup verify: M1–M4 bit-identical to pre-patch code at all 8 (K, width) cells.
- **A1:** warmup rule (`needs_warmup`) extended to M5 before the smoke.
- **A2:** K=10 walltime 4 h → 1 h (resource only).
- **D1:** sourced config overwrote `MODEL` and the reproduction job ran M5. Caught by
  the gate, fixed and rerun. The stray run matched the M5 smoke bit-for-bit
  (a free determinism check).
- M3 reproduction vs v12: test_acc, best/stop epoch and 1000 predictions all match.
- 0 runs at the 200-epoch cap, 0 continuations, all M5 front-ends unchanged at exit.

Figures: `figures_v135/`. Lead with `front_end_vs_k.png`, then `paired_k10.png`.

## 7. Figure map (deep analysis, `figures_v135_deep/`, script `deep_v135.py`)

| Figure | Shows | Finding |
|---|---|---|
| D1 | Accuracy of M1–M5 vs K (w3) | Overview |
| D2 | M3 − M5 vs K, by width | R1 |
| D3 | M5 − M4 vs K | R2 |
| D4 | Per-seed M3 − M5 heatmap, all 13 cells | R3 (seed 46) |
| D5 | M3 − M5 with vs without seed 46 | R3 sensitivity |
| D6 | Seed SD: M3 vs M5 vs M5 without 46 | R4 |
| D7 | Best epoch, M3 vs M5 | No speed difference: freezing doesn't change how fast it learns |
| D8 | M5 − M1 and M3 − M2 vs K | Head effect, secondary |
D4–D7 are exploratory.
