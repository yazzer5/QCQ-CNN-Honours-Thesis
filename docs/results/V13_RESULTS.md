# v13 Results: QNN Readout Width

**Experiment:** v13, widening the 2-qubit QNN head's readout (pre-registered 2026-10-04, `PREREGISTRATION_v13.md`)
**Status:** COMPLETE. 142 runs (K=5: 80, K=10: 60, plus 2 reproduction runs). Every integrity gate passed. Both reproduction runs matched v12/v10 bit for bit.
**Last updated:** 2026-10-12
**Figures:** `figures_v13/` (standard set) and `figures_v13_deep/` (D1–D7, this document)

---

## 1. Question and answer in one paragraph

**Q-v13.** v12 showed that the quantum back-end costs accuracy and that the cost
grows with class count (M3−M2 = 1.4 → 2.7 → 4.7 pp at K = 3, 4, 5). Is that a
readout problem?

**Answer.** Partly. Widening the readout from the Z basis (3 observables,
1 measurement setting) to all 15 Paulis (9 settings) **removes the growth with
class count and roughly halves the deficit**. A **~2.3 pp residual** remains at
both K=5 and K=10, and the readout cannot remove it. At widths 6 and 15 the
quantum head fits the training data exactly as well as the classical head, so the
residual is a **generalisation** deficit, not a capacity limit. Wider readout
also makes the quantum head train 3× faster and removes its catastrophic seeds.

| M3 − M2 (pp) | w3: Z, 1 setting | w6: Z+X, 2 settings | w15: all Paulis, 9 settings |
|---|---|---|---|
| K=5 (pooled, 10 seeds) | 4.67 [2.83, 6.50] | 4.20 [2.26, 6.14] | **2.25 [1.22, 3.27]** |
| K=10 (10 seeds) | 8.23 [3.49, 12.96] | 4.32 [1.99, 6.64] | **2.37 [0.36, 4.38]** |

---

## 2. Integrity

- **Setup verification** (`verify_v13.log`): width-1 and width-3 outputs and gradients
  bit-identical to the pre-patch code. New readouts physically valid (Σ⟨P⟩² = 3 for
  all 15 Paulis). Structural claims hold on the real circuit: 3 / 6 / 12 independent
  features, and from w6 up the feature span doesn't depend on the circuit weights.
  All 24 model/width/K builds match `expected_params`.
- **Reproduction:** K=5 (M2, seed 42, subset A, w3) matched v12 and K=10 matched v10:
  test accuracy, best and stop epoch, and all 1000 / 2000 per-image predictions.
  v10/v12 width-3 results are poolable. Pre-registered fallback A1-v13 **not needed**.
- **Gates:** K=5 smoke 5/5, grid 76/76. K=10 smoke 3/3, grid 58/58. No continuations.
- **Training budget:** 3 runs hit the 200-epoch cap. All are **M1 at K=10, w3** (seeds 44,
  48, 49; best epoch 186–198). M1 is in no v13 contrast. It enters only the K=10
  M2−M1 completion contrast, where M1 may be slightly under-trained, which biases
  M2−M1 upwards. That contrast is null anyway (+0.36 pp, MDE 8.0).
- **Incident (not a defect):** the K=10 grid sat in PBS hold "H" for ~39 h because project
  wa66 exceeded its gdata **inode** quota (1,029,085 / 1,018,000). Cleared with
  `conda clean --all` (−33K files; `qcqcnn` env verified working). Jobs then ran
  unchanged. No result is affected.
- **Parameter table** (all `VERIFIED ... OK`):

  | K | w | M1 | M2 | M3 | M4 | M3 − M2 |
  |---|---|---|---|---|---|---|
  | 5 | 3 | 1620 | 1626 | 1629 | 1623 | +3 |
  | 5 | 6 | 1635 | 1641 | 1653 | 1647 | +12 (0.73%) |
  | 5 | 15 | 1680 | 1686 | 1725 | 1719 | +39 (2.31%) |
  | 10 | 3 | 1640 | 1646 | 1649 | 1643 | +3 |
  | 10 | 6 | 1670 | 1676 | 1688 | 1682 | +12 (0.72%) |
  | 10 | 15 | 1760 | 1766 | 1805 | 1799 | +39 (2.21%) |

  At w15 the spread exceeds the 1% budget, as pre-registered (§4 of the
  pre-registration). The control below shows it doesn't matter.

---

## 3. Accuracy (%)

| K | Subset | Model | w3 | w6 | w15 |
|---|---|---|---|---|---|
| 5 | A (1,3,5,8,9) | M2 | 88.75 ± 3.71 | 88.41 ± 3.59 | 90.44 ± 1.74 |
| 5 | A | M3 | 92.61 ± 0.62 | 92.94 ± 0.69 | 92.65 ± 0.94 |
| 5 | B (0,2,6,8,9) | M2 | 88.16 ± 2.64 | 89.42 ± 2.88 | 90.93 ± 1.99 |
| 5 | B | M3 | 93.63 ± 0.84 | 93.29 ± 0.79 | 93.21 ± 0.67 |
| 5 | pooled | M2 | 88.45 ± 2.54 | 88.92 ± 2.56 | 90.68 ± 1.61 |
| 5 | pooled | M3 | 93.12 ± 0.56 | 93.12 ± 0.47 | 92.93 ± 0.51 |
| 10 | all | M2 | 73.92 ± 6.17 | 77.90 ± 2.77 | 79.22 ± 2.18 |
| 10 | all | M3 | 82.15 ± 1.83 | 82.21 ± 2.78 | 81.59 ± 1.51 |

Mean ± seed SD, n = 10 per cell. The w3 cells are v12 (K=5) and v10 + v13 top-up (K=10).

---

## 4. Pre-registered primary contrasts (Holm within K, 3 tests)

| ID | Contrast | K=5 pooled | K=10 |
|---|---|---|---|
| A | M2(w6) − M2(w3) | +0.46 [−2.95, 3.87], p_holm 0.77, MDE 4.7 | +3.98 [−0.68, 8.63], p_holm 0.13, MDE 6.5 |
| B | M2(w15) − M2(w6) | +1.77 [−0.59, 4.13], p_holm 0.25, MDE 3.3 | +1.33 [−0.10, 2.76], p_holm 0.13, MDE 2.0 |
| C | M3(w15) − M2(w15) | **+2.25 [1.22, 3.27], p_holm 0.0024** | +2.37 [0.36, 4.38], p_holm 0.076, MDE 2.8 |
| C TOST ±1 | | not equivalent (p = 0.99) | not equivalent (p = 0.92) |

Per subset at K=5, C is detected on both A (2.21, p_holm 0.011) and B (2.28, p_holm 0.030).

**Reading.** Neither step of the readout ladder (A, B) is individually
detectable at this power. The residual back-end cost at full readout (C) is
**detected at K=5** and **not shown closed at either K**. At K=10 it doesn't
survive Holm (raw p = 0.025), so there it's "not shown to be zero, not shown to be
non-zero". Its point estimate (2.37) and K=5's (2.25) agree within 0.12 pp.

---

## 5. Secondary contrasts (pre-registered as secondary; uncorrected)

| Contrast | K=5 pooled | K=10 |
|---|---|---|
| Total readout effect, M2(w15) − M2(w3) | +2.23 [0.01, 4.45], p = 0.049 | +5.31 [0.32, 10.29], p = 0.039 |
| Gap removed, dGap = gap(w3) − gap(w15) | +2.42 [0.06, 4.78], p = 0.045 | +5.86 [−0.11, 11.82], p = 0.054 |
| Fraction of gap removed (bootstrap 95% CI) | 52% (15–73%) | 71% (20–91%) |
| Classical-head width control, M3(w15) − M3(w3) | −0.19, **TOST equivalent** (p = 0.0065) | −0.55, not shown (MDE 2.5) |

The width control confirms P4 at K=5: giving M3 a wider 2→w map changes nothing
(the rank-2 collapse through fc2). So M3's extra 39 parameters at w15 don't
explain anything, and the 2.3% budget overrun is harmless. At K=10 the point
estimate is also near zero but underpowered.

---

## 6. Findings

### V1. The readout explains the class-count growth of the back-end cost (exploratory framing, see caveat)

At w3 the gap rises with K (4.67 → 8.23, +3.56 pp). At w6 and w15 it's flat in K:
+0.11 pp (w6) and +0.12 pp (w15), Welch p = 0.93 and 0.90 (Fig. D3). The v12 curve
1.4 → 2.7 → 4.7 → 8.2 pp is therefore a property of the **Z-basis readout**, not of
the quantum head as such.

*Caveat:* this is a comparison of point estimates across K, not a pre-registered
test. The direct interaction test (gap removed at K=10 vs K=5: 5.86 vs 2.42 pp)
is **not significant** (p = 0.25), and "flat in K" is a lack of difference, not
an equivalence result. State it as "consistent with", not "shown".

### V2. A ~2.3 pp residual is left that more measurement cannot fix (primary C)

With every Pauli measured, the circuit's 6 trainable parameters are redundant
(pre-registration §2), and the head is a fixed trig feature map of fc2's two
outputs plus a linear layer. It still trails the classical head by 2.25 pp (K=5,
detected) and 2.37 pp (K=10, not significant after Holm). TOST rules out closure
within ±1 pp at both K.

### V3. The residual is a generalisation gap, not a fitting gap (exploratory; Fig. D5)

Train accuracy at the best checkpoint vs test accuracy:

| K=10 | M2 train | M3 train | M2 test | M3 test | M2 train−test | M3 train−test |
|---|---|---|---|---|---|---|
| w3 (seeds 47–51) | 90.5 | 93.4 | 76.9 | 82.3 | 13.6 | 11.1 |
| w6 | 93.5 | 93.2 | 77.9 | 82.2 | 15.6 | 11.0 |
| w15 | 93.1 | 93.0 | 79.2 | 81.6 | 13.9 | 11.4 |

At w3 the quantum head **under-fits** (2.9 pp lower train accuracy). From w6 up it
fits the training set **as well as M3**, yet still tests 2.4–4.3 pp lower: it
over-fits more. The same holds at K=5 (train accuracy 98.6–99.3% for M2 vs
99.2–99.6% for M3, while the test gap persists). This agrees with and extends
`DEEP_ANALYSIS_ALL_GRIDS.md` §3, which found the same generalisation pattern up to
K=5 at w3: the K=10 w3 under-fitting is the one place readout capacity is the
binding constraint, and widening removes it.

**Interpretation (hypothesis):** the full-readout head computes cos/sin of
pairwise phase differences that include ZZFeatureMap's product term
2(π−x₀)(π−x₁). It's a periodic, high-frequency function of fc2's outputs,
so it can fit training points the rank-2 linear head can't, but it
generalises worse. That is an inductive-bias cost of the encoding, which is
exactly what pre-registration §8 called "the remainder is the encoding itself".

### V4. Wider readout removes the class-specific collapses (exploratory; Fig. D6)

Per-digit deficit M3 − M2 at K=10:
- **w3:** concentrated on digits **4 (+17.5), 5 (+13.1), 6 (+12.0)**, with M2's largest
  extra confusions 6→0 (+13.3 pp) and 4→9 (+13.1 pp). The 3-number readout can't
  separate some class pairs at all.
- **w15:** spread evenly across digits (+1.3 to +5.0 pp), and the largest extra
  confusion is 7→8 (+5.3 pp). No class collapses.

*Caveat:* w3 here is seeds 47–51 only (v10's runs carry no predictions in this
archive), so 5 seeds.

### V5. Wider readout makes the quantum head train fast and stable (P5; Figs. D4, D7)

| | K=5 M2 | K=10 M2 | K=10 M3 (ref.) |
|---|---|---|---|
| Seed SD w3 → w6 → w15 | 2.54 → 2.56 → 1.61 | **6.17 → 2.77 → 2.18** | 1.83 / 2.78 / 1.51 |
| Worst seed w3 → w15 | 85.2 → 87.5 | **62.7 → 75.8** | |
| Median epochs to best w3 → w6 → w15 | 60.5 → 40.5 → 31.0 | **130 → 78.5 → 41.5** | 59 / 44 / 44 |

At K=10 the drop in spread is significant (Brown–Forsythe, w3 vs w15, p = 0.040;
exploratory). At w15 the quantum head reaches its best checkpoint as fast as the
classical head. This matches the mechanism in pre-registration §2: once the
circuit no longer selects which features are visible, its ill-conditioned
optimisation stops mattering. v12's F4 ("the QNN head is the unstable component")
is therefore specifically a w3 phenomenon.

### V6. Per-image confirmation (Fig.: none; numbers only)

McNemar per seed, M3 vs M2 at w15: significant in 4/10 (K=5 A), 5/10 (K=5 B) and
5/10 (K=10) seeds. Pooled over seeds, the images only M3 classified correctly
outnumber those only M2 did: 682 vs 461, 622 vs 394, 2320 vs 1846. The residual is
spread across many images, not a few outliers.

### V7. Which readout step matters is unresolved

K=10: most of the gain came from w3 → w6 (+3.98 of +5.31 pp). K=5: most from
w6 → w15 (+1.77 of +2.23 pp). Neither step is significant alone, so whether the
missing cosine features (A) or the missing sine features (B) are the limiting
factor **cannot be determined** from v13. P2 (sine features matter most) is not
supported.

---

## 7. Pre-registered predictions: verdicts

| Prediction | Verdict |
|---|---|
| P1: A ≥ 0 at both K, larger at K=10 | **Consistent, not confirmed.** +0.46 (K=5), +3.98 (K=10); right ordering, neither significant |
| P2: B > 0 at K=10 | **Not supported.** +1.33, p_holm 0.13 |
| P3: gap shrinks monotonically with width; no prediction on closure | **Holds** (pooled, both K): 4.67 > 4.20 > 2.25 and 8.23 > 4.32 > 2.37. Not monotone on K=5 subset A (3.86 → 4.53 → 2.21). **Not closed** (TOST fails at both K) |
| P4: M3(w15) ≈ M3(w3) | **Confirmed at K=5** (TOST p = 0.0065); K=10 underpowered |
| P5 (exploratory): M2 seed SD falls with width | **Holds**, strongly at K=10 (6.17 → 2.18, p = 0.040) |

Outcome per pre-registration §8: **"Gap shrinks but C ≠ 0."** Readout explains
about half (K=5, 52%) to two-thirds (K=10, 71%) of the w3 deficit, with wide
bootstrap CIs. The remainder is attributed to the 2-feature ZZ encoding.

---

## 8. K=10 completion of v12 (w3, now 10 seeds)

| Contrast | v12 (5 seeds) | v13 (10 seeds) | p_holm |
|---|---|---|---|
| M3 − M2 (back-end cost) | +11.04 | **+8.23 [3.49, 12.96]** | **0.014** |
| M4 − M2 (CNN vs ours) | +11.49 | **+7.90 [2.73, 13.06]** | **0.021** |
| M3 − M4 (front-end vs none) | −0.45 | +0.33, 90% CI [−0.47, 1.13], TOST p = 0.079 | not equivalent |
| M2 − M1 (training the front-end) | +0.33 | +0.36 [−5.35, 6.08], MDE 8.0 | 0.93 |

The new seeds (47–51) put M2 higher than v10's seeds (76.9 vs 70.9%, Welch p = 0.13).
That's seed variance in the w3 quantum head, not a protocol change: the
reproduction run matched v10 exactly. The K=10 back-end cost is therefore 8.2 pp,
now significant. F2's front-end equivalence holds at K = 3, 4, 5 and **narrowly
fails at K=10** (upper 90% bound 1.13 pp).

---

## 9. What this means for the thesis

1. **v12's headline needs one qualifier.** "The quantum back-end's cost grows with class
   count" holds **for the standard Z-basis readout**. With full readout, the cost is
   ~2.3 pp and roughly constant in K.
2. **The fix has a concrete price.** Removing about half to two-thirds of the deficit takes
   9× the measurement settings, and leaves a head whose trainable circuit is
   redundant and whose features are classically computable in closed form. That
   answers the supervisor's "what complexity would fix it?": you can buy most of
   the gap back, but what you buy is no longer a trainable quantum model.
3. **The remaining ~2 pp is an encoding/inductive-bias cost**, visible as over-fitting
   rather than under-fitting. This is a sharper mechanistic statement than v12 had.
4. **Seed instability (F4) is a readout artefact**, gone at w ≥ 6.

### Supervisor summary

> Widening the quantum head's measurement from the Z basis to all Pauli bases
> (1 → 9 measurement settings) roughly halves its deficit to a classical head and
> removes the growth with class count: the gap falls from 4.7 to 2.3 pp at 5
> classes and from 8.2 to 2.4 pp at 10. It also removes the head's seed
> instability and makes it train three times faster. A ~2.3 pp gap remains at
> both class counts. It is a generalisation gap (the quantum head fits the training
> data equally well), which points to the encoding rather than the readout. The
> fix makes the trainable circuit redundant, so the working version is no longer a
> trainable quantum model.

---

## 10. Next

- **v14** (front-end question): matched classical front-end M5, low data and noise.
- **Optional follow-up to V3** (not pre-registered): a classical head computing the same
  closed-form trig features of fc2's outputs. If it matches M2(w15), the residual is
  purely the feature map's inductive bias, with nothing specifically quantum in it.
  Cheap: one model, CPU-scale. (Run as v16a: `V16A_RESULTS.md`.)
- Add D1, D3 and D5 to the Results chapter. Lead with D3 (gap vs K by width).
