# PRE-REGISTRATION: v13 QNN readout width

**Written:** 2026-10-04, before any v13 job ran and before any v13 accuracy existed.
**Follows:** v12 (POST_CORE_RESULTS.md), findings F1 and F5.
**Scope:** v13 answers the *back-end* question only. The front-end question
(low data, noise, matched classical front-end M5) is v14 (`V14_SCOPE.md`).
One question per experiment, so a v14 result can't be confounded with a readout change.

## 1. Question

v12 F1: the quantum back-end costs accuracy, and the cost grows with task
difficulty (M3−M2 = 1.4, 2.7, 4.7, ~11 pp at K = 3, 4, 5, 10).

**Q-v13. Is that cost a readout problem? If the QNN head is allowed to report
more of its quantum state, does the gap to the classical head close, and at what
measurement cost?**

## 2. Why "more observables" means "more measurement bases" (verified offline)

The back-end is 2 qubits. A Z-basis measurement yields 4 probabilities, which
carry exactly 3 independent numbers: ⟨Z0⟩, ⟨Z1⟩, ⟨Z0Z1⟩. **v10–v12's `n_obs=3`
already uses every number the Z basis provides.** A wider readout needs other bases.

The circuit is ZZFeatureMap (uniform amplitudes, input-dependent phases φ_b(x))
followed by RealAmplitudes, a *real orthogonal* unitary U. So any observable
expectation equals (1/4) Σ_{b,b'} (U†OU)_{bb'} e^{i(φ_b' − φ_b)}: a linear
combination of cos and sin of the 6 pairwise phase differences. Because U is real:

- observables with an **even** number of Y's (Z and X bases) give **only cosine** terms;
- observables with an **odd** number of Y's give **only sine** terms.

Checked numerically on the actual circuit by `scripts/verify_v13_readout.py`:

| Width w | Observables | Measurement settings | Independent features | Do the 6 circuit params matter? |
|---|---|---|---|---|
| 3 (v12) | Z0, Z1, ZZ | 1 (ZZ) | 3 of the 6 cosines | **Yes**: training picks which 3 |
| 6 | + X0, X1, XX | 2 (ZZ, XX) | all 6 cosines | **No**: span fixed, absorbed by the linear layer |
| 15 | all 15 Paulis | 9 ({X,Y,Z}²) | 6 cosines + 6 sines = 12 | **No** |

That gives two distinct mechanisms, separated by the design:
- **w3 → w6:** is the head limited because it sees only 3 of the 6 cosine features?
- **w6 → w15:** is it limited because Z/X readout is blind to the sine (phase-sign) half?

**Interpretive constraint, fixed now.** At w ≥ 6 the trainable circuit is
redundant, and the head becomes a fixed feature map (12 closed-form trig
functions of fc2's two outputs) followed by a linear layer. Those features can
be computed classically in closed form. So if w15 closes the gap, the thesis
claim is: *"the back-end's deficit was a readout artefact; fixing it costs 9×
the measurement settings and leaves a head whose trainable quantum part is
redundant and whose features are classically computable."* It is **not**
"the quantum back-end works".

## 3. What changes from v12

Only `--n_obs` ∈ {6, 15} (plus the extra width-3 seeds at K=10). Everything else is v12's
protocol: CE, linear head, `--deterministic`, `--save_predictions`, batch 32,
Adam lr 1e-3, wd 1e-4, dropout 0, 200 epochs max, patience 20, min 30 epochs
for quantum front-ends, 500/125/200 per class, torch QNN backend, noise 0.

Code change (`patch_v13_readout.py`): new readout columns are appended after
the original 3. The n_obs = 1 and 3 code paths are untouched and are
verified **bit-identical** (outputs and gradients) against the pre-patch file.

## 4. Design

| Cell | K | Subsets | Width | Models | Seeds | Jobs |
|---|---|---|---|---|---|---|
| New | 5 | A (1,3,5,8,9), B (0,2,6,8,9), v12's | 6, 15 | M2, M3 | 42–51 | 80 |
| New | 10 | all digits | 6, 15 | M2, M3 | 42–51 | 40 |
| K=10 top-up | 10 | all digits | 3 | M1–M4 | 47–51 | 20 |
| Reused | 5 | A, B | 3 | M1–M4 | 42–51 | v12 |
| Reused | 10 | all digits | 3 | M1–M4 | 42–46 | v10 |

**Total new: 140 jobs + 2 reproduction jobs.**

- **M3 at matched width** is the iso-parametric partner: its classical 2→w map
  replaces the circuit. fc2 (64→2) caps M3's head at rank 2 at every width, so
  M3 is expected to be width-insensitive (checked by a control contrast).
- **Why the top-up is part of v13:** the w3 arm at K=10 needs 10 seeds, the
  same as the new arms. This also completes v12's K=10 point, which had only
  5 seeds and failed the TOST for lack of power.
- **Parameter counts (from `expected_params`, re-verified on Gadi):**

  | K | w | M2 | M3 | M3 − M2 |
  |---|---|---|---|---|
  | 5 | 3 | 1626 | 1629 | +3 |
  | 5 | 6 | 1641 | 1653 | +12 (0.7%) |
  | 5 | 15 | 1686 | 1725 | +39 (2.3%) |
  | 10 | 3 | 1646 | 1649 | +3 |
  | 10 | 6 | 1676 | 1688 | +12 (0.7%) |
  | 10 | 15 | 1766 | 1805 | +39 (2.2%) |

  At w15 the spread exceeds the 1% budget used through v12. This is
  **accepted and stated**: the extra M3 parameters sit in a linear 2→15 map
  that is collapsed to rank 2 by fc2, so they add no function-class capacity.
  M3(w15) − M3(w3), with TOST ±1 pp, tests this directly.

## 5. Run order and integrity gates

Per K (K=5 first, then K=10):
1. **Smoke** (`submit_v13_smoke.sh K`): M2 seed 42 at w6 and w15 on every subset,
   plus one **reproduction job**: w3, M2, seed 42 (K=5 subset A / K=10), which must
   match the existing v12/v10 result **exactly** (test_acc, best/stop epoch,
   per-image predictions).
2. **Gate** (`check_v13.sh K smoke`): rc=0, loss=ce, n_classes, class list,
   `[MULTICLASS] n_obs=W`, params VERIFIED OK, determinism, no continuation,
   reproduction exact. Writes `.smoke_ok_kK`; the grid refuses to run without it.
3. **Grid** (`submit_v13_grid.sh K`), then **gate again** (`check_v13.sh K grid`)
   before any accuracy is read.

**Pre-registered fallback (A1-v13).** If the K=10 reproduction does not match
v10 exactly, v10's seeds 42–46 are **not** pooled. TOPUP_SEEDS becomes 42–51
(40 jobs), `collect_v13.py --no_v10` is used, and the amendment is logged. If
the K=5 reproduction fails, stop: the patch or the environment changed the w3
path, and nothing runs until it's found.

## 6. Analysis

Paired over seeds. K=5 primary is pooled over subsets (per-seed mean of A and B,
df=9), with per-subset results reported alongside. A sign flip between subsets
is reported as subset-dependent. **Every null is reported with its MDE** (80%
power, α=0.05). Equivalence only via TOST, margin **±1 pp** (same as v12 A1).

**Primary family, Holm within each K (3 tests):**

| ID | Contrast | Question |
|---|---|---|
| A | M2(w6) − M2(w3) | Does seeing all 6 cosine features help? |
| B | M2(w15) − M2(w6) | Does adding the sine features help? |
| C | M3(w15) − M2(w15), + TOST ±1 | How much back-end cost remains at full readout? Is it closed? |

**Secondary (reported, not headline):** gap(w) = M3(w)−M2(w) for w = 3, 6, 15;
dGap = gap(w3) − gap(w15); M2(w15) − M2(w3); M3(w15) − M3(w3) with TOST ±1
(width control); seed SD by width.

**K=10 v12 completion (w3, 10 seeds, Holm over 4):** M2−M1, M3−M2,
M3−M4 (TOST ±1), M4−M2. Reported in POST_CORE_RESULTS.md §6 as an update to v12.

## 7. Predictions (written before results)

- **P1.** A ≥ 0 at both K. w6 contains everything w3 can express (same linear
  layer over a superset of features). Expected larger at K=10 than K=5.
- **P2.** B > 0 at K=10. Phase-sign information is available only through the
  Y bases. This is the mechanism we expect to matter most.
- **P3.** The gap shrinks monotonically with width at both K:
  gap(w3) > gap(w6) > gap(w15). **No prediction about whether C closes within
  ±1 pp.** Both outcomes are informative (§8).
- **P4.** M3(w15) ≈ M3(w3), within ±1 pp (rank-2 collapse).
- **P5 (exploratory).** M2's seed SD falls with width. At w ≥ 6 the circuit
  parameters no longer shape the features, which removes the component that
  made QNN-head models 2–7× less stable in v12 (F4).

## 8. How each outcome will be read

| Outcome | Reading |
|---|---|
| C equivalent within ±1 pp | The back-end deficit was a readout artefact. Cost of the fix: 9 measurement settings per input, and the trainable circuit becomes redundant (§2 constraint). |
| Gap shrinks but C ≠ 0 | Readout explains dGap/gap(w3) of the deficit. The rest is the 2-feature ZZ encoding itself. |
| Gap does not shrink | Readout is not the limit beyond w3. The encoding (2 inputs through a periodic feature map) is the bottleneck, and widening measurement is not a fix. |
| B ≈ 0 but A > 0 | Only the number of cosine features mattered; phase-sign is irrelevant for these tasks. |

## 9. What would NOT change this plan

A low or high headline number. No grid is re-run to improve a result. Any change
after the first smoke result is logged as a dated amendment with its reason.
