# v12 Results: Is the Hybrid Worth It? (class-count sweep)

**Phase:** Thesis C, follow-up to the core M1–M4 results
**Experiment:** v12, MNIST class-count sweep (K = 3, 4, 5, plus K = 10 from v10)
**Status:** COMPLETE. K=3, 4, 5 all 80/80 runs, every integrity gate passed. Amendment A1 confirmed before K=4/K=5 were read.
K=10 completed to 10 seeds in v13 (2026-10-12); §6, F1, F2 and §9 updated. v13 readout-width results: `V13_RESULTS.md`.
**Last updated:** 2026-10-12

*(Originally titled "Post-Core Results"; other documents may cite it as `POST_CORE_RESULTS.md`.)*

---

## 1. Why this experiment exists

At the Thesis C progress meeting the core results were presented: training the
quantum front-end (M2) is no worse than freezing it (M1), but the classical CNN
(M4) wins, and the ablation without a quantum back-end (M3) consistently beats M2.
The supervisor's question was:

> If the classical CNN always wins, is there any point building these hybrid
> architectures? Under what conditions, if any, does a quantum front-end help,
> and does that change as the number of classes grows?

That splits into three experimental questions:

| # | Question | Answered by |
|---|---|---|
| Q1 | How does performance change with the number of classes? | Accuracy vs K, all models |
| Q2 | Is the quantum back-end or the front-end driving the results? | M2−M1 (front-end training), M3−M2 (back-end cost) |
| Q3 | Does a quantum front-end beat a classical one under any condition? | M3−M4 (same classical head, quantum vs classical front-end) |

**Reading of M3, corrected for the record.** M3 beating M2 does *not* mean
"only the back-end needs training". M3 *removes* the quantum back-end and replaces
it with a classical head. So M3 > M2 means the **quantum back-end is the
bottleneck**, consistent with the earlier finding that the QNN head is
measurement-limited (1 → 3 observables doubled 10-class accuracy at zero
parameter cost).

---

## 2. Protocol (pre-registered 2026-09-28, `PREREGISTRATION_v12.md`)

- **Identical to v10** except for the class set: CE loss, `n_obs=3`, linear head,
  `--deterministic`, `--save_predictions`, batch 32, Adam lr 1e-3, wd 1e-4,
  dropout 0, max 200 epochs, patience 20, min 30 epochs for quantum front-ends.
- **Split fixed per class:** 500 train / 125 val / 200 test. This makes v10
  (5000/1250/2000 over 10 classes) the K=10 point, with no re-run.
- **Two digit subsets per K**, drawn once with `numpy.random.default_rng(20260928)`
  so that class count isn't confounded with which digits were picked:

  | K | Subset A | Subset B |
  |---|---|---|
  | 3 | 1, 4, 9 | 2, 4, 9 |
  | 4 | 0, 2, 3, 8 | 0, 1, 2, 9 |
  | 5 | 1, 3, 5, 8, 9 | 0, 2, 6, 8, 9 |
  | 10 | 0–9 (v10, seeds 42–46) | — |

- **Seeds 42–51** (10, up from 5; at K=10, 5 seeds gave a 16 pp MDE).
- **Run pattern per K:** full-scale smoke (M2, seed 42, both subsets), then the
  integrity gate, then the 78-job grid, then the gate again before reading any
  accuracy.
- **Contrasts:** paired over seeds, Holm-corrected within each K. Primary result
  is pooled over subsets (per-seed mean of A and B, df = 9). Every null is
  reported with its MDE (80% power, α = 0.05).

### Iso-parametric counts at K=3 (all `VERIFIED ... OK`)

| Model | Front | Backbone | Back-end | Total |
|---|---|---|---|---|
| M1 | 0 | 1464 | 148 | 1612 |
| M2 | 6 | 1464 | 148 | 1618 |
| M3 | 6 | 1464 | 151 | 1621 |
| M4 | 0 | 1464 | 151 | 1615 |

Spread 9 params (0.56%), inside the 1% budget. M2−M1 = M3−M4 = 6 exactly, so
the front-end/back-end attribution logic holds at K=3, as it did at K=2 and K=10.

### Iso-parametric counts at K=4 (all `VERIFIED ... OK`)

| Model | Front | Backbone | Back-end | Total |
|---|---|---|---|---|
| M1 | 0 | 1464 | 152 | 1616 |
| M2 | 6 | 1464 | 152 | 1622 |
| M3 | 6 | 1464 | 155 | 1625 |
| M4 | 0 | 1464 | 155 | 1619 |

Spread 9 params (0.56%). M2−M1 = M3−M4 = 6 exactly. Each extra class adds 4
back-end parameters to every model (148 → 152, 151 → 155), so the budget stays
balanced as K grows.

### Iso-parametric counts at K=5 (all `VERIFIED ... OK`)

| Model | Front | Backbone | Back-end | Total |
|---|---|---|---|---|
| M1 | 0 | 1464 | 156 | 1620 |
| M2 | 6 | 1464 | 156 | 1626 |
| M3 | 6 | 1464 | 159 | 1629 |
| M4 | 0 | 1464 | 159 | 1623 |

Spread 9 params (0.56%) at every K. M2−M1 = M3−M4 = 6 exactly at every K.

---

## 3. Results: K = 3 (complete, 80/80 runs, gate passed)

### Accuracy (%)

| Model | Subset A | Subset B | Pooled mean | Pooled SD |
|---|---|---|---|---|
| M1 fixed Q-front + Q-head | 97.25 | 96.07 | 96.66 | 0.76 |
| M2 trained Q-front + Q-head | 96.87 | 95.85 | 96.36 | 1.59 |
| M3 trained Q-front + classical head | 98.18 | 97.38 | 97.78 | 0.25 |
| M4 classical CNN | 98.15 | 97.22 | 97.68 | 0.54 |

### Contrasts, pooled over subsets (n = 10 seeds)

| Contrast | What it tests | Diff (pp) | 95% CI | p | p (Holm) | MDE (pp) |
|---|---|---|---|---|---|---|
| M2 − M1 | Training the front-end | −0.30 | [−1.14, 0.54] | 0.44 | 0.88 | 1.17 |
| M3 − M2 | Removing the quantum back-end | +1.42 | [0.31, 2.54] | 0.018 | 0.072 | 1.56 |
| M3 − M4 | Quantum vs classical front-end | +0.10 | [−0.19, 0.39] | 0.45 | 0.88 | 0.40 |
| M4 − M2 | Classical CNN vs ours | +1.32 | [0.18, 2.47] | 0.028 | 0.084 | 1.59 |

Per subset, every contrast has the **same sign on A and B**. Subset B is about 1 pp
harder for every model, but no conclusion depends on which digits were picked.
Only on subset A do M3−M2 and M4−M2 survive Holm correction (p_holm 0.029, 0.033).

### What K=3 says, in plain terms

1. **Every model is near the ceiling (96–98%).** At 3 classes there's little room
   for architecture to matter.
2. **Training the quantum front-end adds nothing.** M2−M1 = −0.3 pp, so any real
   benefit is below ~1.2 pp. This matches K=2 and K=10.
3. **The quantum front-end equals a classical front-end, but doesn't beat it.**
   M3−M4 = +0.1 pp, CI [−0.19, +0.39]. This is the most precise result in the
   project so far: on clean MNIST any quantum front-end advantage is below 0.4 pp.
4. **The quantum back-end costs ~1.4 pp at K=3, against ~11 pp at K=10.** That is
   the predicted direction (the bottleneck worsens with more classes), but at K=3
   alone it's suggestive, not significant after correction. K=4 and K=5 decide
   whether the curve is real.
5. **The QNN-head models are the unstable ones, even at K=3.** M2's seed SD is
   1.59 pp against 0.25 for M3, more than 6×. Same pattern as K=10, smaller scale.

### Against the pre-registered predictions

| Prediction | K=3 verdict |
|---|---|
| P1: back-end cost (M3−M2) grows with K | Trending (1.4 pp at K=3, 11.0 at K=10). Needs K=4, 5 |
| P2: front-end training effect (M2−M1) small at every K | Holds |
| P3: quantum vs classical front-end (M3−M4) ≈ 0 | Holds, tightly (< 0.4 pp) |
| P4: QNN-head seed variance larger, growing with K | Holds at K=3 (M2 SD 6× M3) |

---

## 4. Results: K = 4 (complete, 80/80 runs, gate passed)

### Accuracy (%)

| Model | Subset A (0,2,3,8) | Subset B (0,1,2,9) | Pooled mean | Pooled SD |
|---|---|---|---|---|
| M1 | 89.09 | 96.25 | 92.67 | 1.61 |
| M2 | 91.40 | 96.69 | 94.04 | 2.46 |
| M3 | 95.42 | 98.10 | 96.76 | 0.67 |
| M4 | 96.11 | 98.28 | 97.19 | 0.69 |

### Contrasts, pooled (n = 10)

| Contrast | Diff (pp) | 95% CI | p | p (Holm) | MDE | Verdict |
|---|---|---|---|---|---|---|
| M2 − M1 | +1.38 | [−0.54, 3.29] | 0.14 | 0.14 | 2.66 | not significant |
| M3 − M2 | **+2.72** | [0.94, 4.50] | 0.0071 | **0.021** | 2.47 | **detected** |
| M3 − M4 | −0.43 | [−0.86, −0.01] | 0.047 | 0.094 | 0.59 | not significant |
| M4 − M2 | **+3.15** | [1.26, 5.04] | 0.0045 | **0.018** | 2.63 | **detected** |

**Subset effect.** Subset A (0,2,3,8) is much harder than B (0,1,2,9), and the
penalty lands mostly on the quantum-head models: M1 loses 7.2 pp from B to A, M2
loses 5.3 pp, M3 and M4 lose only 2.2–2.7 pp. So M3−M2 is 4.03 pp on A against
1.41 pp on B. Subset B (with the easy digit 1) is actually *easier* than either
K=3 subset. Class count is not the only driver of difficulty (see §7).

## 5. Results: K = 5 (complete, 80/80 runs, gate passed)

### Accuracy (%)

| Model | Subset A (1,3,5,8,9) | Subset B (0,2,6,8,9) | Pooled mean | Pooled SD |
|---|---|---|---|---|
| M1 | 85.90 | 86.58 | 86.24 | 4.06 |
| M2 | 88.75 | 88.16 | 88.45 | 2.54 |
| M3 | 92.61 | 93.63 | 93.12 | 0.56 |
| M4 | 92.85 | 93.61 | 93.23 | 0.56 |

### Contrasts, pooled (n = 10)

| Contrast | Diff (pp) | 95% CI | p | p (Holm) | MDE | Verdict |
|---|---|---|---|---|---|---|
| M2 − M1 | +2.22 | [−0.74, 5.17] | 0.12 | 0.25 | 4.11 | not significant |
| M3 − M2 | **+4.67** | [2.83, 6.50] | 0.00028 | **0.0011** | 2.56 | **detected** |
| M3 − M4 | −0.11 | [−0.76, 0.54] | 0.71 | 0.71 | 0.91 | not significant |
| M4 − M2 | **+4.78** | [2.75, 6.80] | 0.00048 | **0.0014** | 2.82 | **detected** |

Both K=5 subsets agree (M3−M2 = 3.86 on A, 5.47 on B; both survive Holm).

**Training budget check.** No run at any K hit the 200-epoch cap. Latest stop:
epoch 152 (K=5, M1). Median stop epoch rises with K for the QNN-head models
(M2: 51 → 60 → 81) but every run early-stopped naturally, so no comparison is
budget-bound.

## 6. K = 10 anchor (v10, 5 seeds; completed to 10 seeds in v13)

> **Update 2026-10-12 (v13 top-up, seeds 47–51, reproduction of v10 seed 42 exact):**
> 10-seed means M1 73.55 ± 5.11, M2 73.92 ± 6.17, M3 82.14 ± 1.83, M4 81.82 ± 1.65.
> M3−M2 = **+8.23 pp [3.49, 12.96], p_holm 0.014** (now significant); M4−M2 = +7.90
> [2.73, 13.06], p_holm 0.021; M2−M1 = +0.36 (MDE 8.0); M3−M4 = +0.33, TOST not
> equivalent (90% CI [−0.47, 1.13]). The 5-seed figures below are kept for the record;
> the new seeds put M2 6 pp higher than v10's (seed variance of the w3 QNN head,
> Welch p = 0.13). 3 M1 runs hit the 200-epoch cap (see `V13_RESULTS.md` §2).

| Model | Mean | SD |
|---|---|---|
| M1 | 70.59 | 5.17 |
| M2 | 70.92 | 5.40 |
| M3 | 81.96 | 2.33 |
| M4 | 82.41 | 1.98 |

M3−M2 = +11.04 pp (MDE 12.09), M2−M1 = +0.33 pp (MDE 16.08). With df = 4 no
K=10 paired contrast survives Holm, so this row anchors the curve but is not a
stand-alone result. (The separate v10 finding, classical head vs QNN head by
11.4 pp at Welch p = 2.2×10⁻⁵, pools both head types and is a different test.)

---

## 7. Findings across the sweep

### F1. The quantum back-end's cost grows with the number of classes (P1 confirmed)

| K | M3 − M2 (pp) | 95% CI | p (Holm) |
|---|---|---|---|
| 3 | +1.42 | [0.31, 2.54] | 0.072 |
| 4 | +2.72 | [0.94, 4.50] | **0.021** |
| 5 | +4.67 | [2.83, 6.50] | **0.0011** |
| 10 | +8.23 | [3.49, 12.96] | **0.014** (10 seeds, v13; was +11.04 with 5) |

Monotone increase at every step, with all four point estimates positive and every
95% CI excluding zero. Significant after correction at K=4, K=5 and (with 10 seeds) K=10. **This is
the central result: the hybrid's deficit against the classical CNN is the quantum
back-end, and it widens as the task gets harder.** M4−M2 tracks it almost exactly
(1.32, 3.15, 4.78, 7.90), because M3 ≈ M4.

> **Qualifier from v13 (2026-10-12):** this growth is a property of the Z-basis
> readout (n_obs = 3). With all 15 Paulis measured, M3−M2 is 2.25 pp at K=5 and 2.37 pp
> at K=10, roughly flat in K. See `V13_RESULTS.md` V1–V2.

### F2. The quantum front-end is *equivalent* to having no front-end (P3 confirmed, amendment A1)

**What M3−M4 actually compares.** M3 and M4 have the same backbone and the same
classical head; M4 simply has **no front-end** (front = 0 params). So M3−M4 tests
"does adding the 6-parameter trained quantum front-end help?", not "quantum vs a
matched trainable classical front-end". That matched comparison (M5) is still open (v14).

Paired TOST on M3−M4, margin ±1 pp, α = 0.05, fixed before K=4/K=5 were read:

| K | Diff (pp) | 90% CI | p (TOST) | Verdict |
|---|---|---|---|---|
| 3 *(post hoc)* | +0.10 | [−0.13, 0.33] | 2.8×10⁻⁵ | equivalent |
| 4 | −0.43 | [−0.77, −0.09] | 0.0071 | **equivalent** |
| 5 | −0.11 | [−0.64, 0.42] | 0.0065 | **equivalent** |
| 10 *(5 seeds)* | −0.45 | [−1.24, 0.34] | 0.11 | not shown (underpowered) |
| 10 *(10 seeds, v13)* | +0.33 | [−0.47, 1.13] | 0.079 | **not shown** (upper bound 1.13 pp) |

Per subset: equivalent in 5 of 6 (K=4 subset A fails, 90% CI [−1.26, −0.11]).
**On clean MNIST, adding the trained quantum front-end changes accuracy by less
than 1 pp at every class count tested, compared with having no front-end at
all** (equivalence shown at K = 3, 4, 5; narrowly not shown at K = 10). It's no advantage, and not a meaningful loss either. If anything the point estimates lean
slightly classical (K=4: −0.43 pp, raw p = 0.047, not significant after Holm),
but every estimate is inside ±1 pp.

### F3. Training the quantum front-end: no significant effect (P2 holds, with a caveat)

M2−M1: −0.30, +1.38, +2.22, +0.33 pp at K = 3, 4, 5, 10. None significant. At K=4
and K=5 the point estimate is positive but the MDE is 2.7–4.1 pp, because M1 is
very seed-unstable (SD 4.06 at K=5). Reading: no evidence that training the
front-end helps, and a modest benefit at mid K can't be ruled out. This is a
**power statement, not equivalence.** Report the MDEs alongside.

### F4. The quantum head is the unstable component (P4 confirmed)

Pooled seed SD (pp):

| K | M1 | M2 | M3 | M4 |
|---|---|---|---|---|
| 3 | 0.76 | 1.59 | 0.25 | 0.54 |
| 4 | 1.61 | 2.46 | 0.67 | 0.69 |
| 5 | 4.06 | 2.54 | 0.56 | 0.56 |
| 10 | 5.17 | 5.40 | 2.33 | 1.98 |

QNN-head models are 2–7× more variable than classical-head models at every K,
and their variance grows with K. The classical-head models stay below 0.7 pp
through K=5.

### F5 (exploratory, NOT pre-registered). Back-end cost tracks task difficulty, not just K

> **Correction (2026-10-04, `DEEP_ANALYSIS_ALL_GRIDS.md` §2):** this holds only *within*
> multi-class tasks. Adding the binary tasks (FMNIST, MRI at 13–20% error) shows the
> back-end costs little at K=2 however hard the data is (Spearman ρ = 0.32, p = 0.31 across
> all 12 tasks). Number of classes is the main driver; difficulty modulates it once K ≥ 3.
> The suggested readout-capacity mechanism is also revised there (§3): up to K=5 the
> quantum head fits the training set as well as a classical head and loses on
> generalisation and seed variance.

Using M4's error rate as a model-independent difficulty measure, per subset:

| Task | M4 error (%) | M3 − M2 (pp) |
|---|---|---|
| K=4 B | 1.72 | 1.41 |
| K=3 A | 1.85 | 1.32 |
| K=3 B | 2.78 | 1.53 |
| K=4 A | 3.89 | 4.03 |
| K=5 B | 6.39 | 5.47 |
| K=5 A | 7.15 | 3.86 |
| K=10 | 17.59 | 11.04 |

Spearman ρ = 0.86 (p = 0.014, n = 7). The easy 4-class set (0,1,2,9) behaves
like a 3-class task, and the hard one (0,2,3,8) like a 5-class task. Suggested
mechanism: the 3-observable readout has a fixed capacity to separate classes, so
its penalty scales with how hard the classes are to separate, of which class
count is one driver. **Hypothesis-generating only.** It was found after looking
at the data, n = 7 subsets, and subsets are not independent of K. It motivates v13
(readout width) but must not be stated as a confirmed result.

### Pre-registered predictions: final verdicts

| Prediction | Verdict |
|---|---|
| P1: back-end cost (M3−M2) grows with K | **Confirmed.** 1.4 → 2.7 → 4.7 → 8.2 pp; significant at K=4, 5, 10 (K=10 with 10 seeds, v13). Specific to the Z-basis readout (v13) |
| P2: front-end training (M2−M1) small at every K | **Holds as a power statement.** None significant; MDE 1.2–4.1 pp at K ≤ 5 |
| P3: quantum front-end vs none (M3−M4) ≈ 0 | **Confirmed by equivalence test.** Within ±1 pp at K = 4, 5 (confirmatory) and 3 (post hoc) |
| P4: QNN-head variance larger and growing with K | **Confirmed.** 2–7× classical-head SD; rises with K |

---

## 8. Defects and amendments

**v12 D1: seed range hardcoded (caught by the gate, 2026-10-02).**
`train_gadi.py` line 243 fixed `SEEDS = [42, 43, 44, 45, 46]` as the argparse
`choices` for `--seed`. All 40 K=3 grid jobs for seeds 47–51 exited rc=2 within
20 s, before training (0.19 SU each). No results were written, and the 40
seed-42–46 results were unaffected. Fix: `SEEDS = list(range(42, 52))`. `SEEDS`
has no other use in the file (only line 1312, `choices=SEEDS`). Failed logs kept
in `pbs/failed_seedrange/`, and only the 40 missing runs were resubmitted.
**Lesson:** a smoke run only validates the seed it ran. Check any new seed against
`--help` before a grid.

**Environment note.** The two `python` errors were the login shell without conda
activated (Gadi's system `python` is deprecated and has no numpy), not a code
fault. Jobs activate conda themselves. `check_v12.sh` and `submit_v12_grid.sh`
were switched to `python3`.

**Amendment A1: equivalence margin. CONFIRMED 2026-10-03, before K=4/K=5 accuracies were read.**
A non-significant contrast is a power statement, not evidence of equivalence. To
support the Q3 claim, a paired TOST on M3−M4 with margin **±1 pp** (α = 0.05;
equivalent iff the 90% CI lies inside ±1 pp) was added. It was implemented in
`collect_v12.py --tost_margin 1.0` and tested on synthetic data before any K=4
accuracy was opened. The first real collection run, with the margin already
fixed, failed on an unknown argument before loading any data (the old script was
still on Gadi). K=3 had already been read when A1 was proposed, so the K=3 TOST
is labelled post hoc. K=4 and K=5 are confirmatory.

---

## 9. One-paragraph summary for supervisor progress checks

> Across 3, 4, 5 and 10 MNIST classes (10 seeds, two random digit sets per size),
> adding a trained quantum front-end makes no difference: a formal equivalence
> test puts it within ±1 pp of the same network with no front-end at 3, 4 and 5
> classes (narrowly not shown at 10). Training that front-end gives no significant gain over a
> frozen one. The hybrid's deficit against the classical CNN comes almost entirely from
> the quantum back-end, and that cost grows steadily with task difficulty:
> 1.4 pp at 3 classes, 2.7 at 4, 4.7 at 5, and 8.2 at 10 (significant at 4, 5
> and 10 classes after correction). The quantum-head models are also 2–7× less
> stable across seeds. So, on clean data, the quantum front-end offers no
> advantage, and the quantum back-end is a liability that grows with the
> problem. **v13 update:** widening the readout to all Pauli bases removes that
> growth and roughly halves the deficit, leaving ~2.3 pp at both 5 and 10 classes.
> The open question is whether the front-end changes under low-data or noisy conditions (v14).
