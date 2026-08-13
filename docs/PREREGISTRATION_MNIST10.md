# Pre-registration — MNIST 10-class, arm (a)

**Written 2026-08-13, before any v10 job was submitted. Predictions frozen.**

Anything added after the first `qsub` goes in the Outcome section at the
bottom and is labelled post-hoc.

---

## Why this experiment

Three completed binary Fashion-MNIST grids (0,6 / 2,6 / 2,4) returned
M2 − M1 effects of +0.62, +1.04 and −0.30 pp against MDEs of 1.4–2.8 pp.
Twelve architectural contrasts across four tasks; **every one fell inside the
band the design can resolve.** The nulls bound the effects; they do not
establish equivalence.

The 10-class MNIST diagnostic behaved differently: M2 72.9% against M1 66.5%,
a **+6.4 pp** gap, roughly three times any MDE in the project. M2 had not
converged at the epoch cap, so that is a lower bound.

The structural reason: with two classes a single scalar separates them
perfectly, so a better front-end has nowhere to express its advantage — the
head discards the extra information. With ten classes, representation quality
binds. This grid tests the central hypothesis on a task where the effect has
room to appear.

**The hypothesis is unchanged.** Does replacing Long et al.'s frozen,
randomly-initialised quanvolutional front-end with a trainable Circuit 4
ansatz, trained end-to-end, improve image classification?

---

## Design

Four iso-parametric models, re-padded for 10 classes:

| | Front-end | Back-end | Params |
|---|---|---|---|
| M1 | frozen Henderson quanvolution | QNN head, 3 observables | 1,640 |
| M2 | trainable Circuit 4 | QNN head, 3 observables | 1,646 |
| M3 | trainable Circuit 4 | classical head | 1,649 |
| M4 | bilinear resize | classical head | 1,643 |

Spread 0.55%, verified by `scripts/verify_v10_multiclass.py`.
M2 − M1 = 6 params exactly (the front-end); M3 − M4 = 6 params exactly.

Attribution unchanged: M2−M1 isolates front-end trainability, M2−M3 the
quantum back-end, M3−M4 the quantum front-end.

**Protocol.** 5000 / 1250 / 2000 split (test balanced at 200 per class),
batch 32, AdamW lr 1e-3 wd 1e-4, dropout 0.0, **max_epochs 200**,
patience 20, min_epochs 30, `--qnn_backend torch`, seeds 42–46, alpha 0,
`--deterministic`, `--save_predictions`.

200 epochs rather than 100 because M2 had not converged in the diagnostic.

---

## Decisions made before the grid, and why

**Three observables, not one.** `<Z0>`, `<Z1>`, `<Z0Z1>` read from the same
final statevector. Zero extra circuit parameters, verified. The diagnostic
established the QNN back-end is measurement-limited rather than
parameter-limited: 1→3 observables doubled 10-class accuracy while +50% more
variational parameters changed nothing (p = 0.975).

**The classical head is shaped like the quantum head.** A naive
`Linear(64,10)` classical head is 650 parameters against the QNN head's 176 —
a 474-param gap on a ~1,600-param budget, which would make every contrast
uninterpretable. M3/M4 instead get `fc2(64→2) → mid(2→3) → fc3(3→10)`, so the
only structural difference between M2 and M3 is whether the 2→3 map is a
quantum circuit or a linear layer.

**`--head_variant linear`, not `tanh`.** The QNN's 2→3 map is nonlinear and
bounded in [−1,1]; a bare `Linear(2,3)` after `Linear(64,2)` collapses to a
single rank-2 linear map. `tanh` is available and would be the closer
analogue. **`linear` is chosen for continuity with the binary M3/M4 head,
which was also a pure linear map.** This is a real limitation, it is
acknowledged here rather than discovered later, and if a reviewer objects the
correct response is to run the `tanh` variant as a robustness check — not to
re-choose after seeing the results.

**BCE-on-one-hot loss retained**, not switched to cross-entropy, so v10 stays
comparable to v6–v9 on everything except the class count. Unusual for
multiclass; disclose it.

---

## Predictions

1. **M2 > M1 in sign, and the gap exceeds its MDE.** The diagnostic gave
   +6.4 pp. Predicted here: **M2 − M1 between +3 and +8 pp, p < 0.05.** This
   is the first pre-registered prediction in the project that expects a
   *significant* result rather than a null.

2. **McNemar's test on pooled per-image predictions agrees in sign with the
   accuracy-mean t-test.** If they disagree, the accuracy means are the ones
   to distrust, and the reason must be found before either is reported.

3. **All four models land well below their binary accuracies** (~65–80%
   rather than ~99%), i.e. the task is not saturated and there is headroom
   for architecture to matter.

4. **M3 − M4 (quantum front-end) is positive.** It was +1.24 pp at Fashion
   2,4, the only raw-significant attribution contrast in the project. If the
   front-end helps at all, ten classes should show it more clearly.

5. **`--deterministic` produces exact run-to-run reproduction at fixed seed.**
   Verifiable directly: re-run one config twice and diff the accuracies. If
   they still differ, something else injects nondeterminism and must be found
   before the grid is trusted.

### What would falsify the central hypothesis

M2 − M1 null *at 10 classes with adequate power* would be substantive
evidence against front-end trainability, in a way the binary nulls are not.
That is the point of running it: this design can actually be wrong.

---

## Analysis plan, fixed in advance

- Primary: paired t-test on M2 − M1 test accuracy across seeds 42–46, with
  MDE reported alongside whatever the result is.
- Secondary: McNemar on pooled per-image predictions, reported with the
  per-seed table, treating the pooled p as an upper bound on the evidence
  because the five runs share a test set.
- Holm correction within each declared family: three attribution contrasts;
  five gap@best contrasts.
- Per-class accuracy and the confusion matrix reported for M1 and M2, to
  check whether any advantage is concentrated in particular digits.

---

## Outcome

**Filled in 2026-08-13, after the grid completed (20/20 clean, integrity gate
passed on all runs).** Loss function note first: this document specified BCE
retained for continuity with v6–v9, but a smoke test triggered the
convergence-at-cap check described in `AMENDMENT_2026-08-13_loss.md`, and the
grid that actually ran used softmax cross-entropy (`--loss ce`), decided
*before* any full run was submitted. That is a deviation from what is written
above and is disclosed here rather than silently reconciled.

| # | Prediction | Result | Verdict |
|---|---|---|---|
| 1 | M2 − M1 between +3 and +8 pp, p < 0.05 | +0.33 pp, p = 0.94, MDE = 16.08 pp | **Falsified** — nowhere near either the predicted magnitude or significance |
| 2 | McNemar agrees in sign with the accuracy-mean t-test | Per-image predictions were saved (`--save_predictions`) but McNemar has not yet been run on this grid's output | **Not yet evaluated** — open item, not a result either way |
| 3 | All four models land at ~65–80%, well below binary saturation | M1 70.59%, M2 70.92%, M3 81.96%, M4 82.41% | **Mostly held** — M1/M2 inside the predicted band; M3/M4 landed just above it (~82%), closer to binary-task territory than expected |
| 4 | M3 − M4 (quantum front-end) positive | −0.45 pp, MDE = 1.38 pp | **Falsified on sign** — direction reversed from the +1.24 pp seen at Fashion 2,4, though both are null and the true effect could still be positive, negative, or zero |
| 5 | `--deterministic` reproduces exact accuracy at fixed seed | Not separately verified in this grid's job logs (no raw result JSONs available for this write-up — see note below) | **Not yet evaluated** — open item |

**What actually turned out to matter, not predicted in advance:** per-seed
M2−M1 differences swing from −11.60 pp to +13.05 pp with no consistent sign
(seed 42: +13.05, seed 43: −11.60, seed 44: +1.40, seed 45: −6.40, seed 46:
+5.20). QNN-head models (M1/M2) show 12–15 pp seed-to-seed spread at 10
classes, vs 5–6 pp for classical-head models (M3/M4). This instability, not
the M2−M1 sign or magnitude, is the primary finding of this grid — it was not
anticipated in the predictions above and is reported as post-hoc.

The single-seed diagnostic result that motivated running this grid at all
(+6.4 pp, seed 42 only) sits well inside the ±13 pp range observed once all
five seeds were run. It should be read as one noisy draw, not as the effect
size — prediction (1) was calibrated against it and that calibration is now
known to have been on unstable ground.

**Data note for this repository entry:** this write-up and the summary table
above are reconstructed from the fallback constants embedded in
`scripts/plot_v10_results.py` and from the project's running log
(`MNIST_LOG.md` §4b), not from raw per-run JSONs — those were not included in
the upload that added this code/doc bundle to the repository. If the raw
`result_M*.json` files become available, re-verify this table against them
directly and run the McNemar and determinism checks (predictions 2 and 5)
before relying on this outcome section for the thesis.
