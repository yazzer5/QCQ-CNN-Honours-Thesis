# Pre-registration — Fashion-MNIST 2,4, arm (a)

**Registered:** 2026-08-12, before any 2,4 grid job was submitted.
**Experiment:** FMNIST pair 2,4 (Pullover vs Coat), arm (a) iso-parametric.
4 models × 5 seeds {42–46} = 20 runs. Code v9 (identical to v8; tag only).
**Position:** experiment 3 of 3 in the difficulty gradient. This grid closes
the sequence.

Five seeds, deliberately — not ten. Ten would drop the MDE from ~1.7 pp to
~1.1 pp and give M2−M1 a real chance to resolve, but 2,4 is the **easiest**
pair and therefore the place any effect is smallest. Spending the extra power
here would spend it in the wrong place. The three grids stay at equal n so the
gradient comparison rests on a single protocol; the power question is deferred
to a possible 10-seed re-run of **0,6**, the hardest pair, once all three
points are visible.

---

## What the two completed grids showed

| | 0,6 (v7, hardest) | 2,6 (v8) |
|---|---|---|
| M1 | — | 85.70% |
| M2 | — | 86.74% |
| M3 | — | **87.70%** (best) |
| M4 | **84.20%** (best) | 87.00% |
| grouped (M1,M2) − (M3,M4) | −1.18 pp (p = 0.011) | −1.13 pp |
| M2−M1 | positive | +1.04 pp (p = 0.075, MDE 1.61) |
| attribution contrasts | all null | all null, all below MDE |

## Predictions

1. **All three attribution contrasts (M2−M1, M2−M3, M3−M4) null again**, each
   observed |Δ| below its MDE (~1.6–1.9 pp expected at ~88%). This has held at
   both completed pairs and 2,4 is easier still.

2. **M2 > M1 in sign, magnitude ≤ 1.0 pp.** The direction has held at both
   pairs (+1.04 pp at 2,6). If it holds a third time, three independent
   same-sign results are worth reporting as a consistent trend **even though
   each is individually null** — that is a legitimately stronger statement
   than any one grid supports, and it is the honest way to use three nulls.

3. **The grouped QNN-head penalty stays near −1.1 pp rather than shrinking.**
   This is the prediction that matters. The original difficulty-gradient
   hypothesis said the penalty should narrow as the task eases; it moved only
   0.05 pp from 0,6 to 2,6. If 2,4 also lands near −1.1 pp, **the penalty is
   not difficulty-driven** and the dose-response framing must be dropped.

4. **gap@best monotone in quantum content: M2 > M1 > M3 > M4.** This held at
   2,6 (7.90 / 7.31 / 5.97 / 4.61, M2−M4 p = 0.0040). Note this is the same
   *shape* as the claim retracted at 0,6 — but that retraction was over a
   broken metric, and v8+ measures it correctly at a single checkpoint. A
   third consistent grid would make it the most robust finding in the project.

5. **M4 does not necessarily win.** M3 beat M4 at 2,6, reversing 0,6. No
   prediction is registered on the winner; report what is measured.

## What each outcome means

- Prediction 3 failing (penalty does not shrink) is **informative, not a
  failure**: it redirects effort to Aim 3 (noise) and away from further
  class-pair sweeps, and it is the correct reason to stop this sequence at
  three pairs.
- Prediction 4 holding across three pairs would support a **training-dynamics**
  claim — quantum components generalise worse at matched parameter count —
  which is a claim about *how* the models train, not about accuracy. It must
  not be presented as a performance result, since every accuracy contrast is
  null.
- Prediction 2 holding a third time is the strongest available statement about
  the thesis's core contribution: consistent direction, individually
  underpowered. Say exactly that.

## Statistics — commitments made in advance

- Every null reported **with its MDE**. A null bounds an effect; it never
  establishes equivalence.
- The gap@best family is **five tests**. Holm correction applied at 2,6 left
  M2−M4 and M1−M4 surviving and moved M3−M4 to p ≈ 0.099. The same correction
  will be applied here and reported alongside the raw p-values.
- Fixed-seed runs vary ~0.8 pp from GPU non-determinism, comparable to the
  between-seed SD (0.41–0.79 pp at 2,6). Roughly half the observed variance
  may be machine noise, which weakens the pairing. Disclosed, not corrected —
  `torch.use_deterministic_algorithms` stays off so all three grids share one
  protocol.

---

## Outcome

*(leave blank until all 20 runs are in and the integrity checks pass)*
