# PRE-REGISTRATION: v12 MNIST class-count sweep

**Written:** 2026-09-28, before any v12 job ran.
**Motivation:** Thesis C progress meeting. The supervisor's question: is the
hybrid worth building if the classical CNN always wins? Specifically:
(Q1) how does performance change with the number of classes, (Q2) is the
quantum back-end or the front-end driving the results, and (Q3) is there any
condition where the quantum front-end helps?

## What changes from v10

Only the class set. Everything else is v10's protocol, byte for byte:
CE loss, n_obs=3, linear head, `--deterministic`, `--save_predictions`,
batch 32, Adam lr 1e-3, wd 1e-4, dropout 0, max 200 epochs, patience 20,
torch QNN backend, noise_alpha 0.

**Split, fixed PER CLASS:** 500 train / 125 val / 200 test per class
(= v10's 5000/1250/2000 at K=10). So v10 is the K=10 point of this sweep and
is **not re-run**. Total data grows with K. This is stated, not hidden: per-class
data is held constant so the only thing that changes is the number of classes.

## Design

| K | Subset A | Subset B |
|---|---|---|
| 3 | 1,4,9 | 2,4,9 |
| 4 | 0,2,3,8 | 0,1,2,9 |
| 5 | 1,3,5,8,9 | 0,2,6,8,9 |
| 10 | v10 (0-9), seeds 42-46 | (none) |

Subsets drawn once with `numpy.random.default_rng(20260928)`. They will not be
redrawn after results are seen. Two subsets per K separate the effect of class
count from the effect of which digits were picked.

Models M1, M2, M3, M4 (same definitions and param-padding logic as v10).
Seeds 42-51 (10 seeds, up from 5, because 5 seeds gave a 16 pp MDE at K=10).
Jobs: 3 K x 2 subsets x 4 models x 10 seeds = **240**.

## Run order (per K, K = 3 then 4 then 5)

1. Smoke test: M2, seed 42, both subsets (2 jobs). Full scale, not reduced.
2. Integrity gate (`pbs/check_v12.sh K smoke`): rc=0, loss=ce, n_classes=K,
   correct class list, param VERIFIED OK, determinism enabled, no continuation.
3. Only if the gate passes: the grid (78 jobs, skipping the smoke run).
4. Gate again on the whole grid (`check_v12.sh K grid`) before reading any accuracy.

## Contrasts (paired over seeds, Holm-corrected within each K)

| Contrast | Question |
|---|---|
| M2 - M1 | Does training the quantum front-end help? |
| M3 - M2 | What does replacing the quantum back-end with a classical head give? |
| M3 - M4 | With a classical head, quantum front-end vs classical conv front-end |
| M4 - M2 | Headline gap: classical CNN vs our model |

Primary: pooled over subsets (per-seed mean over A and B), paired t-test, df=9.
Per-subset results are reported alongside. A contrast whose sign flips between
subsets is reported as subset-dependent, not averaged away.
**Every null is reported with its MDE (80% power, alpha 0.05).** A null is a
power statement, never evidence of equivalence.

## Predictions (written before results)

- **P1.** M3 - M2 (the cost of the quantum back-end) grows with K. The QNN head's 3
  observables are a fixed-width bottleneck, and more classes need more separable
  directions. Expected to be small at K=3 and to approach v10's ~11 pp at K=10.
- **P2.** M2 - M1 is small at every K. Training the front-end does not overcome
  a readout bottleneck downstream.
- **P3.** M3 - M4 is near zero at every K: no quantum front-end advantage on
  clean MNIST.
- **P4.** QNN-head seed variance (M1/M2) grows with K. v10 showed 12-15 pp
  spread at K=10, against 5-6 pp for the classical-head models.

If P1 holds, the answer to the supervisor is: the hybrid loses because of the
back-end readout, and it loses more as the task gets harder. That motivates the
v13 controls: a readout-width fix, and a matched classical front-end (M5).

## What would NOT change this plan

A low headline number. Grids are not re-run to improve a result. Any change
after the first smoke result is logged as a dated amendment with its reason.
