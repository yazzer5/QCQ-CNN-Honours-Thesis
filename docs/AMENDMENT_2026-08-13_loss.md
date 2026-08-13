# Amendment to PREREGISTRATION_MNIST10.md — training loss

**Date:** 2026-08-13
**Status:** made BEFORE any grid job was submitted. No M1 run existed at the
time of this decision, so nothing about the primary M2 − M1 contrast entered
into it.
**Change:** the v10 10-class grid uses softmax cross-entropy (`--loss ce`)
instead of the sigmoid + binary cross-entropy (`--loss bce`) used in v6–v9.

---

## 1. What prompted the check

`train_gadi.py` scored training with `sigmoid` + BCE against a one-hot
target. At two classes this is unremarkable and it is what all four binary
grids used. At ten classes it scores each of the ten logits independently,
so nine of the ten targets are "no" for every image, and the loss drives
every logit toward `sigmoid⁻¹(0.1) ≈ −2.2`.

M1 and M2 form logits as `fc3(Linear(3,10))` applied to QNN expectation
values bounded in `[−1,1]`. M3 and M4's classical head is unbounded. The
pressure to produce strongly negative logits is therefore an asymmetric
constraint, and it falls on exactly the two models the primary contrast sits
between.

Softmax cross-entropy normalises across classes, so only the argmax matters
and the bounded output range stops being a constraint. It is also what Long
et al. (2025) eq. 21 specifies.

## 2. The prediction that motivated the test, and how it fared

On synthetic 10-class data (CPU, one seed, **not MNIST**), BCE left M1 and
M2 at chance for 20+ epochs while M3/M4 learned, and at 60 epochs M2 came
out *below* M1 — the sign of the primary contrast reversed.

**That prediction was wrong on the real task.** Under BCE on MNIST, M2
escaped chance at epoch 8. There was no extended plateau. The synthetic
evidence overstated the magnitude of the effect, and this is recorded here
rather than dropped.

## 3. The decision rule as originally written

Fixed in `run_loss_smoke.pbs` before submission:

> Does either arm leave a model within 2 pp of chance (10%) past epoch 20?
> Neither plateaus → keep BCE. Only BCE plateaus → switch to CE. Only CE
> plateaus → stop, unexpected. Both plateau → the loss is not the problem,
> do not run the grid.

The rule was chosen so the decision would rest on convergence behaviour
alone and never on the size or sign of M2 − M1.

## 4. What the smoke test found

Two jobs, M2/seed42, identical in every respect except the loss. Confirmed
by `[PROTOCOL RESOLVED]`, which differed only in `loss=`. Both rc=0, both
passed the full integrity gate, both `total=1646`, both
`pool=6250 test=2000`, both with recomputed accuracy matching reported.

| | first epoch val > 12% | last epoch near chance | epochs run | stopped by | best epoch | val movement, last 10 ep | test acc |
|---|---|---|---|---|---|---|---|
| bce | 8 | 7 | 200 | **epoch cap** | 198 | **+1.04 pp, still climbing** | 71.55% |
| ce | 1 | 0 | 134 | early stopping | 114 | −1.44 pp, converged | 77.75% |

**Under the rule as written, the verdict is "keep BCE".** Neither arm
plateaued past epoch 20.

## 5. Why the rule was amended anyway

The runs surfaced a failure the rule did not test for. Under BCE, M2 had not
converged at the 200-epoch cap: best validation accuracy landed at epoch
198, two short of the limit, still gaining roughly 1 pp per 10 epochs. Under
CE it early-stopped at 134 with the cap nowhere near binding.

If the cap binds, the grid measures how far each model got within a fixed
epoch budget, not where each model converges. M1 (frozen front-end) and M2
(trainable) have no reason to approach their ceilings at the same rate, so a
capped M2 − M1 is partly a comparison of convergence speed. That is the same
class of confound the iso-parametric re-padding was built to remove.

This is a fact about convergence, not about the primary contrast — no M1 run
existed — so acting on it preserves the intent of the original rule even
though it departs from its letter.

**Stated plainly: the original rule looked for one failure mode (a plateau
at chance), that failure did not occur, and a different one (the epoch cap
binding) did. The rule was amended after seeing data. It was amended before
any grid job ran, and on a criterion independent of M2 − M1.**

## 6. Options considered

| Option | Assessment |
|---|---|
| Follow the rule as written: keep BCE | Clean on process, but knowingly reports a truncated result. |
| **Amend on convergence-at-cap: switch to CE** | **Adopted.** Decided pre-grid, on a criterion independent of the primary contrast. |
| Keep BCE, raise the cap | Faithful to the pre-registration, but the required cap is unknown; extrapolation suggests 400+ epochs with no principled stopping point. |

## 7. What this does and does not affect

- **Does not affect v6–v9.** The binary grids are untouched. At two classes
  BCE and CE are close enough that nothing there is in question.
- **Does affect comparability of absolute accuracies** between v10 and the
  binary grids. The pre-registration already states that v10 must not be
  pooled with v6–v9, so no additional comparability is lost.
- **Does not affect** models, circuits, parameter counts (M1 1640 / M2 1646
  / M3 1649 / M4 1643), data, splits, seeds, observables, head variant, or
  optimiser.

## 8. Predictions 1–5

Unchanged. The loss change alters the optimisation path, not any of the
architectural contrasts the predictions concern.

One consequence worth recording in advance: BCE's 71.55% for M2 sits close
to the earlier diagnostic's 72.9%, whereas CE gives 77.75%. Absolute
accuracies in the v10 grid will therefore run above the diagnostic figures,
and the diagnostic's +6.4 pp M2 − M1 gap is **not** a prediction for this
grid — it was measured under a different loss and a binding cap.

## 9. Supporting artefacts

- `run_loss_smoke.pbs` — decision rule, written before submission
- `v10loss.o176111296` (bce), `v10loss.o176111297` (ce) — full logs
- `results_loss_smoke_bce/`, `results_loss_smoke_ce/` — result JSONs
- `PATCH_train_gadi.diff` — the `--loss` switch and the `[PROTOCOL
  RESOLVED]` readback
