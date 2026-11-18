# Pre-registration v13.5: M5, the untrained-front-end control

**Written:** 2026-10-12, before any M5 job was submitted.
**Follows:** v12 (`POST_CORE_RESULTS.md`) and v13 (`V13_RESULTS.md`).
**Requested by:** supervisor, applying to every experiment from v12 to v15:
"M3, but the front-end is not trained, included as M5."
**Scope:** v13.5 runs M5 only, at every v12 and v13 cell, in one grid. M1–M4 are
not re-run: their existing results are the comparators.

## 1. Question

v12 found that training the quantum front-end gives no significant gain over a
frozen one (M2 − M1). That comparison has two weaknesses:

- It was tested only with the **quantum** head, which is unstable across seeds
  and may mask a front-end effect.
- M1's frozen front-end is **Henderson RandomLayers**, a different circuit from
  M2's Circuit 4. So M2 − M1 changes two things at once: the circuit type and
  whether it is trained.

**Q-v13.5. With the classical head and the same Circuit-4 front-end, does
training the front-end change accuracy?** M3 − M5 changes exactly one thing.

## 2. Model M5 (fixed now)

M5 is **M3 built by the unchanged `build_model("M3", seed)`, with every
front-end parameter set `requires_grad=False`**.

- Same seed, so the **same initial weights as M3** (verified tensor by tensor).
  The frozen front-end is the Circuit-4 layer at its seed-`s` random
  initialisation.
- Backbone and classical head are trained as in M3. At width w the head is
  M3's width-w head.
- **Same parameter count as M3** (architecturally identical). There are 6
  fewer trainable parameters: the frozen front-end.
- **Training protocol is M3's, including the min-epochs rule.** This is the
  only design that makes M3 − M5 a one-variable contrast. The patch adds M5
  wherever the code lists M3 for min-epochs, and the dry run and smoke gate
  stop if M5 resolves a different value from M3.
  *(This replaces the earlier suggestion of giving M5 M1's treatment. M1's
  front-end is a different circuit, so matching M1 buys nothing, and matching
  M3 keeps the primary contrast clean.)*
- **Runtime proof:** every M5 job prints `front-end after run: UNCHANGED OK`
  only if the front-end weights at exit are bit-identical to their initial
  values. The gate requires this in every log.

**Naming:** `V14_SCOPE.md` used "M5" for a *matched classical front-end*. That
model is renamed **M6** from now on. M5 means only this model.

## 3. Code change

`patch_v135_m5.py`, on top of the v13 repo (readout patch + v12 seed fix):
`--model` accepts M5; `build_model` and `expected_params` are wrapped so M5 →
M3 + freeze, while **M1–M4 call the original functions with the original
arguments**. `verify_v135_m5.py` proves M1–M4 are bit-identical (initial
weights, outputs, gradients, expected_params) to the pre-patch file at every
v13.5 cell. One reproduction smoke job (M3, K=5 A, w3, seed 42) must match
v12's result exactly.

Everything else is v12/v13's protocol: CE, linear head, `--deterministic`,
`--save_predictions`, batch 32, AdamW lr 1e-3, wd 1e-4, dropout 0, 200 epochs
max, patience 20, 500/125/200 per class, noise 0.

## 4. Design

| K | Subsets (v12, seed 20260928) | Widths | M5 seeds | Jobs |
|---|---|---|---|---|
| 3 | A {1,4,9}, B {2,4,9} | 3 | 42–51 | 20 |
| 4 | A {0,2,3,8}, B {0,1,2,9} | 3 | 42–51 | 20 |
| 5 | A {1,3,5,8,9}, B {0,2,6,8,9} | 3, 6, 15 | 42–51 | 60 |
| 10 | all digits | 3, 6, 15 | 42–51 | 30 |

**130 M5 jobs + 1 reproduction job.** Smoke = seed 42 at all 13 cells (kept and
reused by the grid) + the reproduction. Grid = the other 117.

Comparators (read, not re-run): K=3,4,5 w3 from v12; K=10 w3 from v10 (seeds
42–46) + v13 top-up (47–51), poolable because v13's reproduction matched; K=5,10
w6/w15 M2, M3 from v13.

## 5. Analysis

Paired over seeds. K=3, 4, 5 primary is pooled over subsets (per-seed mean of
A and B, df=9), with each subset reported alongside. A sign flip between
subsets is reported as subset-dependent. **Every null is reported with its MDE**
(80% power, α=0.05). Equivalence only via TOST, margin **±1 pp** (v12 A1).

**Primary, Holm within each K** (K=3, 4: 2 tests; K=5, 10: 4 tests):

| ID | Contrast | Widths | Question |
|---|---|---|---|
| F | M3 − M5, + TOST ±1 | every width | Does training the quantum front-end help (classical head)? |
| R | M5 − M4, + TOST ±1 | 3 | Does an untrained Circuit-4 front-end beat having no front-end? |

**Secondary (uncorrected, with MDE):**
- I: (M3 − M5) − (M2 − M1) at w3. Is the training effect different with a
  quantum head? Descriptive only: M1's front-end is a different circuit.
- H: M5 − M1 at w3. Both front-ends frozen, but head and circuit differ.
  Descriptive only.
- C: M5(w15) − M5(w3), + TOST ±1, at K=5 and 10. Width control (expected ≈ 0:
  fc2 caps the head at rank 2, as for M3 in v13).
- Seed-to-seed SD of M5 vs M3. Does freezing the front-end change stability?

## 6. Predictions (written before any M5 result)

- **P1.** F is equivalent to 0 within ±1 pp at every K and width. Basis: v12 F2
  (M2 ≈ M1) and M3 ≈ M4 within ±1 pp. If a trained front-end does no better
  than no front-end, training it should not matter either.
- **P2.** R is equivalent to 0 within ±1 pp at K=3, 4, 5. **No prediction at
  K=10**, where v12/v13 resolution is weakest.
- **P3.** C is equivalent to 0 (width-insensitive, like M3).
- **P4 (exploratory).** M5's seed SD ≤ M3's. Freezing removes one trainable
  component, but the head is classical in both, so a large difference is not
  expected.

## 7. How each outcome will be read

| Outcome | Reading |
|---|---|
| F equivalent at every K | Training the quantum front-end has no measurable effect. The trainable-front-end claim of the thesis reduces to "a random fixed Circuit-4 layer does as well". |
| F > 0 (detected) | Training the front-end helps when the head is classical. v12's null M2 − M1 was masked by the quantum head (check with I). |
| F < 0 (detected) | Training the front-end hurts (overfitting through the front-end). Report with per-seed plot and train/test fit. |
| R equivalent, F equivalent | The front-end, trained or not, is interchangeable with a bilinear resize on clean MNIST. Low data and noise (v14, v15) are where it could still matter. |

## 8. Run order and gates

1. `setup_v135.sh`: build repo from v13, patch, verify. Must end in `ALL CHECKS PASSED`.
2. Smoke (`submit_v135_smoke.sh`): 14 jobs.
3. Gate (`check_v135.sh smoke`): rc=0, loss, n_classes, class list, n_obs,
   params VERIFIED, determinism, no continuation, M5 freeze banner and exit
   self-check, M5 min-epochs = M3's, reproduction exact.
4. Grid (`submit_v135_grid.sh`), then gate again (`check_v135.sh grid`) before
   any accuracy is read.

**If the reproduction fails:** stop. The patch or the environment changed M3,
and nothing runs until it is found.

## Amendments

*(None yet. Date every amendment and keep it, as in v12 and v13.)*

**A1-v13.5 (2026-10-12, before any v13.5 job was submitted).** The setup patch
found no min-epochs rule naming M3 on a min-epochs line, so it changed nothing.
The verify dry run passed the "same min-epochs" check only because both runs
print the *global* `min_epochs_quantum=30`. Reading `train_gadi.py` showed the
real per-model rule (line 1021):
`needs_warmup = model_name in ("M2", "M3") or isinstance(model.back_end, QNNBackend)`.
Under it M5 would have received **0** warmup epochs, against M3's 30, so
M3 − M5 would have differed in two things. Fix: add "M5" to that tuple, and log
the *resolved* per-run value (`[V135] warmup: model=… min_epochs=…`). The gate
now requires `min_epochs=30` in every M5 log and in the M3 reproduction log.
M1–M4 behaviour is unchanged (membership for M1–M4 is identical). Caught by
reading the code before the smoke; no results affected.

**A2-v13.5 (2026-10-12, resource change only, before any K=10 job started).**
K=10 walltime lowered from 4 h to 1 h and the in-job time budget from 3.5 h to
0.75 h, the same as K=3/4/5. Reason: 4 h jobs queue much longer on gpuvolta,
and v13's slowest K=10 job took 18 min (M3 median 5 min). The three K=10 smoke
jobs, still queued at 4 h, were deleted and resubmitted at 1 h. Training is
deterministic and stops on early stopping, so walltime does not affect results.
Any job that hits the budget is flagged by the gate (CONT=1).

**D1-v13.5 (2026-10-12, caught by the smoke gate before any grid job).**
`run135.pbs` sources `config135.sh` after PBS sets the job's variables, and the
config defined `MODEL=M5`, which overwrote the reproduction job's `MODEL=M3`.
The reproduction therefore trained M5 (all intended M5 smoke jobs were
unaffected). Fix: the config variable is renamed `GRID_MODEL`, and the submit
scripts pass `MODEL=$GRID_MODEL`. The stray run (M5, K=5 A w3, seed 42) was kept
in `stray_v135/` and turned out to be a free determinism check: it matched the
M5 smoke run of the same cell bit-for-bit (test_acc 0.895, best epoch 29, stop
epoch 49, all 1000 per-image predictions). The M3 reproduction was resubmitted.
**Lesson:** a sourced config file must never define a name a job script takes
as input; check the header of every distinct job type, not only the gate's verdict.
