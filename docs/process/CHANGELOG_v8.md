# CHANGELOG v7 → v8

**Date:** 2026-08-10
**Purpose of v8:** run FMNIST 2,6 arm (a) — experiment 2 of the difficulty
gradient — with the generalisation-gap metric fixed at source.

**No model, architecture, protocol or numerical change.** M1–M4 are
byte-identical to v7, as are the split sizes, batch size, optimiser, learning
rate, weight decay, dropout, epoch caps, patience, warmup and seeds. Accuracy
numbers from v8 are directly comparable to the completed v7 Fashion 0,6 grid.

---

## 1. Generalisation gap now measured at the best checkpoint (`train_gadi.py`)

**The problem.** v7 wrote no gap field at all. The gap was reconstructed
afterwards as `final-epoch train_acc − test_acc`. Those two numbers come from
different models: `train_model()` restores the best-validation checkpoint
before testing, so `test_acc` is the epoch-12–17 model for M1/M2 while the
final history entry is epoch 39–61. M1/M2 keep overfitting past their peak, so
their final-epoch train accuracy is inflated relative to the model actually
evaluated — which manufactured the "gap is monotone in quantum content,
p < 10⁻⁴" claim that was later withdrawn (`FMNIST_LOG.md` §5.1).

**The fix.** Immediately after `model.load_state_dict(best_state)` and before
anything else touches the model, the training set is evaluated in eval mode.
That number and `test_acc` now come from the same weights.

Two further defects the post-hoc reconstruction could not have fixed even in
principle:

- `history["train_acc"]` is a running average across an epoch during which the
  weights were still updating. It is not the accuracy of any single model.
- `history["train_acc"]` is train-mode, `test_acc` is eval-mode. At
  `dropout=0.0` those coincide, but arm (b) runs at dropout 0.5, and the QNN
  back-end is precisely the architecture where that difference is not benign
  (defect 06).

**New fields in every `result_*.json`:**

| field | meaning |
|---|---|
| `gap_at_best` | `train_acc_at_best − test_acc`, same checkpoint, both eval mode |
| `train_acc_at_best` | training-set accuracy of the restored best checkpoint |
| `best_epoch` | epoch that produced the restored checkpoint |
| `stop_epoch` | last epoch actually run |
| `gap_metric` | string describing the definition, so a JSON is self-documenting |
| `train_acc_final_trainmode` | the old numerator, retained **only** so the v7 0,6 grid can be recomputed on the old definition for continuity — not for any new claim |

`best_epoch` is derived from `history` rather than from a loop variable,
because `save_checkpoint()` does not carry it and a resumed run would
otherwise report `None`. If the loop variable and the history disagree, a
`[WARN]` is printed and history wins. Ties go to the earliest epoch, matching
the strict `>` used to accept a new best.

**Cost:** one extra forward pass over 5,000 images per run. Negligible.

## 2. Code version stamped into output

`CODE_VERSION = "v8"` is printed at the top of every run and written to both
`result["code_version"]` and `result["protocol"]["code_version"]`. A result
file can now be attributed to a code version without inference.

## 3. Class pair echoed from inside the trainer

`[CLASS PAIR] trainer received classes=[2, 6]` is printed after argument
parsing. The v7 defect where `qsub -v CLASSES=0,6` silently became
`CLASSES=0` plus a stray variable was invisible from the submit side — this is
the only place it is detectable.

## 4. `scripts/collect_results.py`

- `gap@best`, `best ep` and `stop ep` columns in the summary table.
- Paired t-tests on `gap_at_best` for M2−M1, M2−M3, M3−M4, M1−M4, M2−M4.
- **MDE printed for every attribution contrast**, with the observed Δ flagged
  as above or below it. A null is a power statement, not evidence of
  equivalence, and the collector now makes that impossible to omit.
- Warns loudly if some runs carry `gap_at_best` and others do not — pre-v8
  results must not be pooled with v8 on that column.

## 5. PBS

- `pbs/run8_fashion_grid.pbs` — v7's script with names, output dirs and
  banner updated. Writes to `results_v8_fashion_c<ab>/`.
- `pbs/submit_run8_smoke.sh` — submits **one** job (M2, seed 42) using the
  **same** PBS script as the grid, so the smoke test exercises the identical
  code path rather than a parallel one that can drift.
- `pbs/submit_run8_fashion.sh` — **refuses to submit the grid** unless a v8
  smoke result exists for that pair (override with `--force`). It also skips
  re-submitting M2/seed 42, since the smoke test already produced it and a
  re-run would overwrite that file with a different number (fixed seeds are
  not bit-reproducible on this stack, ~0.8 pp).
- v6/v7 PBS scripts moved to `pbs/archive_v7/`.

## 6. qiskit imported lazily (v8.1, 2026-08-11)

**The problem.** Gadi job **175978087** — M2 / fashion 2,6 / seed 42, the
first v8 smoke test — died 18 seconds in, before `main()` printed a single
line. The traceback's last frame was inside qiskit's own `__init__.py`:

```
ImportError: Qiskit is installed in an invalid environment that has both
Qiskit >=1.0 and an earlier version.
```

`train_gadi.py` imported qiskit at module scope, so **any** problem in the
qiskit install killed **every** run — including runs that never touch qiskit.
The job used `--qnn_backend torch`, which has no qiskit dependency at all.
Confirmed environmental rather than a v8 regression: the identical import
fails from the v7 tree, on a line byte-identical between v7 and v8.

**The fix.** The three qiskit names are now resolved inside `_import_qiskit()`,
called from `build_qnn_backend()` only when `QNN_BACKEND == "qiskit"`. A
broken qiskit now degrades the reference path alone instead of taking down
the pipeline. When it does fail it raises a `RuntimeError` naming the pinned
stack and the diagnostic commands, rather than surfacing qiskit's packaging
error with no indication of which component needed it.

**Scope.** `--qnn_backend torch` is the default and is what every PBS script
passes, so all α = 0 production runs are unaffected. `--qnn_backend qiskit`
still requires a working install and is still needed for
`scripts/verify_qnn_torch.py` and for the noise phase (Aer's `NoiseModel`
cannot be expressed in the torch path). `scripts/verify_qnn_torch.py` keeps
its module-scope import, correctly — it exists to compare against qiskit.

**This defers the environment repair, it does not perform it.** The env has a
stale `qiskit-2.3.1.dist-info/` record (dated Apr 1) alongside the real
0.45.3 install (Apr 7); qiskit's guard scans distribution records and refuses
when it sees both. Repair before the noise phase.

---

## Verification performed before release

All tests were run against the **extracted release zip**, not a working copy.

**Static.** Every `.py` compiles; every `.sh`/`.pbs` passes `bash -n`.
`_block_unitary()` in `quanv_torch.py` contains exactly 4 `RY` and 2 `CRZ`
and zero `RZ`/`CNOT`/`CZ`, matching `MODELS_SOURCE_OF_TRUTH.md` §3. Both CRZ
diagonals were checked by hand in the |q_a q_b⟩ basis: `ctrl_is_a=False`
gives `[1, m, 1, p]` (acts on q_b = 1) and `ctrl_is_a=True` gives
`[1, 1, m, p]` (acts on q_a = 1). Correct in both directions.

**Full four-model run, n_train 2000 / n_val 500 / n_test 500, pair 2,6,
seed 42**, chosen large enough that the QNN plateau is actually cleared:

| model | params | test acc | best ep | stop ep | gap@best |
|---|---|---|---|---|---|
| M1 | 1,604 ✔ | 86.40% | 22 | 28 | +9.90 pp |
| M2 | 1,610 ✔ | 83.00% | 26 | 32 | +13.65 pp |
| M3 | 1,600 ✔ | 85.60% | 19 | 25 | +6.05 pp |
| M4 | 1,594 ✔ | 89.20% | 39 | 40 (cap) | +1.40 pp |

All four printed `[CODE VERSION] v8`, `[CLASS PAIR] ... [2, 6]`, and a
parameter assertion that passed. Zero inversions. **best epoch ≠ stop epoch
in every case**, which is the entire point of the change — the gap is being
measured mid-training, at the checkpoint that is actually tested. M4 ran to
the epoch cap without early stopping, consistent with the 0,6 observation
that M4 converges far later than the QNN-head models.
These accuracies are single-seed at reduced scale and are **not** a result;
they establish only that the pipeline runs and behaves sanely on 2,6.

**Checkpoint resume.** Forced a `rc=99` time-budget requeue at epoch 7, then
resumed. Two cases checked:
- best epoch *after* the resume → reported epoch 29 of a 30-epoch history.
- best epoch *before* the resume (epoch 7, beaten by nothing afterwards) →
  the loop variable is `None` because `save_checkpoint()` does not carry it,
  and the history derivation correctly recovered **7**. No `[WARN]`.
  This is the specific failure the derivation exists to prevent.

**Submit-script guards**, with a `qsub` stub on `PATH`:
- grid with no smoke result → refuses, exit 1
- `2,66` → refuses; `6,6` → refuses
- grid with a smoke result present → submits exactly **19** jobs, skips
  M2/seed 42, and all 19 carry `CLASSPAIR=2_6`
- `CLASSPAIR` expansion checked for `2_6`, `2_4` and `0_6` → correct
  `--classes` string and correct `results_v8_fashion_c<ab>` directory. The
  scripts are pair-generic, so the same package serves the 2,4 grid.

**`collect_results.py`** was exercised on both the real four-model output and
a synthetic five-seed grid; the MDE block returns ~1.5–2.2 pp, consistent
with the ~1.8 pp figure in `FMNIST_LOG.md` §2.

**Re-verified with qiskit entirely ABSENT from the environment** (v8.1) — a
stronger test than the earlier stub, and the one that would have caught the
175978087 failure before it reached the queue:

- `import train_gadi` succeeds with no qiskit installed at all.
- All four models retrained on 2,6 (n=2000/500/500, seed 42) and returned
  **bit-identical** results to the pre-change run — M1 86.40% (best ep 22),
  M2 83.00% (26), M3 85.60% (19), M4 89.20% (39). The lazy import changes no
  numerics.
- `--qnn_backend qiskit` raises the diagnostic `RuntimeError` naming the
  pinned stack, instead of qiskit's context-free packaging error.

**Still not covered:** the qiskit reference path cannot be executed here at
all, because the pinned 0.45.3 stack will not install in the test
environment. It is unchanged from v7 apart from where its imports live.

**Not verified locally:** the qiskit reference back-end path
(`--qnn_backend qiskit`), because the pinned qiskit 0.45.3 stack is not
installed in the local test environment. That path is unchanged from v7 and is
not used by these runs (`--qnn_backend torch` is the default and what the PBS
script uses), but it has not been re-exercised.
