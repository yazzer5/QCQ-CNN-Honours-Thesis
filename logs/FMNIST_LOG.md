# FMNIST LOG — Fashion-MNIST phase

**Student:** Yasser (z5477389) · **Supervisor:** A/Prof Vidhyasaharan Sethu
**Thesis:** End-to-End Trainable Quantum-Classical-Quantum CNNs for Image Classification
**Code version:** v7 · **Last updated:** 2026-08-03

Companion document: `MNIST_LOG.md` (MNIST 3-vs-5 grid and the 10-class diagnostic).

---

## 1. Why this phase exists

The MNIST 3-vs-5 grid returned a complete null: all four architectures within
0.08 pp of each other, every comparison p > 0.57. That was a **ceiling effect**,
not a finding — at 99.2% accuracy there is ~0.8 pp of headroom and the design
resolves ~0.25–0.40 pp. See `MNIST_LOG.md` §2.

This phase obtains a task with enough headroom that a real architectural
difference could become visible.

### Terminology

- **pp (percentage points)** — absolute difference between two percentages.
  99.22% vs 99.14% is 0.08 pp. Distinct from *percent*: 84 → 88% is +4 pp but a
  ~4.8% relative rise.
- **MDE (minimum detectable effect)** — smallest true difference the design has
  ~80% power to detect. Below it, real effects are usually missed. Lets a null
  be stated as a power result rather than a hedge.
- **Generalisation gap** — train accuracy minus test accuracy, both measured on
  the **same** model (see §6.1 for why this qualifier matters).

---

## 2. Task selection: Fashion-MNIST pairs over multiclass

Two routes to headroom were considered.

### MNIST 10-class — rejected

Rejected on two independent grounds, both later confirmed empirically by the
diagnostic in `MNIST_LOG.md` §4:

1. **It breaks the controlled comparison.** The QNN back-end emits a single
   expectation value into `Linear(1,2)`. Ten classes needs a wider readout,
   which changes parameter counts unequally across M1–M4, breaks the
   iso-parametric bound, and alters what M2−M3 and M3−M4 isolate.
2. **It does not deliver headroom.** A small CNN reaches ~95–96% on 10-class
   MNIST — the same ceiling, after a major architectural change.

Difficulty in MNIST/Fashion-MNIST comes from **class confusability**, not class
count.

### Fashion-MNIST binary pairs — adopted

M4 baseline, seed 42, one seed per pair:

| classes | pair | M4 acc | headroom |
|---|---|---|---|
| 0,1 | T-shirt/top vs Trouser | 98.80% | ~1 pp (saturated) |
| 0,3 | T-shirt/top vs Dress | 95.30% | 4.7 pp |
| 2,4 | Pullover vs Coat | 88.40% | 11.6 pp |
| 4,6 | Coat vs Shirt | 88.60% | 11.4 pp |
| 2,6 | Pullover vs Shirt | 87.10% | 12.9 pp |
| **0,6** | **T-shirt/top vs Shirt** | **84.20%** | **15.8 pp** |

Pair 0,6 gives more headroom than either multiclass option at **zero
architectural cost**.

Thesis A specifies pair (0,1). That pair is saturated and will reproduce the
MNIST null; run it for protocol fidelity, but it cannot answer the question.

### Difficulty gradient (planned)

| Rank | classes | pair | M4 baseline |
|---|---|---|---|
| 1 (hardest) | 0,6 | T-shirt/top vs Shirt | 84.20% |
| 2 | 2,6 | Pullover vs Shirt | 87.10% |
| 3 | 2,4 | Pullover vs Coat | 88.40% |

All three are upper-body garment confusions, so difficulty varies while the
*type* of discrimination is roughly held constant. A trend across three ordered
points is far more defensible than one significant comparison, and if all three
are null that locates the bottleneck away from task difficulty.

**Caveats to state:** only six of the 45 possible pairs were sampled — describe
as a targeted pilot, not an exhaustive search. And the baselines are one seed
each; 2,6 (87.10%) and 2,4 (88.40%) are 1.3 pp apart, inside the ~1.16 pp
per-run SE, so the ranking may not survive. Report the measured 5-seed ordering
once available rather than the pilot ordering.

### Statistical caveat

Binomial noise scales with p(1−p), so lower accuracy is measured less precisely.
At n = 1000 test images:

| accuracy | per-run SE | MDE (5 seeds) |
|---|---|---|
| 99.2% | ~0.28 pp | ~0.25–0.40 pp |
| 84.2% | ~1.16 pp | **~1.8 pp** |

The gain is **not** the full 15.8 pp of headroom. It is that effects above
~1.8 pp become detectable at all, where the ceiling previously made any effect
unobservable.

---

## 3. Baseline verification (prerequisite for this phase)

Before the Fashion-MNIST results can be read as a comparison against Long et
al., M1 must actually *be* Long's architecture. This was verified against their
published source and is documented in full in `MNIST_LOG.md` §8.

**Outcome: M1 reproduces Long's implementation exactly**, apart from the QNN
ansatz depth — Long's paper specifies 6 variational parameters, their code uses
4, and the two contradict each other. We follow the paper.

That discrepancy was **measured rather than argued**, on this dataset and class
pair (Fashion 0,6, the task with the most headroom), for both models carrying
the QNN back-end:

| model | reps | params | test acc (5 seeds) | Δ vs reps=1 | p |
|---|---|---|---|---|---|
| M1 | 1 | 1,602 | 83.76 ± 0.85% | — | — |
| M1 | 2 | 1,604 | 83.74 ± 0.71% | −0.02 pp | 0.975 ns |
| M2 | 1 | 1,608 | 84.48 ± 0.56% | — | — |
| M2 | 2 | 1,610 | 84.34 ± 1.40% | −0.14 pp | 0.773 ns |

Both differences are **below the fixed-seed GPU non-determinism floor** (~0.8 pp,
§6.5). The choice of ansatz depth does not affect any result in this document.

`reps=2` retained: matches Long's paper, preserves the iso-parametric bound.

---

## 4. Code changes (v6 → v7)

No model, protocol or numerical code was altered, so the MNIST results remain
comparable. Full detail in `CHANGELOG_v7.md`.

**4.1 `qsub -v CLASSES=0,6` never worked.** PBS's `-v` is itself a
comma-separated list, so this parsed as `CLASSES=0` plus a stray variable `6`.
The class pair never reached the trainer. Same family as defects 04 and 06: the
job would have completed and produced plausible numbers for the wrong
experiment. Fixed by passing `CLASSPAIR=0_6` and converting inside the job.

**4.2 Time-budget continuation reverted to MNIST.** The `rc=99` branch
resubmitted the MNIST script and dropped the class pair. Now resubmits itself
with `CLASSPAIR` carried through.

**4.3 Offline data.** Compute nodes have no internet. v6 relied on a fallback
inside torchvision's download path; v7 adds `_open_offline_first()` (tries
`download=False` first) and `scripts/preflight_data.py`, which MD5-verifies all
8 archives, extracts them so torchvision's existence check passes immediately,
proves an offline open, and validates the class pair. Runs at the top of every
job.

**Pre-submission verification:** 8/8 MD5s matched; FashionMNIST loaded with the
CDN unreachable; pair 0,6 has 6,000 train + 1,000 test per class against the
3,125/500 required; splits exactly 5000/1250/1000; labels remap {0,6} → {0,1}
with test balanced 500/500.

---

## 5. Run log

### 2026-08-02 — Smoke test (M2, seed 42)

M2 chosen deliberately: most machinery exercised, slowest model, and an
independent reference value to check against.

```
MODEL=M2 DATASET=fashion_mnist SEED=42 CLASSES=0,6 (FULL PROTOCOL)
INVERSION count: 0
[M2] FINAL test_acc=0.8360
```

Pipeline confirmed: class pair reached the trainer (fix 3.1 verified in situ),
zero inversions, full protocol, offline data path worked on a compute node.

### 2026-08-02 — 0,6 grid complete (20/20)

All `rc=0`, zero inversions, all `CLASSES=0,6`, all `FULL PROTOCOL`, no protocol
warnings. ~48 SU.

| Model | Front-end | Head | Test acc | Best val | Gap@best | Best ep | Stop ep |
|---|---|---|---|---|---|---|---|
| M1 | frozen Henderson | QNN | 83.98 ± 0.62% | 84.59% | 4.66 pp | 11.8 | 39.0 |
| M2 | trainable Circuit 4 | QNN | 84.60 ± 1.11% | 85.33% | 4.56 pp | 17.4 | 41.8 |
| M3 | trainable Circuit 4 | classical | 85.26 ± 1.03% | 86.00% | 4.38 pp | 36.2 | 47.8 |
| M4 | bilinear resize | classical | 85.68 ± 0.91% | 86.46% | 2.71 pp | 51.0 | 61.0 |

Per-seed test accuracy (seeds 42–46):

```
M1  84.9  84.3  83.4  83.5  83.8
M2  82.8  84.5  85.4  84.7  85.6
M3  84.8  85.7  83.7  86.3  85.8
M4  84.4  86.1  85.2  85.9  86.8
```

### 2026-08-11/12 — 2,6 grid complete (20/20), code v8

Code v8 changes protocol/numerics not at all; it fixes the gap metric to
`gap_at_best = train_acc(best checkpoint, eval mode) − test_acc(same
checkpoint)`, resolving §6.1. Integrity gate: `rc=0`, 0 inversions,
`CLASSES=2,6` echoed in every log, `n_params` 1604/1610/1600/1594, no
smoke/reduced-scale runs in the grid. ~48 SU.

| Model | Front-end | Head | Test acc | Best val | Gap@best | Best ep | Stop ep |
|---|---|---|---|---|---|---|---|
| M1 | frozen Henderson | QNN | 85.70 ± 0.41% | 88.37% | +7.31 pp | 13.6 | 39.0 |
| M2 | trainable Circuit 4 | QNN | 86.74 ± 0.58% | 89.22% | +7.90 pp | 20.8 | 39.0 |
| M3 | trainable Circuit 4 | classical | **87.70 ± 0.79%** | 90.08% | +5.97 pp | 32.4 | 43.4 |
| M4 | bilinear resize | classical | 87.00 ± 0.42% | 89.46% | +4.61 pp | 36.6 | 46.6 |

Attribution contrasts (paired t, n=5, df=4): M2−M1 +1.04 pp (t=2.40, p=0.075,
ns) — MDE 1.61 pp, below MDE. M2−M3 −0.96 pp (p=0.110, ns), MDE 1.75 pp.
M3−M4 +0.70 pp (p=0.246, ns), MDE 1.92 pp. All null; M2−M1 is the closest to
significance of any attribution contrast run so far. Gap@best: M2−M4
+3.29 pp (p=0.0040 \*\*), M1−M4 +2.70 pp (p=0.0080 \*\*), M2−M3 +1.93 pp
(p=0.0447 \*) — the QNN-head models show a materially larger train/test gap
than M4, replicating the M1/M2/M3-vs-M4 split from §6.1's corrected 0,6
numbers on an independent pair.

Full stats: `results/results_v8_fashion_c26/RESULTS_2_6.txt`. Plots and a
standalone report: `results/results_v8_fashion_c26/analysis/`.

### 2026-08-12 — 2,4 grid complete (20/20), code v9

Code v9 is v8 with a version tag change only. Integrity gate: `rc=0`, 0
inversions across all 20 jobs (including the M2/seed42 smoke run),
`CLASSES=2,4` echoed in every log, `n_params` 1604/1610/1600/1594, no
smoke/reduced-scale runs counted in the grid. This closes the three-pair
difficulty sequence (0,6 → 2,6 → 2,4).

| Model | Front-end | Head | Test acc | Best val | Gap@best | Best ep | Stop ep |
|---|---|---|---|---|---|---|---|
| M1 | frozen Henderson | QNN | 87.30 ± 0.80% | 89.09% | +6.96 pp | 19.8 | 40.6 |
| M2 | trainable Circuit 4 | QNN | 87.00 ± 1.19% | 89.63% | +7.90 pp | 20.2 | 39.0 |
| M3 | trainable Circuit 4 | classical | **88.42 ± 0.80%** | 90.14% | +4.80 pp | 30.4 | 41.8 |
| M4 | bilinear resize | classical | 87.18 ± 0.40% | 88.85% | +3.54 pp | 40.6 | 50.6 |

Attribution contrasts: M2−M1 −0.30 pp (t=−0.49, p=0.647, ns), MDE 2.26 pp.
M2−M3 −1.42 pp (p=0.167, ns), MDE 3.13 pp. M3−M4 +1.24 pp (t=3.63,
p=0.022 \*, uncorrected) — the only nominally-significant accuracy contrast
across all three Fashion-MNIST pairs, but it does not survive Holm correction
across the three attribution tests and sits just below its own MDE (1.27 pp),
so it is flagged as marginal rather than confirmed. Gap@best: M2−M4 +4.36 pp
(p=0.0027 \*\*), M1−M4 +3.42 pp (p=0.0192 \*) — same QNN-head-vs-M4 split as
0,6 and 2,6. M2−M3 gap contrast +3.10 pp (p=0.0113 \*).

**M2 vs M1 direction flips sign between the two new pairs** (+1.04 pp at 2,6,
−0.30 pp at 2,4), both magnitudes well inside their respective MDEs — the
data are consistent with a small, real, direction-uncertain effect, not with
either "M2 beats M1" or "M2 equals M1" as established fact. See §6.7 for the
pooled read across both pairs.

Full stats: `results/results_v9_fashion_c24/RESULTS_2_4.txt`. Plots and a
standalone report: `results/results_v9_fashion_c24/analysis/`.

---

## 6. Analysis

### 5.1 Metric correction — read this before using any gap number

An earlier draft of this log reported the generalisation gap as
`final-epoch train accuracy − test accuracy` and claimed it was monotone in
quantum content at p < 10⁻⁴.

**That claim was withdrawn.** `train_gadi.py` line 1030 restores the
best-validation checkpoint before testing, so the tested model is the one from
epoch ~12–17, while final-epoch train accuracy comes from epoch ~39–61. The two
numbers describe **different models**. M1/M2 continue overfitting after their
peak, so their final-epoch train accuracy is inflated relative to the model
being evaluated — and that inflation produced the apparent ordering.

Measured consistently at the best epoch:

| contrast | Δ gap@best | t | p |
|---|---|---|---|
| M2 − M1 | −0.10 pp | −0.11 | 0.914 ns |
| M2 − M3 | +0.18 pp | +0.20 | 0.854 ns |
| M3 − M4 | +1.67 pp | +1.50 | 0.207 ns |
| M1 − M4 | +1.95 pp | +2.66 | 0.057 |
| **M2 − M4** | **+1.86 pp** | **+3.60** | **0.023 \*** |

M1, M2 and M3 are statistically indistinguishable on overfitting. Only M4 stands
apart. The dose-response reading was an artefact of the metric definition.

**Lesson for the confirmatory runs:** fix the metric definition *before* the 2,6
and 2,4 grids. Any gap must compare train and test on the same checkpoint.

### 5.2 Accuracy — pre-registered comparisons

| contrast | isolates | Δ | t (df=4) | p |
|---|---|---|---|---|
| M2 − M1 | front-end trainability | +0.62 pp | +0.83 | 0.454 ns |
| M2 − M3 | quantum back-end | −0.66 pp | −1.00 | 0.375 ns |
| M3 − M4 | quantum front-end | −0.42 pp | −1.11 | 0.328 ns |

Mean within-model SD = 0.92 pp → **MDE ≈ 1.8 pp**. The adjacent gaps are
0.4–0.7 pp, below the MDE. All three pre-registered contrasts reproduce the
MNIST null.

### 5.3 What does reach significance

| contrast | Δ | p |
|---|---|---|
| M1 − M4 | −1.70 pp | **0.046 \*** |
| M2 − M4 | −1.08 pp | **0.031 \*** |
| QNN head (M1,M2) vs classical head (M3,M4) | −1.18 pp | **0.011 \*** |

The end-to-end contrast is significant; it is simply spread across two
individually-undetectable steps. **The fully classical control M4 is the best
model on this task**, and models carrying the QNN head are reliably ~1.2 pp
behind.

The accuracy ordering M1 < M2 < M3 < M4 is monotone in quantum content, but
monotonicity alone is weak evidence — 1 in 24 orderings of four items matches
any given ranking by chance. The significant grouped contrast is the stronger
statement.

### 5.4 Convergence behaviour

M1/M2 peak at epoch 12–17; M4 peaks at epoch 51 and is still improving. So the
QNN-head models exhaust the available signal roughly four times sooner. Because
the best checkpoint is restored, the subsequent overfitting does not reach the
test set — it is wasted compute, not damage. `min_epochs = 30` forces M1/M2
through ~25 epochs past their peak for no benefit.

### 5.5 Fixed-seed runs are not bit-reproducible

M2/seed 42 ran twice with identical configuration: **83.60%** (smoke) and
**82.80%** (grid). Epochs 1–2 are bit-identical (train 0.4968, val 0.5128);
epoch 3 diverges (0.7186 vs 0.7190).

Cause: non-deterministic GPU float reduction order
(`torch.use_deterministic_algorithms` is not set). Not a bug, but three
consequences:

- The seed does not fully determine a run, which **weakens the pairing** in the
  paired t-tests — some "paired" variance is machine noise.
- Run-to-run noise at fixed seed (~0.8 pp) is comparable to between-seed SD
  (0.92 pp). **Roughly half the observed variance may be non-determinism.**
- Must be disclosed. Re-running one grid deterministically would materially
  strengthen §6.3.

### 5.6 Why the QCQ architecture is binary-shaped

The 10-class diagnostic (`MNIST_LOG.md` §4) shows the QNN head has two
structural limits: a single-observable readout that can only express a 1-D
ordering of classes, and a 64 → 2 compression before the circuit.

Neither binds for binary classification — one scalar thresholds two classes
perfectly, and 64 → 2 is far less lossy with only two classes to separate. This
is why M1/M2 sit at ~84% here rather than ~30% on 10 classes, and it is the
strongest available argument for staying binary and obtaining headroom from
harder class pairs.

### 5.7 Cross-experiment synthesis (v6 + v7 + v8 + v9, 80 runs total)

With all three Fashion-MNIST pairs and the MNIST saturation reference
complete, the following holds across the full set:

**The pre-planned difficulty gradient collapsed.** 0,6 was intended as the
hardest pair (M4 84.20%), 2,6 the middle pair, 2,4 the easiest. Measured M4
accuracy: 0,6 = 84.20%, 2,6 = 87.00%, 2,4 = 87.18% — 2,6 and 2,4 landed only
0.18 pp apart, inside the ~0.8 pp non-determinism floor established in §5.5.
The gradient exists between 0,6 and the other two pairs but not between 2,6
and 2,4. Framing this as three ordered difficulty levels is not supported by
the data; framing it as "one hard pair, two similarly-easy pairs" is.

**M2 vs M1, pooled across the two new pairs.** Per-seed win count: M2 beat M1
in 7 of 10 seeds (5 at 2,6, 2 at 2,4). Both grid-level contrasts are
individually null (§5's run-log entries above) and the sign flips between
pairs. A 7/10 win rate with both point estimates inside their MDE is a power
statement, not a directional finding — it is consistent with a small true
positive effect obscured by noise, and equally consistent with zero true
effect. This does **not** license abandoning the M2 > M1 hypothesis; it
licenses running enough seeds to resolve it (see the six-step plan in §9).

**Gap@best is monotone in quantum content at both correctly-measured pairs**
(2,6 and 2,4; 0,6 used the withdrawn metric, §6.1). M2 > M1 > M3 > M4 in
gap@best at 2,6; a similar split at 2,4. M2−M4 survives Holm correction
across the five gap contrasts reported per grid (p = 0.0040 at 2,6, p =
0.0027 at 2,4). The QNN head consistently overfits more than the classical
head at a fixed best-checkpoint comparison — this is now a three-for-three
replication once 0,6 is excluded for its metric issue.

**The pre-registered grouped QNN-head accuracy penalty (M1,M2 vs M3,M4)
replicated at 0,6 (−1.18 pp, p = 0.011) but not at 2,4** (M3−M4 alone, the
closest single-pair analogue, sits at p = 0.022 uncorrected and below its own
MDE). It was not separately tested as a grouped contrast at 2,6 in the
original run; doing so retrospectively is possible but was not pre-registered
for that grid, so it is reported here as an open item rather than a result.

**M3 is the best single model at both 2,6 and 2,4**, not M4 as predicted
before either grid ran (§9's pre-registered predictions, now falsified on
that specific point). M4 was best only at 0,6. This reverses the working
assumption that the fully classical control dominates as tasks get easier —
worth flagging explicitly to the supervisor as a registered prediction that
did not hold.

Plots for all four experiments (accuracy by model, training curves, and
gap@best where measured) are in each experiment's `results/<dir>/analysis/`
folder, with an index at `results/analysis_index.html`.

---

## 7. Figures

| file | content |
|---|---|
| `fig1_accuracy_and_gap.png` | accuracy and gap side by side — **panel (b) subtitle is wrong, regenerate with gap@best before use** |
| `fig2_training_curves.png` | per-seed train/val curves, all four models |
| `fig3_gap_vs_acc_and_epochs.png` | gap-vs-accuracy; best vs stop epoch — **also uses the withdrawn metric** |

Both flagged figures need regenerating against §6.1.

**Per-experiment analysis plots (2026-08-13):** each of the four completed
grids now has its own `analysis/plots/` folder and a standalone
`analysis/report.html` under its results directory:

| Experiment | Path |
|---|---|
| MNIST 3v5 (v6) | `results/results_v6_mnist_3v5/analysis/` |
| Fashion 0,6 (v7) | `results/results_v7_fashion_c06/analysis/` |
| Fashion 2,6 (v8) | `results/results_v8_fashion_c26/analysis/` |
| Fashion 2,4 (v9) | `results/results_v9_fashion_c24/analysis/` |

Each contains `accuracy_by_model.png`, `training_curves.png`, and (v8/v9 only,
since v6/v7 predate the gap@best fix) `gap_at_best.png`. A top-level index at
`results/analysis_index.html` links all four reports. These are generated
directly from the raw `result_*.json` files by
`build_analysis.py` (script not yet checked into the repo — see
§9 open items) and are independent of `fig1–fig3` above, which remain the
figures drafted for the thesis document itself.

---

## 8. To raise with supervisor

1. **Task selection** — endorse the difficulty gradient over multiclass, on the
   grounds in §2, now supported empirically by the diagnostic.
2. **MDE** — the null must be written as a power statement with the MDE given,
   not as evidence of equivalence.
3. **The metric correction (§6.1)** — flag proactively. It was caught and
   retracted before it entered the thesis; that is the process working, but it
   must be disclosed rather than quietly fixed.
4. **Non-determinism (§6.5)** — decide whether to re-run one grid with
   deterministic algorithms enabled.
5. **Thesis A §4.4.1** — the claim that minibatch averaging cancels
   parameter-shift gradients is mathematically incorrect. Parameter-shift gives
   the exact per-sample gradient; averaging exact per-sample gradients gives the
   exact minibatch gradient by linearity, and the ± shifts are subtracted
   *within* each sample before averaging. Needs rewriting.
6. **Aim 3 (noise)** — largest unstarted piece, 5/6 of the 450-run matrix.
   Given M1/M2 show no accuracy advantage, the interesting reframing is whether
   noise acts as a **regulariser**.
7. **M1 framing** — not a strict replication of Long et al. (batch 32 vs 1,
   autograd vs parameter-shift, early stopping vs flat 100 epochs, 5 seeds vs
   10). Describe as a re-implementation under a harmonised protocol.

---

## 9. Where we are up to

### Completed
- v7 built; two silent-failure bugs fixed; offline data path verified end-to-end
- **Baseline verified: M1 matches Long's implementation** (§3, `MNIST_LOG.md` §8)
- **reps=1 vs reps=2 ablation** — null, ansatz depth immaterial (§3)
- **Fashion-MNIST 0,6 grid complete — 20/20 clean** (code v7)
- **Fashion-MNIST 2,6 grid complete — 20/20 clean** (code v8, corrected gap@best metric)
- **Fashion-MNIST 2,4 grid complete — 20/20 clean** (code v9) — closes arm (a)
  for all three planned pairs
- Analysis complete for all four grids (v6/v7/v8/v9, 80 runs total), including
  one metric retraction (§6.1) and the cross-experiment synthesis (§5.7)
- Per-experiment accuracy/training-curve/gap@best plots and standalone HTML
  reports generated for all four grids (§7)

### Established
- Pre-registered accuracy contrasts null at every pair tested, even with
  15.8 pp headroom at 0,6 (MDE ≈ 1.8–2.3 pp depending on pair)
- **M3 (trainable front-end + classical head), not M4, is the best model at
  2,6 and 2,4** — M4 was best only at 0,6. This falsifies pre-registered
  prediction (1) below; see §5.7.
- **The difficulty gradient did not hold between 2,6 and 2,4** — they landed
  0.18 pp apart in M4 accuracy, inside the ~0.8 pp non-determinism floor.
  Prediction (2) below is falsified as stated; the gap does narrow from 0,6
  to the other two pairs, just not further between them.
- M1/M2 converge ~2–3× earlier than M3/M4 at every pair tested — prediction
  (3) holds directionally, though the multiplier shrinks as pairs get easier
  (4× at 0,6, ~2× at 2,6/2,4)
- Gap@best is **not** flat across M1/M2/M3 as prediction (4) expected — it is
  monotone in quantum content (M2 > M1 > M3 > M4) at both 2,6 and 2,4, and
  M2−M4 survives Holm correction at both pairs (§5.7)
- **M2 vs M1 direction flips sign** between 2,6 (+1.04 pp) and 2,4 (−0.30 pp);
  pooled 7/10 seed win rate for M2. Both individual contrasts remain below
  MDE — underpowered, not resolved. The core M2 > M1 hypothesis stays open,
  not abandoned (see six-step plan below)
- Fixed-seed runs vary ~0.8 pp from GPU non-determinism
- All four grids are **valid and none need re-running** on integrity grounds
  — every job passed the 7-point gate cleanly

### Outcome of the pre-registered predictions (registered 2026-08-11/12, before 2,6/2,4 ran)

| # | Prediction | Outcome |
|---|---|---|
| 1 | M4 remains best model on both pairs | **Falsified** — M3 was best at both |
| 2 | M1/M2-vs-M3/M4 gap narrows monotonically across 0,6→2,6→2,4 | **Partially falsified** — narrows 0,6→2,6, then flat 2,6→2,4 |
| 3 | M1/M2 converge ~3–4× earlier than M4 | **Held**, magnitude softening as task eases |
| 4 | Gap@best flat across M1/M2/M3, only M4 lower | **Falsified** — gap@best is monotone in quantum content |

Registering these before running paid off precisely by failing on two of
four points — that is what makes the exercise a real test rather than a
formality, and both misses are now documented findings in their own right
(§5.7) rather than silently revised expectations.

### NEXT — six-step improvement plan (prioritised, per cross-experiment synthesis)

1. **Save per-image predictions** for every future run → enables McNemar's
   test, a more powerful within-model-pair comparison than the current paired
   t-test on aggregate accuracy
2. **Use the full 2,000-image Fashion-MNIST test set** (currently 1,000) —
   free resolution improvement, no extra training cost
3. **Enable `torch.use_deterministic_algorithms`** → should roughly halve the
   ~0.8 pp non-determinism floor identified in §5.5, strengthening every
   paired comparison
4. **Pilot a better difficulty ladder** via M4-only runs across ~10 candidate
   Fashion-MNIST pairs before committing another 4-model × 5-seed grid to a
   pair — the 2,6/2,4 collapse shows accuracy alone is a poor difficulty proxy
   chosen a priori
5. **Fix the 10-class MNIST multiclass readout** to 3+ observables and
   rebalance the parameter budget (`MNIST_LOG.md` §9)
6. **Run 10-class MNIST properly** — the diagnostic already shows
   M2−M1 ≈ +6.4 pp, roughly 3× any MDE seen in the binary experiments, and is
   the single most promising unexplored direction for resolving the central
   M2 vs M1 hypothesis with adequate power

### Also outstanding
- **Long-faithful arm (b)** for all three pairs (0,6 / 2,6 / 2,4) — 60 runs,
  not yet started; config in `MNIST_LOG.md` §8.6
- **Environment repair** (stale qiskit dist-info) — deferred, must happen
  before Aim 3 (noise), which needs Aer's NoiseModel via the qiskit path
- **v8/v9 code files** (`train_gadi.py` etc.) not yet checked into
  `code/v8_fashion/` or `code/v9_fashion/` in this repo — only results, job
  logs, and docs have been added so far (2026-08-13). Add once available.
- Regenerate `fig1_accuracy_and_gap.png` and `fig3_gap_vs_acc_and_epochs.png`
  against gap@best (§7) — still not done, separate from the new
  per-experiment `analysis/` plots

### Deferred
- Noise sweep, reframed as a regularisation question
- Barren-plateau measurement
- MRI experiments
- Pair 0,1 for protocol fidelity to Thesis A (saturated; expect a null)
