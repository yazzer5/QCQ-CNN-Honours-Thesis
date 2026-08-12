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

---

## 7. Figures

| file | content |
|---|---|
| `fig1_accuracy_and_gap.png` | accuracy and gap side by side — **panel (b) subtitle is wrong, regenerate with gap@best before use** |
| `fig2_training_curves.png` | per-seed train/val curves, all four models |
| `fig3_gap_vs_acc_and_epochs.png` | gap-vs-accuracy; best vs stop epoch — **also uses the withdrawn metric** |

Both flagged figures need regenerating against §6.1.

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
- **Fashion-MNIST 0,6 grid complete — 20/20 clean**
- Analysis complete, including one metric retraction (§6.1)

### Established
- Pre-registered accuracy contrasts null even with 15.8 pp headroom
  (MDE ≈ 1.8 pp)
- **M4, the fully classical control, is the best model on this task**
- QNN-head models ~1.2 pp behind, p = 0.011
- M1/M2 converge ~4× earlier than M4 (epoch 12–17 vs 51)
- Only M4 differs significantly on overfitting; M1/M2/M3 indistinguishable
- Fixed-seed runs vary ~0.8 pp from GPU non-determinism
- The 0,6 grid is **valid and does not need re-running** — the baseline
  verification confirmed M1, so its M1 column stands

### NEXT EXPERIMENTS

Full list with rationale in `MNIST_LOG.md` §9. Fashion-MNIST portion:

| # | Experiment | Runs | Command |
|---|---|---|---|
| 1 | 0,6 arm (a) iso-parametric | — | ✅ DONE |
| 2 | **2,6 arm (a)** | 20 | `./submit_run7_fashion.sh 2,6` |
| 3 | **2,4 arm (a)** | 20 | `./submit_run7_fashion.sh 2,4` |
| 4 | 0,6 / 2,6 / 2,4 arm (b) Long-faithful | 60 | config change, see `MNIST_LOG.md` §8.6 |

**One full-scale smoke test before each grid.** Non-negotiable.

### Register this prediction BEFORE running 2,6 and 2,4

Writing it down first is what makes these runs confirmatory rather than
exploratory, and examiners distinguish the two:

1. **M4 remains the best model** on both pairs.
2. **The M1/M2 vs M3/M4 accuracy gap narrows** as the task eases — 0,6 (84%)
   showed 1.18 pp; expect less at 2,6 (87%) and less again at 2,4 (88%),
   trending toward the 0.08 pp seen at MNIST saturation.
3. **M1/M2 continue to converge ~3–4× earlier** than M4.
4. **Gap@best remains flat across M1/M2/M3**, with only M4 lower.

If (2) holds across three difficulty levels, that is a dose-response result on
task difficulty and far stronger than any single comparison. If it fails, the
ordering is not difficulty-driven and that redirects effort to Aim 3.

### Immediate housekeeping
1. Regenerate the two flagged figures using gap@best (§7) — **do not show the
   current versions**
2. Fix the gap metric definition in code before further runs
3. Run `scripts/verify_reps.py` to completion (login node, free)

### Deferred
- Noise sweep, reframed as a regularisation question
- Barren-plateau measurement
- MRI experiments
- Pair 0,1 for protocol fidelity to Thesis A (saturated; expect a null)
