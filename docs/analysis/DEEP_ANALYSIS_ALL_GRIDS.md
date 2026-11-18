# Deep Analysis: All Arm-(a) Grids (v6–v12)

**Date:** 2026-10-04
**Data:** 420 runs across 12 tasks and 4 datasets. All are iso-parametric arm (a), M1–M4, noise-free.
**Scripts:** `analysis/common.py`, `a1_contrasts.py`, `a2_predictions.py`, `a3_dynamics_cost.py`
**Outputs:** `figures_all/F01–F11` (200 dpi), `tables_all/*.csv`
**Status:** Exploratory and secondary. Only the v12 contrasts and TOST were pre-registered
(see `V12_RESULTS.md`). Everything here goes beyond accuracy and was decided after
seeing the data, so it is hypothesis-generating unless stated otherwise.

| Task | Grid | Classes | Seeds | Protocol | Per-image predictions |
|---|---|---|---|---|---|
| MNIST 3v5 | v6 | 2 | 5 | BCE, max 100, patience 10 | no |
| FMNIST 0v6 / 2v6 / 2v4 | v7 / v8 / v9 | 2 | 5 | BCE, max 100, patience 10 | no |
| Brain MRI | v11 | 2 | 5 | BCE, max 100, patience 10 | yes |
| MNIST K=3, 4, 5 (two subsets each) | v12 | 3–5 | 10 | CE, max 200, patience 20 | yes |
| MNIST 10-class | v10 | 10 | 5 | CE, max 200, patience 20 | yes |

"Difficulty" below means M4's test error on that task, a model-independent measure.

---

## Headline: five things the accuracy table alone didn't show

1. **The quantum back-end loses on reliability, not capacity.** Up to 5 classes, the
   quantum-head models fit the training set as well as the classical ones (~99% train
   accuracy) but generalise worse, and each seed fails on *different* images. Majority-vote
   10 seeds and the gap almost vanishes (K=5: 88.8% → 97.5%, matching M3's ensemble).
   This changes the v13 plan: widening the readout may not be the main fix.
2. **The back-end's cost is driven by the number of classes, not task difficulty.** Hard
   binary tasks (FMNIST at ~13% error, MRI at ~20%) show only a small back-end cost.
   This corrects F5 in `V12_RESULTS.md`.
3. **The quantum front-end never helps after correction, and on the one real-world dataset
   it hurts.** Brain MRI: M3−M4 = −3.7 pp (Holm p = 0.002). On MNIST, M3 and M4 get the
   same images wrong about as often as two seeds of M4 do.
4. **Training the front-end: at most +0.6 pp on average across all 12 tasks.** It's
   consistently slightly positive on the harder multi-class tasks, but never robust across seeds.
5. **Cost.** In simulation, M2 costs 3–5× M4's compute per run and is never significantly more accurate.
   M3 costs ~2× M4 for equal accuracy.

---

## 1. Cross-task synthesis (F01, F02, F03)

Paired over seeds within each task, Holm-corrected within task. Pooled with a
DerSimonian–Laird random-effects model across the 12 tasks. The pooling is
**descriptive**: tasks differ in protocol, and the six v12 subsets share MNIST.

| Contrast | Pooled (pp) | 95% CI | Tasks positive | Significant (Holm) | Equivalent ±1 pp |
|---|---|---|---|---|---|
| M2 − M1 (train the Q front-end) | +0.24 | [−0.13, 0.60] | 8 / 12 | 0 / 12 | 2 / 12 |
| M3 − M2 (drop the Q back-end) | **+1.94** | [1.08, 2.80] | 11 / 12 | 4 / 12 | 1 / 12 |
| M3 − M4 (Q front-end vs none) | −0.26 | [−0.66, 0.13] | 6 / 12 | 1 / 12 (MRI, negative) | 6 / 12 |
| M4 − M2 (classical CNN vs ours) | **+2.33** | [1.24, 3.42] | 11 / 12 | 6 / 12 | 1 / 12 |

Heterogeneity is high for every contrast except M2−M1 (I² = 0.87–0.95), so the pooled
mean summarises a range rather than one true effect. That heterogeneity is the point of §2.

**Training the front-end (M2−M1).** Across all tasks the pooled effect is +0.24 pp, and
the CI rules out an average benefit above ~0.6 pp. The seed-level test finds nothing anywhere.
A secondary per-image test (image as the unit, seeds averaged; valid because each task's
test set is fixed) finds M2 > M1 at K=4 {0,2,3,8} (+2.3 pp), K=5 {1,3,5,8,9} (+2.9) and
K=5 {0,2,6,8,9} (+1.6), all p < 0.001. Those are the tasks where the quantum head struggles
most. This test conditions on these specific seeds, so it says "for these runs, on this
test set", not "for a new seed". **Reading:** training the front-end may partly compensate
for a weak quantum head on harder multi-class tasks, but the effect doesn't survive
seed-to-seed variation. Treat it as a lead, not a result.

**Quantum front-end vs none (M3−M4).**
- MNIST (7 tasks): every estimate within ±0.7 pp. Equivalent at ±1 pp on 6 of 7 (the
  exception, K=4 {0,2,3,8}, leans slightly classical).
- Fashion-MNIST: 2v4 is the only positive outlier (+1.24 pp, 95% CI [0.29, 2.19], Holm
  p = 0.089, so not significant after correction). 0v6 and 2v6 are inconclusive.
- **Brain MRI: −3.68 pp, 95% CI [−4.78, −2.58], Holm p = 0.002.** The quantum front-end
  makes the network significantly worse on the only non-benchmark dataset. Its
  generalisation gap is also larger (17.2 vs 13.6 pp, §3).

So across 12 tasks there is no condition where the quantum front-end helps after
correction, and one where it clearly hurts.

---

## 2. The back-end's cost depends on the number of classes (F01, F02)

| Task type | Difficulty (M4 error) | M3 − M2 |
|---|---|---|
| MNIST 3v5 | 0.9% | 0.00 pp |
| Binary FMNIST (3 tasks) | 12.8–14.3% | +0.66 to +1.42 pp (none significant) |
| Binary MRI | 19.7% | +1.79 pp (Holm 0.088) |
| MNIST K=3 / 4 / 5 | 1.7–7.2% | +1.3 to +5.5 pp |
| MNIST K=10 | 17.6% | +11.0 pp |

Across all 12 tasks, back-end cost vs difficulty gives Spearman ρ = 0.32, p = 0.31: not
significant. The binary tasks sit low on the cost axis however hard they are. Within the
multi-class tasks (K ≥ 3), cost does track difficulty (ρ = 0.86, n = 7).

**Correction to `V12_RESULTS.md` F5:** "back-end cost tracks task difficulty,
not just K" holds only *within* multi-class tasks. Across all tasks, the number of
classes is the main driver. A quantum head with 2 classes costs little even on hard
data. With more classes it costs more, and more again when those classes are hard to separate.

---

## 3. Why the quantum head loses: variance, not capacity (F07, F09, F10, F08)

**3a. It fits the training data just as well, then generalises worse.**

| Task | M2 train acc at best ckpt | M3 train acc at best ckpt | M2 gap | M3 gap |
|---|---|---|---|---|
| K=4 {0,2,3,8} | 99.1% | 99.5% | 7.7 pp | 4.1 pp |
| K=5 {1,3,5,8,9} | 99.0% | 99.6% | 10.3 pp | 7.0 pp |
| K=5 {0,2,6,8,9} | 99.1% | 99.4% | 11.0 pp | 5.8 pp |
| MNIST-10 | 87.0% | 94.0% | 16.1 pp | 12.1 pp |

Up to K=5 the quantum head is not short of capacity on the training set; its problem is
generalisation. At K=10 it also starts to underfit (87% vs 94% train), which is where a
readout-width limit would show up.

**3b. Each seed fails on different images, so ensembling recovers almost everything (F07).**
Majority vote over all seeds on the same fixed test set:

| Task | M2 single seed | M2 ensemble | M3 ensemble | Gap: single → ensemble |
|---|---|---|---|---|
| K=4 {0,2,3,8} | 91.4% | 98.6% | 98.75% | 4.0 → 0.1 pp |
| K=5 {1,3,5,8,9} | 88.8% | 97.5% | 97.5% | 3.9 → 0.0 pp |
| K=5 {0,2,6,8,9} | 88.2% | 97.6% | 97.0% | 5.5 → −0.6 pp |
| MNIST-10 (5 seeds) | 70.9% | 89.3% | 91.3% | 11.0 → 2.0 pp |

On the four hardest multi-class tasks, ensembling gains 7–18 pp for the quantum-head
models against 3–9 pp for the classical-head ones (on the easy K=3 tasks, 1.6–2.5 vs
0.7–1.1 pp). Error overlap between two seeds of M2 is low (Jaccard
0.14–0.26), lower than between two seeds of M3 (0.19–0.29).

**3c. There are no "good seeds" to pick.** Choosing the best of 5 seeds *by validation
accuracy* (a legitimate selection rule) leaves the K=5 gap at 5.4–6.5 pp and the K=10 gap
at 10.9 pp. Every quantum-head run is worse; they're just worse on different images.

**3d. Losses land on specific digits, and sometimes a whole class collapses (F04, F05).**
At MNIST-10, M2's mean recall on digit 3 is 51% (M3: 79%). In 3 of 5 M2 seeds at least
one digit is recalled below 50%; M1 does this in 2 of 5, and M3/M4 never do. At K=5 the
worst-class recall is ~77–80% for M2 against ~87–89% for M3.

**3e. It learns 2–3× slower and less smoothly (F08).** Epochs to reach 95% of the best
validation accuracy at K=5 {1,3,5,8,9}: M2 45, M1 41, M3 15, M4 21. At MNIST-10: M2 72,
M1 109, M3 22, M4 29. Validation curves are rougher for the quantum heads. On Brain MRI,
M1/M2 validation accuracy peaks around epoch 12 and then declines (overfitting), while
M3/M4 keep improving.

**What this means.** The quantum back-end's deficit is mostly an
**optimisation/variance problem**: each training run lands in a different, mediocre
solution. It is not a hard ceiling on what three measurements can represent, at least up
to K=5. Readout width may matter at K=10, where the quantum head also underfits.

**Implication for v13.** Testing readout width alone could miss the main cause. v13 should
compare, at K=5 and K=10:
- (a) wider readout (more observables)
- (b) a variance fix at the same width (stronger regularisation, or lower learning rate /
  longer schedule for the head)
- (c) both

Seed ensembling is the upper bound here, but it costs N× the compute, so it's a
diagnostic, not a recommendation.

---

## 4. The quantum front-end doesn't change *which* images are wrong (F06)

Error overlap (Jaccard) between M3 and M4 on the same seed, against the overlap between two
different seeds of M4 (the noise floor):

| Task | M3/M4 | M4 seed vs seed |
|---|---|---|
| K=3 {1,4,9} | 0.35 | 0.35 |
| K=3 {2,4,9} | 0.41 | 0.40 |
| K=4 {0,2,3,8} | 0.32 | 0.19 |
| K=4 {0,1,2,9} | 0.38 | 0.31 |
| K=5 {1,3,5,8,9} | 0.28 | 0.24 |
| K=5 {0,2,6,8,9} | 0.33 | 0.27 |
| MNIST-10 | 0.36 | 0.33 |
| **Brain MRI** | **0.46** | **0.63** |

On every MNIST task, M3 and M4 share errors at least as much as two runs of M4 do. Adding
the quantum front-end does not change what the network finds hard; it behaves like a
re-seed of M4. On MRI the overlap drops well below the floor: there the front-end
changes which images fail, and for the worse (§1).

---

## 5. Compute cost (F11)

Measured in GPU simulation from per-epoch timers and PBS logs. Real quantum hardware would
add shot-sampling cost on top, so these are **lower bounds** for the quantum models.

| Task | M2 s/epoch | M4 s/epoch | M2 SU/run | M4 SU/run | M2/M4 SU | M3/M4 SU |
|---|---|---|---|---|---|---|
| K=3 {1,4,9} | 1.78 | 0.44 | 1.42 | 0.50 | 2.8× | 1.7× |
| K=5 {1,3,5,8,9} | 3.00 | 0.73 | 2.97 | 0.78 | 3.8× | 1.9× |
| MNIST-10 | 5.96 | 1.46 | 8.22 | 1.55 | 5.3× | 2.4× |
| Brain MRI | 2.61 | 0.39 | 1.62 | 0.45 | 3.6× | 2.2× |

The quantum head costs more per epoch *and* needs more epochs (§3e), so the cost ratio
grows with K. M4 is the cheapest model on every task. The only tasks where a quantum model
has a higher mean accuracy than M4 are FMNIST 2v6 and 2v4 (M3 +0.7 and +1.2 pp, at about
2× the compute), and neither difference survives correction. **Nowhere does a quantum
model buy a significant accuracy gain for its extra compute.**

---

## 6. Caveats

- **Pooling.** The 12 tasks are not independent (the v12 subsets share MNIST) and come
  from two protocols (BCE/patience 10/5 seeds vs CE/patience 20/10 seeds). The pooled
  estimates are descriptive summaries, not a formal meta-analysis.
- **Per-image tests** (§1, secondary) condition on the seeds that were run. They test
  "these runs on this test set", and don't replace the pre-registered seed-level test.
- **Brain MRI** is one dataset with 5 seeds and the BCE protocol. The −3.7 pp result is
  strong (Holm p = 0.002) but should be replicated before it's generalised beyond "on
  this dataset".
- **One budget-bound run.** v10 M1 seed 44 hit the 200-epoch cap (best epoch 198, test
  71.15%). It's one run of 20, but flag it in the v10 write-up. No v6–v9, v11 or v12 run
  hit its cap.
- **Ensembles** (§3b) use N× the compute and are a diagnostic of error structure, not a
  proposed model.
- Compute figures are simulation time on Gadi V100 nodes, not quantum-hardware cost.

---

## 7. Updated one-paragraph summary for the supervisor

> Across 12 tasks on four datasets, the quantum front-end never improves accuracy after
> correction: on MNIST it is equivalent to having no front-end, and on brain MRI it is
> significantly worse (−3.7 pp). The hybrid's deficit comes from the quantum back-end,
> whose cost grows with the number of classes, not with difficulty alone. Looking past
> accuracy, the quantum back-end isn't short of capacity: up to five classes it fits the
> training data as well as a classical head, but generalises worse, learns 2–3× slower,
> and each run fails on different images, so averaging ten runs almost closes the gap. It
> also costs 3–5× more compute in simulation. So the hybrid's problem is the instability
> of training the quantum classifier, which points v13 at variance, not just readout width.

---

## 8. Files

| Figure | What it shows |
|---|---|
| F01_forest_all_contrasts | All 4 contrasts × 12 tasks with 95% CI, plus pooled estimate |
| F02_contrast_vs_difficulty | Back-end cost and front-end effect against task difficulty |
| F03_accuracy_table_all_tasks | Mean ± SD accuracy, every task × model |
| F04_per_class_recall | Per-digit recall, every multi-class task |
| F05_confusion_mnist10_M2_vs_M3 | MNIST-10 confusion matrices and their difference |
| F06_error_overlap | Shared errors between models against the seed noise floor |
| F07_seed_ensemble | Ensemble vs single-seed accuracy |
| F08_learning_curves | Validation curves ± SD, 4 representative tasks |
| F09_seed_stability | Seed SD and worst-seed shortfall, all tasks |
| F10_generalisation_gap | Train − test at the best checkpoint |
| F11_compute_cost | Seconds per epoch; accuracy vs training time |

Tables (`tables_all/`): `accuracy_by_task_model`, `contrasts_all_tasks`,
`contrasts_pooled_random_effects`, `contrast_vs_difficulty`, `per_class_recall_runs`,
`per_class_recall_summary`, `worst_class_and_collapse`, `error_overlap_runs`,
`error_overlap_summary`, `per_image_paired_test`, `seed_ensemble`,
`dynamics_overfit_cost_by_task_model`, `compute_cost_from_pbs_logs`.
