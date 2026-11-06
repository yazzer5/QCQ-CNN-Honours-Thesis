# Pre-registration v14: low data (sample efficiency), M1–M5

**Written:** 2026-10-15, before any v14 job was submitted and before any v14 accuracy existed.
**Follows:** v12 (`POST_CORE_RESULTS.md`), v13 (`V13_RESULTS.md`), v13.5 (`V13_5_RESULTS.md`).
**Supervisor brief:** "V14 is low data, 50–500 training samples per class, and see if the models have sample efficiency." M5 is included, as for every experiment from v12 to v15.

## 1. Question

At 500 images per class, the trained quantum front-end is equivalent to no front-end (v12 F2: M3 ≈ M4 at K = 3–5), training it helps mainly by rescuing bad initialisations (v13.5 R1–R3), and the quantum head costs accuracy (v12 F1, v13). Data-rich, clean MNIST is the condition least likely to show a front-end benefit. Low data is one of the two remaining places (with noise, v15) where one could appear.

**Q-v14. As the training set shrinks from 500 to 50 images per class, does any front-end or head effect change? In particular, does the trained quantum front-end (M3) gain an advantage over no front-end (M4), and does that depend on training it (M3 vs M5)?**

## 2. Models (unchanged code), all at MAX readout width 15

| | Front-end | Head (width 15 = all 15 Paulis, 9 measurement settings) |
|---|---|---|
| M1 | Henderson RandomLayers, frozen | QNN |
| M2 | Circuit 4, trained | QNN |
| M3 | Circuit 4, trained | classical (2→15 map, rank 2 through fc2) |
| M4 | none (bilinear resize) | classical |
| M5 | Circuit 4, frozen at seed init (= M3 with the front-end frozen) | classical |

**Why width 15 (decided 2026-10-15, before any v14 job).** v13 showed the width-3 QNN head is the unstable component: wide seed spread, catastrophic seeds, 3× slower training (v13 V5), and a back-end cost that grows with K (V1). Low data would compound that instability and swamp any front-end effect. At w15 the QNN head is stable, fits as well as the classical head, and leaves a ~2.3 pp generalisation gap (V2, V3), so v14 tests the quantum head in its best form and front-end contrasts are sharper. Consequence, fixed now: at w15 the QNN head's 6 circuit parameters are redundant (v13 pre-registration §2), so the QNN head is a fixed trig feature map plus a linear layer; M1 vs M2 differ only in the front-end.

**Parameter counts at w15 (v13 table, re-verified by `verify_v14.py`):**

| K | M1 | M2 | M3 | M4 | M5 |
|---|---|---|---|---|---|
| 5 | 1680 | 1686 | 1725 | 1719 | 1725 |
| 10 | 1760 | 1766 | 1805 | 1799 | 1805 |

The spread exceeds the 1% budget, as accepted in v13: the extra classical-head parameters sit in a 2→15 map collapsed to rank 2 by fc2 (v13 P4: M3(w15) ≈ M3(w3), TOST at K = 5; v13.5 P3: same for M5).

**New at w15: M1 and M4 have never been run at width 15**, so v14 also runs them at N = 500.

**Not in v14: the matched 6-parameter classical front-end (M6, `V14_SCOPE.md`).** Consequence, fixed now: if M3 > M4 appears at low N, v14 can say "a trained quantum front-end beats no front-end", **not** "beats a classical front-end of the same size". Any such claim is worded accordingly.

## 3. What changes from v13/v13.5 (w15)

Width 15 for every model (§2). Then two things, both set by N:

1. **`--n_train = N × K`**, N ∈ {50, 100, 250}. Val stays 125/class and test 200/class.
2. **Max-epoch cap = 200 × 500 / N** (2000, 1000, 400; 200 at N = 500). At N = 50 an epoch is 1/10 of the gradient steps of an N = 500 epoch, so v12's 200-epoch cap would be a 10× tighter optimisation budget. Scaling the cap keeps the step budget at least v12's. **Early stopping (patience 20 epochs) and the 30-epoch warmup (M1, M2, M3, M5) are unchanged** and still decide when a run ends. `verify_v14.py` checks that no LR schedule depends on `max_epochs`, so the cap is only a cap. At N = 500 nothing changes, which the reproductions prove.

Everything else is v13/v13.5: CE, linear head, `--deterministic`, `--save_predictions`, batch 32, AdamW lr 1e-3, wd 1e-4, dropout 0, noise 0.

**Data split (unchanged `load_dataset`, recorded as protocol facts).** A balanced pool of (N + 125) images per class (the first ones in MNIST order) is permuted by the seed and cut into train (N·K) and val (125·K). Consequences, stated before results:
- Train class counts are a random draw around N, not exactly N. In the dry run on synthetic data at N = 50 they ranged ±15–20% around 50. The real spread is printed by `verify_v14.py` and in every job log.
- The val set changes with N. The test set is fixed (first 200/class of the MNIST test split).
- **All five models at the same (K, subset, N, seed) see identical train, val and test images** (proved per job by the split audit, below). Every contrast is paired over seeds, so the draw is common to both arms of every contrast.

**Code change (`patch_v14_lowdata.py`):** a wrapper around `load_dataset` that returns its result untouched and prints a split audit (per-class counts, order-independent image hashes, overlap counts). It saves and restores every RNG state. `verify_v14.py` proves (a) M1–M5 are bit-identical to the pre-patch file, (b) a dry run with the audit on and off gives identical predictions. Two reproduction jobs prove it at full scale.

## 4. Design

| K | Subsets (v12, seed 20260928) | N per class | Models | Seeds | Jobs |
|---|---|---|---|---|---|
| 5 | A {1,3,5,8,9}, B {0,2,6,8,9} | 50, 100, 250 | M1–M5 | 42–51 | 300 |
| 5 | A, B | 500 | M1, M4 | 42–51 | 40 |
| 10 | all digits | 50, 100, 250 | M1–M5 | 42–51 | 150 |
| 10 | all digits | 500 | M1, M4 | 42–51 | 20 |
| | | | | **total** | **510 + 2 reproductions** |

At N = 500, M2 and M3 are **read, not re-run**, from v13 (`results_v13_k{K}_{TAG}_w15`), and M5 from v13.5 (`results_v135_k{K}_{TAG}_w15`).

**One smoke test** (53 jobs, one submission, one gate): seed 42 at every (K, subset, N, model) cell (51, kept and reused by the grid) + 2 reproductions at N = 500, w15, K = 5 A, seed 42 in the patched repo: **M3 must equal v13 exactly and M5 must equal v13.5 exactly** (test_acc, best/stop epoch, every per-image prediction). The grid is the other 459 jobs.

K = 10 may be dropped **only before the smoke is submitted**, on SU grounds alone, logged as an amendment.

## 5. Analysis

Paired over seeds. K = 5 primary is pooled over subsets (per-seed mean of A and B, df = 9), with A and B reported alongside; a sign flip between subsets is reported as subset-dependent. **Every null is reported with its MDE** (80% power, α = 0.05). Equivalence only via TOST, margin **±1 pp**.

Contrasts: **F** = M3 − M4, **T** = M3 − M5, **R** = M5 − M4, **BE** = M3 − M2, **Q** = M2 − M1.
**slope(X)** = per-seed OLS slope of X on log₂ N over N = 50, 100, 250, 500 (pp per doubling of the data), then a one-sample t-test over seeds. A negative slope means the contrast grows as data shrinks.

**Primary, Holm within each K (4 tests):**

| ID | Contrast | Question |
|---|---|---|
| F@50 | M3 − M4 at N = 50, + TOST | Does the trained quantum front-end beat no front-end when data is scarcest? |
| T@50 | M3 − M5 at N = 50, + TOST | Does training the front-end matter when data is scarcest? |
| R@50 | M5 − M4 at N = 50, + TOST | Does a frozen random Circuit-4 front-end beat no front-end at N = 50? |
| slope(F) | trend of M3 − M4 with N | Does the front-end effect depend on data size? (the sample-efficiency question) |

**Secondary (uncorrected, with MDE):** F, T, R, BE, Q at every N; slope(T), slope(R), slope(BE), slope(Q); AULC differences for F, T, R (normalised area under accuracy vs log₂ N); per-subset results; seed SD vs N per model; best epoch vs N; runs at the cap.
**Exploratory (labelled as such):** whether seed 46 remains the worst M5 (v13.5 R3), and any sensitivity analysis excluding it.

## 6. Predictions (written before any v14 result)

- **P1.** F is equivalent to 0 within ±1 pp at N = 250 and 500 (as M3 ≈ M4 at 500 in v12; v13 showed M3 is width-insensitive). At N = 50 and 100, F is **not detected**, and slope(F) is **not detected**. Basis: the front-end is 6 parameters in front of a ~1.6k-parameter network, and at full data it behaves like a resize. This is the "no sample-efficiency advantage" prediction; a detected F > 0 at low N would refute it.
- **P2.** T ≥ 0 (point estimate) at every N, as in v13.5's 13/13 cells. No prediction on size or slope.
- **P3.** R ≤ 0 (point estimate) at every N, as in v13.5.
- **P4.** BE > 0 at every N (≈ 2.3 pp at N = 500, v13 C), and larger at N = 50 than at N = 500 (slope(BE) < 0). Basis: at w15 the remaining back-end cost is a generalisation gap (v13 V3), and over-fitting worsens with less data.
- **P5 (exploratory).** Seed SD rises as N falls for every model. At w15, M2's SD stays within 2× M3's at every N (v13 V5: the w3 instability is gone at w15).

## 7. How each outcome will be read

| Outcome | Reading |
|---|---|
| F detected > 0 at low N, slope(F) < 0 | The trained quantum front-end helps when data is scarce. **Not** a quantum advantage claim without M6: "a trained 6-parameter quantum front-end beats no front-end at low N". Check T: if T ≈ 0 too, a frozen random circuit does the same, and the benefit is fixed-feature regularisation. |
| F equivalent / not detected at every N, slope null | No sample-efficiency benefit from the quantum front-end down to 50 images per class. Bounded negative result (state the MDE). |
| F detected < 0 at low N | The front-end costs accuracy when data is scarce (more to fit, or poorer features). Report with per-seed plot and train/test fit. |
| T > 0 grows as N falls | Training the front-end matters more with less data. If it shrinks, the training benefit needs data, and frozen ≈ trained at low N. |
| BE grows as N falls | The quantum head's cost is worse with little data: a sample-*in*efficiency of the QNN head even at full readout, consistent with v13's generalisation-gap reading. |
| BE shrinks or closes at low N | With little data the classical head over-fits as much as the quantum one; the w15 gap is a data-rich phenomenon. |

## 8. Run order, gates and stop rules

1. `setup_v14.sh`: build repo from `qcq_cnn_v135`, patch, verify. Must end in `ALL CHECKS PASSED`.
2. **Smoke** (`submit_v14_smoke.sh`): 53 jobs, one submission.
3. **Gate** (`check_v14.sh smoke`): per log rc=0, loss=ce, n_classes, class list, n_obs = 15, params VERIFIED, determinism, no continuation, pre-registered max_epochs for N, warmup 30 (M1/M2/M3/M5) or 0 (M4), M5 freeze banner + exit self-check, split audit (train = N·K, val = 125·K, test = 200·K, all classes, zero overlap). Across logs: all models at a cell saw identical splits; test set identical everywhere; seeds draw different train sets. Both reproductions exact.
4. **Grid** (`submit_v14_grid.sh`, refuses without the smoke gate), then **gate again** (`check_v14.sh grid`) before any accuracy is read.

**Stop rules (fixed now):**
- A reproduction fails → stop. Nothing runs until the cause is found.
- Any smoke run stops at its max-epoch cap → stop before the grid; the cap is revisited as a dated amendment.
- Any continuation (time budget hit) in the smoke → stop; walltime is revisited as a resource amendment.
- In the grid, cap hits are reported per cell; a cell with > 2/10 is labelled budget-limited and its contrasts reported as such.

## 9. What would NOT change this plan

A low or high headline number. No grid is re-run to improve a result. Any change after the first smoke result is read is logged below as a dated amendment with its reason.

## Amendments

*(No amendments: nothing in the design changed after the first smoke result was read.)*

**Editorial correction (2026-10-19, after the results; no effect on any run).** §8 said "47 jobs" and listed "n_obs=3" among the gate checks. Both were left over from the width-3 draft written earlier on 2026-10-15, before the switch to width 15 (§2). The submitted package ran 53 smoke jobs, and the gate (`gate_v14.py`) checked n_obs = 15, as the job logs show. The text is corrected here; the original wording is kept in this note.
