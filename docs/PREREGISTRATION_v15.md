# Pre-registration v15: IBM-calibrated noise on the quantum front-end, M1–M5

**Written:** 2026-10-17, before any v15 job was submitted and before any v15 accuracy existed.
**Follows:** v12 (`POST_CORE_RESULTS.md`), v13 (`V13_RESULTS.md`), v13.5 (`V13_5_RESULTS.md`), v14 (`V14_RESULTS.md`).
**Supervisor brief:** "V15 is noise analysis, so noisy input alpha from real IBM calibrated noise." M5 is included, as for every experiment from v12 to v15.

## 1. Question

On clean MNIST the trained quantum front-end never beats having no front-end: not across class count (v12), readout width (v13), training vs freezing (v13.5) or data size down to 50/class (v14, bounded at about +1 pp). Hardware noise is the last condition named in the thesis aims. It is also the condition where the front-end can only lose, since noise acts on the quantum front-end and on nothing else in M3/M4.

**Q-v15. When the quantum front-end runs under ibm_brisbane-calibrated noise, scaled by alpha, how much accuracy does it cost (with and without training under that noise), and does it change any front-end comparison (M3 vs M4, M3 vs M5)?**

## 2. What is noisy, and what is not

| | Front-end (noise acts here) | Head | Noise in v15 |
|---|---|---|---|
| M1 | Henderson RandomLayers, frozen (4 qubits, 1 CNOT) | QNN, w15 | front-end |
| M2 | Circuit 4, trained (two 2-qubit blocks, 2 CRZ each) | QNN, w15 | front-end |
| M3 | Circuit 4, trained | classical | front-end |
| M4 | none (bilinear resize) | classical | **none: identical at every alpha** |
| M5 | Circuit 4, frozen at seed init | classical | front-end |

**The QNN head (M1, M2) is noiseless in v15, by design.** The thesis question is the front-end. Keeping the head exact means M2 vs M3 still isolates the head, and every front-end contrast compares like with like. The head under noise is out of scope; if wanted later, it is a separate experiment.

All models at width 15, N = 500/class (the v13/v13.5/v14 protocol): CE, linear head, `--deterministic`, `--save_predictions`, batch 32, AdamW lr 1e-3, wd 1e-4, dropout 0, max 200 epochs, patience 20, warmup 30 (M1, M2, M3, M5). `--noise_alpha` stays 0.0 (that flag is the old back-end placeholder and is guarded). Front-end noise is set through `V15_ALPHA`.

## 3. The noise model (fixed now)

**Calibration:** `calib/ibm_brisbane_calib.json`, extracted from Qiskit's `FakeBrisbane` snapshot (ibm_brisbane, Eagle r3, 127 qubits, calibration of 2025-02-26; qiskit-ibm-runtime 0.50.0). sha256 `da9374fa89e6…`, checked by every job. Device medians: T1 232 µs, T2 150 µs, SX error 2.43e-4 (60 ns), ECR error 7.68e-3 (660 ns), readout error 2.00%. One faulty edge (error 1.0) is excluded from the medians.

**Channels (the ones Qiskit Aer's `NoiseModel.from_backend` builds).** Every logical gate is charged the native IBM gates it compiles to: RY, RX → 2 SX; RZ → virtual (no error); CRZ → 2 CX → 2 ECR; CNOT → 1 ECR. Each native gate: a depolarizing channel, then thermal relaxation (T1, T2, gate duration) on its qubits, with the depolarizing parameter set so the total average gate infidelity equals the reported gate error (Aer's own formula). Measurement: the per-qubit `readout_error`, applied to ⟨Z⟩ (`from_backend` uses the same symmetric value for IBM's BackendV2 devices). Expectation values are exact, as in every run since v5 (no shot noise).

**Scaling by alpha.** alpha multiplies every error rate: gate errors, readout error, and the relaxation rates (T1 → T1/alpha, T2 → T2/alpha). alpha = 1 is the calibrated device; alpha = 0 is the noiseless simulator, and there the original forward pass runs (bit-exact with v13/v13.5/v14).

**Verification (`verify_v15.py`, before any job).** (a) Our native channels equal `NoiseModel.from_backend(FakeBrisbane)`'s SX, ECR and readout errors on the same qubits to 1e-15. (b) Circuit 4 under noise equals Qiskit Aer density-matrix simulation at alpha = 0.5, 1, 2, 4, 16 (≤ 5e-8; the residual is Aer dropping mixture terms below 1e-8). (c) A random 4-qubit RX/RY/RZ/CNOT circuit (M1's gate set) equals Aer (≤ 1e-8). (d) Against a full simulation of the block transpiled to ibm_brisbane's native gates on a median-quality edge, our model's noise shift correlates 0.987 with the device's and is about **9% smaller** (the device's extra single-qubit gates and the placement of T1 decay). So alpha = 1 in this model corresponds to roughly alpha ≈ 1.1 on that device. Stated, not corrected.

**Size of the perturbation (front-end output, mean |Δ⟨Z⟩|, random inputs):** Circuit 4: 0.019 / 0.037 / 0.073 / 0.142 at alpha = 0.5 / 1 / 2 / 4 (0.26 and 0.45 at 8 and 16). M1: 0.013 / 0.026 / 0.052 / 0.103. The outputs lie in [−1, 1]. At alpha = 1 the noise is mostly a contraction of ⟨Z⟩ towards 0 (readout and depolarizing), plus a small T1 bias towards +1.

**Omissions (stated limits):** device-median calibration (no qubit mapping), no routing SWAPs (M1's CNOT is between non-neighbouring qubits; charged one ECR, which is optimistic for M1), no crosstalk, no idle errors, exact expectations (no shot noise). These are also the omissions of an unscheduled Aer `from_backend` simulation.

## 4. Design

| K | Subsets (v12, seed 20260928) | alpha (training = test) | Models | Seeds | Jobs |
|---|---|---|---|---|---|
| 5 | A {1,3,5,8,9}, B {0,2,6,8,9} | 0, 0.5, 1, 2, 4 | M1, M2, M3, M5 | 42–51 | 400 |
| 5 | A, B | 0 | M4 | 42–51 | 20 |
| 10 | all digits | 0, 0.5, 1, 2, 4 | M1, M2, M3, M5 | 42–51 | 200 |
| 10 | all digits | 0 | M4 | 42–51 | 10 |
| | | | | **total** | **630** |

**Cross-evaluation (no extra jobs).** At the end of every job, the trained model is evaluated on the test set at every alpha in {0, 0.5, 1, 2, 4, 8, 16}. Before that, it is evaluated at its own training alpha and must reproduce the per-image predictions `train_gadi.py` saved, exactly. Only a MATCH is used.

This gives two arms from one grid:
- **Matched:** trained and tested at alpha (the job's own test accuracy). Noise-aware training: the front-end and everything after it learn under the noise they are tested with.
- **Deployed:** trained at alpha = 0, tested at alpha (the alpha = 0 job's cross-evaluation). A clean-trained model run on noisy hardware. Includes alpha = 8, 16 as stress levels.

**The alpha = 0 jobs are reproductions.** Each must equal the stored run exactly: M1, M4 from v14 (N = 500, w15); M2, M3 from v13 (w15); M5 from v13.5 (w15). That is what licenses pooling, and proves the noise code is invisible at alpha = 0.

**One smoke test** (63 jobs, one submission, one gate): seed 42 at every (K, subset, alpha, model) cell, kept and reused by the grid. Pre-registered reproduction gate: K = 5 A, alpha = 0, seed 42, all five models, exact (test_acc, best/stop epoch, every prediction). The other alpha = 0 smoke cells are compared and reported. The grid is the other 567 jobs.

K = 10 may be dropped **only before the smoke is submitted**, on SU grounds alone, logged as an amendment.

## 5. Analysis

Paired over seeds. K = 5 primary is pooled over subsets (per-seed mean of A and B, df = 9), with A and B reported alongside. **Every null is reported with its MDE** (80% power, α = .05). Equivalence only via TOST, margin **±1 pp**.

Contrasts (each at every alpha, both arms): **F** = M3 − M4, **T** = M3 − M5, **R** = M5 − M4, **BE** = M3 − M2, **Q** = M2 − M1.
Noise costs: **NC_m(alpha)** = m(alpha) − m(0), matched arm; **DEP_m(alpha)** = m deployed at alpha − m(0); **REC_m(alpha)** = matched − deployed at alpha (what noise-aware training recovers).
**slope(X)** = per-seed OLS slope of X on alpha over alpha = 0, 0.5, 1, 2, 4 (pp per unit alpha), one-sample t over seeds.

**Primary, Holm within each K (4 tests):**

| ID | Contrast | Question |
|---|---|---|
| NC3@1 | M3(1) − M3(0), + TOST | Does real-device noise cost the trained quantum front-end? |
| F@1 | M3(1) − M4, + TOST | At device noise, does the trained quantum front-end beat no front-end? |
| T@1 | M3(1) − M5(1), + TOST | At device noise, does training the front-end matter? |
| slope(F) | trend of M3 − M4 with alpha | Does noise erode (or reveal) a front-end effect? |

**Secondary (uncorrected, with MDE):** F, T, R, BE, Q, NC_m at every alpha; both arms; DEP_m and REC_m at every alpha; slope(T), slope(R), slope(BE), slope(NC_m); per-subset results; seed SD vs alpha; runs at the cap.
**Exploratory (labelled as such):** whether seed 46 remains the worst M5 (v13.5 R3, v14 L5); anything found after reading results.

## 6. Predictions (written before any v15 result)

- **P1.** NC3@1 is equivalent to 0 within ±1 pp (TOST) at K = 5 and K = 10, and NC_m is not detected for any model at alpha ≤ 2. Basis: at alpha = 1 the front-end output moves by ~0.04 on a [−1, 1] scale, mostly a smooth contraction that the first convolution can rescale when trained under it.
- **P2.** F@1 is not detected and its point estimate is ≤ 0 (M3 has not beaten M4 in any v12–v14 cell). slope(F) has a point estimate ≤ 0: noise can only cost the quantum front-end, and M4 is unaffected. Detection of slope(F) is not predicted.
- **P3.** T ≥ 0 (point estimate) at every alpha (as in v13.5 13/13 and v14 8/8 cells).
- **P4.** Deployed arm: DEP_m ≤ 0 for every model with a quantum front-end, more negative as alpha grows, and REC_m ≥ 0 (noise-aware training recovers at least part of it). No prediction on size at alpha = 1.
- **P5 (exploratory).** slope(BE) is not detected (the head is noiseless, and M2 and M3 share the same front-end and the same noise). In the deployed arm M1 degrades less than the Circuit-4 models, because its front-end is perturbed less (0.026 vs 0.037 at alpha = 1).

## 7. How each outcome will be read

| Outcome | Reading |
|---|---|
| NC3 ≈ 0 (TOST) at alpha = 1, F not detected | Device-level noise costs nothing measurable once the model trains under it, and still no front-end advantage. Noise does not change the v12–v14 conclusion. |
| NC3 detected < 0 at alpha = 1 | Current-hardware noise costs the quantum front-end accuracy even with noise-aware training. Report size with MDE and the per-seed plot; compare with REC. |
| F detected < 0 at some alpha, slope(F) < 0 | Under noise the quantum front-end is worse than no front-end, with a dose-response. The strongest negative result available for the front-end. |
| F detected > 0 at some alpha | Would contradict P2. Check T and R first (a frozen random front-end doing the same means regularisation by fixed features, not training), then per-subset consistency. Worded as "beats no front-end", never "quantum advantage" (no matched classical front-end, as v14). |
| DEP ≪ 0 while NC ≈ 0 | Noise is harmful only to models trained without it: a deployment message (train under the device's noise), not an architecture one. |
| T grows with alpha | Training the front-end matters more under noise (the trainable circuit adapts to the noise). If T shrinks, training buys less once noise dominates. |

## 8. Run order, gates and stop rules

1. `setup_v15.sh`: build `qcq_cnn_v15` from `qcq_cnn_v14`, patch, verify. Must end in `ALL CHECKS PASSED`.
2. **Smoke** (`submit_v15_smoke.sh`): 63 jobs, one submission.
3. **Gate** (`check_v15.sh smoke`): per log rc = 0, loss = ce, n_classes, class list, n_obs = 15, params VERIFIED, determinism, no continuation, max_epochs = 200, warmup, M5 freeze banner + exit self-check, split audit (train 500·K, val 125·K, test 200·K, zero overlap); `[V15] NOISE installed` with the right model, front-end kind, alpha and calibration hash; install self-checks PASS; cross-evaluation self-check MATCH; sidecar complete and consistent with the result JSON. Across logs: every model and every alpha at a (K, subset, seed) saw identical images; one test set per (K, subset). The K = 5 A reproductions exact.
4. **Grid** (`submit_v15_grid.sh`, refuses without the smoke gate), then **gate again** (`check_v15.sh grid`) before any accuracy is read.

**Stop rules (fixed now):**
- A K = 5 A reproduction fails → stop. Nothing runs until the cause is found.
- Any install self-check fails or any cross-evaluation self-check is not MATCH in the smoke → stop.
- Any smoke run stops at the 200-epoch cap → stop before the grid; revisited as a dated amendment.
- Any continuation (time budget hit) in the smoke → stop; walltime revisited as a resource amendment.
- In the grid: cap hits are reported per cell; a cell with > 2/9 grid runs at the cap is labelled budget-limited.

## 9. What would NOT change this plan

A low or high headline number. No grid is re-run to improve a result. The calibration file is fixed by its hash. Any change after the first smoke result is read is logged below as a dated amendment with its reason.

## Amendments

**A1 (2026-10-17, before the smoke; resource only, no result read).** gdata had about 10.3K inodes free out of 1,018,000, shared across wa66. `run15.pbs` now deletes a job's checkpoint folder at the end of the job, and only when that job's cross-evaluation self-check is MATCH. Checkpoints are only a fallback inside the job's own cross-evaluation; the gate, collect and plots never read them. No model, data, training or analysis change. Logged per job as `[V15] checkpoint dir removed after a MATCH cross-eval`.

**SU decision (2026-10-17, before the smoke).** 19.79 KSU available. All of q4 so far (v13, v13.5, v14, about 780 jobs) used 5.21 KSU, roughly 6.7 SU per job. 630 v15 jobs are about 4–6 KSU, allowing for some noise overhead. **K = 10 is kept.**
