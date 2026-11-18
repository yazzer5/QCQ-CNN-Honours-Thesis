# Pre-registration v16c: diagnostics (no training, no Gadi)

**Written:** 2026-10-21, 21:30 AEDT, **before** any v16c input file was opened or any quantity computed.
At the time of writing, the v16a z-dumps, the code package and the M5 per-seed CSVs had **not** been received.
**Status: exploratory.** These analyses support or undermine the head-frequency mechanism and explain seed 46. They do not replace or modify any pre-registered primary of v13–v16b.

**What is already known (so the predictions are not blind on these points):**
- Seed 46's frozen front-end is the worst M5 in every cell of v13.5 (13), v14 (9) and v15 (15). C1 is a prediction about *conditioning*, which has never been computed. C2's accuracy side is already known to have one extreme seed, so leave-46-out is reported for every C2 number.
- v16a: the QNN head equals 12 closed-form trig features of z = fc2's output + Linear. The ZZ phase contains 2(π − z0)(π − z1).
- z-dumps exist only for seeds 42–44. **Seed 46 cannot be linked to head frequency in v16c.**

## Part A: front-end conditioning (explains seed 46)

**A0. Rebuild and cross-check (gate).** For seeds 42–51, build the Circuit-4 front-end through the code's own init path (same call order and RNG as training). Compare with the stored M5 front-end parameters for the same seed (M5 is frozen, so stored = init). Gate: **exact equality** (max |Δ| = 0, or bitwise equal float32). The rebuilt circuit's forward pass on 100 stored patches must also equal the code's own front-end forward to ≤ 1e-6.
**Stop rule:** any seed fails → Part A stops; nothing in Part A is reported until the cause is found and logged as a dated amendment. *(Replaced by A4.1: no stored parameters exist.)*

**A1. Patches.** 10,000 2×2 patches, drawn once (numpy RNG seed 0) uniformly over (training image, patch position) from the K = 10 training set, after the code's own preprocessing (14×14 pooled MNIST; *corrected by A4.2*). Positions and stride exactly as the front-end applies them (to be read from the code). **The same 10,000 patches are used for every seed.** All patches are kept, including all-background ones, because that is the distribution the front-end sees.

**A2. Jacobian.** Per patch, J = ∂⟨Z_i⟩/∂θ_j, i, j = 1..4: the 4 Z expectation outputs w.r.t. the 4 encoding angles θ ∈ [0, π] (θ = πx for pixel x ∈ [0, 1]). Computed by autograd through the code's own front-end, cross-checked against central finite differences (h = 1e-4) on 100 patches, agreement ≤ 1e-5. If the code's input to the circuit is not the angle in [0, π], the derivative is still taken w.r.t. the angle, and this is logged as an amendment before computing.

**A3. Per-seed metrics.**
- **Primary conditioning score: median σ_min over patches.** Absolute σ_min is meaningful because every seed maps to the same fixed output range [−1, 1]: it measures how much pixel information the worst direction loses.
- Secondary: median σ_min/σ_max; fraction of patches with σ_min < 0.05; per-channel variance of the 4 Z outputs over the patches (and their minimum across channels).
- Sensitivity (descriptive): all metrics recomputed on non-background patches only (any pixel > 0).

**C1.** Seed 46 has the lowest median σ_min of the 10 seeds.
Verdict: **holds** if lowest on the primary. **Partial** if lowest on ≥ 2 of the 3 secondary conditioning metrics (σ_min/σ_max, fraction < 0.05, min channel variance) but not the primary. **Fails** otherwise. Chance level for "lowest of 10" is 0.10.

**C2.** Conditioning correlates positively with M5 accuracy across seeds at K = 10.
- Accuracy score per seed: mean within-cell z-score of M5 test accuracy over the **unique** K = 10 M5 cells of v13.5 (w3, w6, w15 at N = 500), v14 (N = 50, 100, 250) and v15 (matched arm, α > 0). Cells that are bit-identical reproductions of another cell (v14 N = 500, v15 α = 0) are dropped so no run counts twice.
- Test: Spearman ρ(median σ_min, accuracy score), n = 10, one-sided (ρ > 0).
- Verdict: **holds** if ρ > 0 with one-sided p < .05; **direction only** if ρ > 0 and p ≥ .05; **fails** if ρ ≤ 0.
- Reported alongside, not substituted: leave-46-out ρ (n = 9); ρ for each secondary metric; per-cell ρ and the count of cells with ρ > 0.
- Interpretation fixed now: if C2 holds only with seed 46 included, the reading is "one bad initialisation is badly conditioned", not "conditioning predicts accuracy in general".

## Part B: head frequency (direct evidence for the mechanism)

**B0. Inputs and checks.** v16a dumps `results_v16a_z_k10_all_n{50,500}/v16_z_{M2,M3}_seed{42,43,44}.npz` (12 runs, all bit-exact reproductions of v13/v14). Labels and predictions are taken from the npz; if absent, from the matching result JSON (`test_predictions`). Gate: per run, predictions from the npz (if present) equal the JSON's; argmax accuracy equals the JSON `test_acc`; the test set is the same images (identical label vectors) across all 12 runs (see A1 below: 2,000 images). **Stop rule:** a run failing these is excluded and named; no substitution.

**B1. Definitions.** z = (z0, z1), fc2's raw test-set output, which feeds ZZFeatureMap(2, reps = 1) directly (to be confirmed in the code; if a scaling is applied first, f is computed on the scaled input and this is logged as an amendment before computing).
- f0 = 2|π − z1|, f1 = 2|π − z0| (the product term's local frequency in each coordinate); **f = √(f0² + f1²)** per test image.
- f covers the product term only; the single-qubit phases 2z_i add a constant frequency of 2 per coordinate and are ignored.
- For M3, z feeds no ZZ map; "pseudo-f" from M3's z is a reference for the natural scale of z only.

**H1.** M2's median f is higher at N = 50 than at N = 500.
Test: per seed, median f at N = 50 vs N = 500 (same seed, same test images). Verdict: **holds** if higher in 3/3 seeds (exact one-sided sign test p = .125: the most n = 3 can give, so H1 is directional evidence, not a significance claim). Reported: per-seed medians and ratios; image-level paired Wilcoxon per seed (descriptive); M3's pseudo-f N50/N500 ratio as reference.

**H2.** Within M2, misclassified images have higher f than correctly classified ones.
Test: one-sided Mann–Whitney U (errors > correct) per (N, seed), 6 tests. Combined within each N over the 3 seeds by unweighted Stouffer Z. Effect size: AUC = P(f_error > f_correct).
Verdict: **holds** if the combined p < .05 at both N; **partial** at one N; **fails** at neither.
**Specificity control (reference, not a prediction):** the same test with M3's pseudo-f and M3's own errors. Reading fixed now: if M3 shows an association of similar size (AUC within 0.03 of M2's), H2 does not single out the ZZ map; errors could just be atypical images landing far from the training clusters in z.

**H3.** M2's misclassification rate rises across quartiles of f.
Quartiles of f within each run. Test: Cochran–Armitage trend test per run, combined within N by Stouffer. Verdict: **holds** if Q4 error > Q1 error in all 6 runs and combined trend p < .05 at both N; **partial** if the combined trend is significant at one N or Q4 > Q1 in 4–5 runs; **fails** otherwise. Full monotonicity (Q1 < Q2 < Q3 < Q4) reported per run, descriptively. M3's error by pseudo-f quartile shown as reference.

## Multiplicity and reporting

Five predictions, all exploratory: no Holm. Raw p-values reported with the number of tests. Every result in V16C_RESULTS.md is labelled exploratory. No prediction is reworded after results are read; changes before computing are logged below as dated amendments.

## Outputs

`figures_v16c/` (200 dpi) + generating script `v16c_diagnostics.py`:
1. per-seed conditioning bar chart, seed 46 highlighted;
2. conditioning vs M5 accuracy score scatter (ρ with and without 46);
3. f histograms: M2 vs M3 (pseudo-f), N = 50 vs N = 500;
4. M2 error rate by f quartile (M3 reference).
`V16C_RESULTS.md` with verdicts on C1, C2, H1, H2, H3, a plain-language summary and a supervisor paragraph.

## Amendments

**A1 (2026-10-21 21:20 AEDT, after inspecting file structure only: npz keys/shapes and JSON key names; no z value, f, accuracy or error rate read).** The v16a test set is **2,000 images** per run (`n_test = 2000`), not 10,000. B0's gate therefore requires identical 2,000-label vectors across the 12 runs. Nothing else changes.

**A2 (same time, same status).** The code package was not in the archive: the PBS logs only list `train_gadi.py`, `qnn_torch.py`, `quanv_torch.py`, `closed_form_zz_v16.py`, `v16_tools.py`. **Part B proceeds before the code arrives**, with f computed on z as stored (fc2's raw output). v16a's verify already established that fc2 is the QNN head's input and that the head equals ZZFeatureMap(2, reps = 1) + RealAmplitudes. If `qnn_torch.py` shows any transform of z before the feature map, f is recomputed mechanically with that transform and **both versions are reported**; no other analysis choice changes.

**A3 (same time).** Part A is blocked: the code files above and the stored M5 front-end parameters are needed, and no M5 `.pt` checkpoints were found on Gadi under v13.5. If no stored parameters exist anywhere, the A0 gate cannot be run as written; any replacement gate is logged here before any conditioning number is computed.

**Part B was computed on 2026-10-21 after A1–A3** (all three predictions failed; see V16C_RESULTS.md). One post-hoc quantity was added and is labelled post hoc wherever it appears: the number of ZZ product-phase wraps across the middle 90% of each run's test images.

**A2 resolved (2026-10-22 12:50 AEDT).** `qnn_torch.py` `QNNBackendTorch.forward` takes `x0, x1 = x[:, 0], x[:, 1]` (fc2's output, unscaled) and applies `P(2*(pi-x0)*(pi-x1))` on q1. f was computed on the right quantity; Part B is not recomputed.

**A4 (2026-10-22 12:50 AEDT, after reading the code and the runs-CSV cell lists; no conditioning quantity computed).**
1. **A0 gate replaced: no stored front-end parameters exist.** No M5/M3 checkpoint survives (v15 deleted checkpoints; the v13.5 search found none). Result JSONs record only front-end gradient statistics. The M5 patch snapshots the init in memory only (`[V135] M5 front-end after run: UNCHANGED OK`). Init in the code: `TrainableQuanvFrontEndTorch(seed)` draws `torch.rand(6, generator=torch.Generator().manual_seed(seed)) * 2π`, a private generator independent of the global RNG and of build order. Replacement gate, all three parts required:
   - **G-a (code path, here):** for each seed, `train_gadi.build_model("M5", seed).front_end.weights` equals `TrainableQuanvFrontEndTorch(seed).weights` bitwise, has `requires_grad=False`, and is bitwise identical when built under two different global seeds (0 and 999).
   - **G-b (training environment, Gadi):** the same 6 weights printed with `float.hex()` from Gadi's torch must equal the local ones bitwise. **Until G-b passes, every Part A number is labelled provisional.** If G-b fails, Part A is recomputed with Gadi's weights and the provisional numbers are discarded, not reported.
   - **G-c (forward):** the analysis forward equals the code's `front_end.forward` to ≤ 1e-6 on the 100 cross-check patches (as A0).
   This is weaker than the pre-registered gate: it proves the analysis uses the weights the code generates, not weights read back from a finished run.
2. **Patch definition corrected.** Images enter the front-end at **28×28 in [0, 1]**. The front-end cuts 196 non-overlapping 2×2 patches (stride 2) and outputs 14×14; "14×14 pooled MNIST" in A1 described the output. Patches are drawn from the K = 10 training split returned by the code's `load_dataset` at seed 42, N = 500 per class (5,000 images; N_VAL 1,250, N_TEST 2,000 as in the v16a protocol). One split serves every seed, as A1 requires.
3. **Structure (stated, no analysis change).** The pixel pairs (top row) and (bottom row) feed two independent 2-qubit blocks with the same 6 weights. Output order is [Z_a^top, Z_b^top, Z_a^bot, Z_b^bot], so J is block-diagonal: σ(J) is the union of the two 2×2 blocks' singular values, and σ_min(J) is the smaller block's σ_min. w5 is structurally dead (final CRZ before Z measurement).
4. **C2 cells fixed from the CSVs:** v13.5 K = 10 w3, w6, w15; v14 N = 50, 100, 250; v15 matched α_train = 0.5, 1, 2, 4 (`test_acc`). 10 unique cells × 10 seeds. Dropped as duplicates: v14 N = 500 (source v13.5 = w15) and v15 α = 0. Check: v15 `test_acc` must equal `acc_eval_<α_train>`; otherwise matched accuracy is read from that column.
