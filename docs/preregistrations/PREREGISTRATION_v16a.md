# Pre-registration v16a: classical replica of the w15 QNN head (M2c) + z-dumps

**Written:** 2026-10-20, before any v16a job was submitted and before any v16a accuracy existed.
**Part of:** `V16_PLAN.md` (Part 1). **Follows:** v13 (`V13_RESULTS.md`), v14 (`V14_RESULTS.md`), v15 (`V15_RESULTS.md`).
**Protocol copied from:** `PREREGISTRATION_v14.md`.

## 1. Question

At full readout (w15) the QNN head is, algebraically, 12 fixed trig features of fc2's two outputs z plus a linear layer; its 6 circuit parameters are redundant (v13 §2). It costs 2.3 pp against the classical head at N = 500 and 13.5 pp (K = 5) / 17.1 pp (K = 10) at N = 50, almost all of it a generalisation gap (v13 V3, v14 L2–L3).

**Q-v16a. Does a model that computes the same 12 features in closed form, with no circuit at all, reproduce M2's accuracy, its errors and its low-data collapse?** If it does, the head's cost belongs to the classical feature map, and nothing quantum contributes to it.

## 2. Model M2c

**Definition.** M2 with the QNN module and the Linear(15 → K) that reads it replaced by `ZZTrig12` (12 closed-form features, 0 parameters) and a fresh Linear(12 → K). Same trained Circuit-4 front-end, backbone and fc2 as M2.

**Features, derived from the code's circuit** (`qnn_torch.QNNBackendTorch` = Qiskit ZZFeatureMap(2, **reps = 1**) + RealAmplitudes(2, reps = 2), little-endian, phase gates P(2z₀), P(2z₁) and P(2(π−z₀)(π−z₁)) between two CX(0→1)). After the feature map the state is ½ Σ_b e^{iφ_b}|b⟩ with ζ = 2(π−z₀)(π−z₁):

| basis | φ |
|---|---|
| \|00⟩ | 0 |
| \|01⟩ (b₀ = 1) | 2z₀ + ζ |
| \|10⟩ (b₁ = 1) | 2z₁ + ζ |
| \|11⟩ | 2z₀ + 2z₁ |

RealAmplitudes is a real orthogonal U(θ), so for any observable O, ⟨O⟩ = Tr O/4 + ½ Σ_{j<k} [Re M_jk cos d_jk − Im M_jk sin d_jk], with M = UᵀOU and d_jk = φ_k − φ_j. The 12 features are cos and sin of the six phase differences

d₀₁ = 2z₀ + ζ, d₀₂ = 2z₁ + ζ, d₀₃ = 2z₀ + 2z₁, d₁₂ = 2z₁ − 2z₀, d₁₃ = 2z₁ − ζ, d₂₃ = 2z₀ − ζ.

Code: `closed_form_zz_v16.py` (shared with v16b as its "twin").

**Pre-run checks** (`verify_v16a.py`, must pass on Gadi before any job):
- The real module's 15 outputs equal an independent statevector of ZZFeatureMap(reps = 1) + RealAmplitudes(reps = 2), column by column (≤ 1e-6). reps = 2 is shown *not* to match.
- **Span check (as specified in V16_PLAN.md):** every one of the 15 Pauli expectations is a linear combination of the 12 features plus a constant, residual ≤ 1e-6, over random z (|z| up to 12) and random RealAmplitudes parameters (6 draws plus the seed-42 initialisation).
- The converse also holds: each feature is a linear combination of the 15 Paulis. So for generic θ the two heads have **the same feature span**, and M2c can represent every function M2 can (checked end-to-end: M2c with its Linear set to W_fc3·C reproduces M2's logits).

**Parameter counts** (verified in every job log):

| K | M2 (w15) | M2c | M3 |
|---|---|---|---|
| 5 | 1686 | **1665** | 1725 |
| 10 | 1766 | **1730** | 1805 |

M2c = M2 − 6 circuit parameters − Linear(15→K) + Linear(12→K).

**Initialisation and pairing.** M2 is built at the seed exactly as before. The Linear(12→K) is then created with PyTorch's default initialisation under `torch.manual_seed(seed)`, inside a forked RNG. Every global RNG stream is left exactly where building M2 leaves it. So at the same seed M2c has M2's front-end, backbone and fc2 initialisation, M2's data split and M2's batch order; only the head differs. Verify proves the RNG equality and the identical weights.

**What M2c and M2 still differ in (stated now, so the reading can't drift).** The two heads have the same function class. They differ only in parameterisation: M2's 15 → K map over a rank-12 feature set, with the 6 circuit parameters rotating that map, against M2c's direct 12 → K map. AdamW weight decay acts on different coordinates in the two. Any detected difference is therefore an optimisation or implicit-regularisation effect of the circuit parameterisation, not a difference in what the heads can represent.

## 3. Protocol (unchanged from v14)

w15, CE, linear head, `--deterministic`, `--save_predictions`, AdamW lr 1e-3 wd 1e-4, batch 32, dropout 0, noise off (V15_ALPHA unset), val 125/class, test 200/class (fixed), patience 20, 30-epoch warmup (M2c has M2's Circuit-4 front-end), max-epoch cap 200 × 500 / N (2000 at N = 50, 200 at N = 500). Data split from the unchanged `load_dataset`, so every model at a (K, subset, N, seed) sees identical images (the gate checks this).

**Invocation.** `--model M2 --v16_head m2c`. Every M2 code path (warmup, logging, training loop) applies unchanged. The patch (`patch_v16a.py`) inserts one block that wraps `build_model`, `expected_params` and `load_dataset` by name and edits no existing line. Without a v16 flag it is a pass-through: verify proves M1–M5 bit-identical to the pre-patch file.

**`--save_z`.** At exit, after the result JSON is written and with every RNG state restored, the trained model is run on the test set. Its predictions must reproduce the saved ones (self-check MATCH). The output of fc2 (the unique Linear(64 → 2), verified to be the QNN head's input) is written to `v16_z_<MODEL>_seed<S>.npz` with labels and predictions. Every v16a job also writes `v16_run_<MODEL>_seed<S>.json` (labels, self-check, parameter count).

## 4. Design

| Arm | Model | K | Subsets | N per class | Seeds | Jobs |
|---|---|---|---|---|---|---|
| c | M2c | 5 | A {1,3,5,8,9}, B {0,2,6,8,9} | 50, 500 | 42–51 | 40 |
| c | M2c | 10 | all digits | 50, 500 | 42–51 | 20 |
| z | M2, M3 with `--save_z` | 10 | all digits | 50, 500 | 42–44 | 12 |
| r | M2 (reproduction gate) | 5 | A | 500 | 42 | 1 |
| | | | | | **total** | **73** |

Comparators are **read, not re-run**: M2-w15 and M3-w15 from v13 at N = 500 (`results_v13_k{K}_{TAG}_w15`) and from v14 at N = 50 (`results_v14_k{K}_{TAG}_n50`).

**Smoke (11 jobs, one submission, one gate):** seed 42 at each of the 6 M2c cells, the 4 z cells, and the reproduction job. **Grid:** the other 62.

**Reproductions (gated, smoke and grid):** all 12 z-jobs and the reproduction job must equal the stored v13 (N = 500) / v14 (N = 50) runs **exactly**: test_acc, best epoch, stop epoch and every per-image prediction.

## 5. Analysis (`collect_v16a.py`)

Paired over seeds 42–51. K = 5 is pooled over subsets (per-seed mean of A and B, df = 9), with A and B reported alongside. Every null is reported with its MDE (80% power, α = .05).

**Primary.** D = M2c − M2(w15) at each (K, N), with 95% CI and a TOST at **±1 pp**. Holm within each K over the two N cells, applied to the TOST p-values (equivalence) and separately to the two-sided difference tests.

**Secondary (uncorrected):**
- BE = M3 − M2 and BEc = M3 − M2c at each (K, N), and the collapse BEc(50) − BEc(500).
- Per-image error agreement: Jaccard of error sets and Cohen's κ on correct/incorrect. Compared same-seed M2c–M2, cross-seed M2–M2, and cross-seed M2c–M2.
- Per-subset D.

**Not analysed in v16a:** the z-dumps (v16c, diagnostics).

## 6. Predictions (written before any v16a result)

- **P1.** D is equivalent to 0 within ±1 pp (Holm-adjusted TOST p < .05) in **all 4** (K, N) cells.
- **P2.** In every (K, N) cell, the mean same-seed error agreement J(M2c_s, M2_s) is at least the mean cross-seed agreement between two M2 seeds, J(M2_s, M2_t), s ≠ t.
- **P3.** M3 − M2c reproduces M3 − M2, including the low-data collapse. Operationally, both must hold at both K:
  - BEc at N = 50 is detected > 0 (p < .05);
  - BEc(50) − BEc(500) is detected > 0 (p < .05).

## 7. How each outcome will be read (from V16_PLAN.md)

| Outcome | Reading |
|---|---|
| P1 holds | The head's cost belongs to the classical feature map; nothing quantum contributes. |
| P1 fails with D detected > 0 (M2c better) | The circuit parameterisation adds optimisation difficulty on top of the feature map. |
| D detected < 0 (M2c worse) | **Check the implementation first** (feature formulas, surgery, initialisation) before any reading. |
| P1 fails, nothing detected | Underpowered at n = 10 in that cell; report CI and MDE. No equivalence claim there. |
| P2 holds | The two heads fail on the same images: same mechanism, not just the same average. |
| P3 holds | The low-data collapse is a property of the 12-feature map itself. |

## 8. Run order, gates and stop rules

1. `setup_v16a.sh`: build `qcq_cnn_v16a` from `qcq_cnn_v15` (code only), patch, verify. Must end in `ALL CHECKS PASSED`.
2. **Smoke** (`submit_v16a_smoke.sh`, 11 jobs). It refuses without the verify log, this file in the repo, and every stored comparator file present.
3. **Gate** (`check_v16a.sh smoke`):
   - per run: rc = 0, no continuation, loss = ce, n_classes, class list, n_obs = 15, trainer parameter check VERIFIED OK, determinism, max_epochs = cap(N), warmup 30, noise off;
   - M2c head line VERIFIED with the §2 counts;
   - RUN self-check MATCH;
   - SAVEZ MATCH (z jobs);
   - split audit (train N·K, val 125·K, test 200·K, all classes, zero overlap);
   - same images within (K, subset, N, seed); one test set per (K, subset); N = 50 splits equal to the stored v14 M2 logs where found;
   - reproductions exact.
4. **Grid** (`submit_v16a_grid.sh`, refuses without the smoke gate), then **the gate again** (`check_v16a.sh grid`) **before any accuracy is read**. The smoke gate prints epochs and times only, no accuracies.

**Stop rules (fixed now):**
- Any reproduction fails → stop. Nothing is read until the cause is found.
- Any smoke run stops at its epoch cap → stop before the grid; the cap is revisited as a dated amendment.
- Any continuation (time budget hit) in the smoke → stop; walltime is revisited as a resource amendment.
- In the grid, cap hits are reported per cell; a cell with > 2/10 is labelled budget-limited.

## 9. What would NOT change this plan

A high or low headline number. No grid is re-run to improve a result. Any change after the smoke is submitted is logged below, dated, **before** it is acted on.

## Amendments and defects

- **A0 (2026-10-20, before the smoke; resource only).** Walltime is 1 h for every cell, including N = 50 (v14 used 2 h at N = 50 with a 1.75 h budget). This is the user's decision for K = 10; the trainer's time budget is 0.85 h. A continuation in the smoke triggers the §8 stop rule and a dated resource amendment. Budget does not affect results unless it is hit.
- **A0b (2026-10-20, before the smoke; implementation note, not a design change).** M2c is invoked as `--model M2 --v16_head m2c` rather than as a new model name, so it inherits every M2 code path without editing `train_gadi.py`. Its result JSON is therefore named `result_M2_…` inside `results_v16a_c_*`. The run sidecar records `model = M2c`.

- **D1 (2026-10-21, before the smoke; defect, found by verify on Gadi).** The first setup run failed verify at the M2c dry run: `train_model()` (line 1061) reads `model.back_end.qnn.weight` for the v5 `qnn_weight_drift` print. M2c's head has no circuit, so this raised `AttributeError`. Everything else passed:
  - the real circuit is ZZFeatureMap reps = 1 (column match 1.8e-15);
  - span check residual 2.6e-14, converse 3.2e-14;
  - surgery, parameter counts 1665/1730, RNG and weight identity, function-class check 3.9e-7;
  - M1–M5 bit-identical;
  - `--save_z` invisible to training, self-check MATCH.

  **Fix (no design change):** the module in M2c's QNN slot gets a constant 6-element `weight` buffer. It is not a parameter, is non-persistent (not in the state dict or checkpoints) and is unused by `forward`. The diagnostic prints drift = 0 for M2c. Parameter counts, outputs, gradients and training are unchanged. Verify gained two checks: the buffer is not a parameter or in the state dict, and the drift line reads 0 in the M2c dry run. No job had been submitted.

*(Date every further amendment and keep it.)*
