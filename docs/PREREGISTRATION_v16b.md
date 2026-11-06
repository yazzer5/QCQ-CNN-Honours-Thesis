# Pre-registration v16b: Design 1, the BFR head, against M4 and M2-w15 (reused)

**Written:** 2026-10-21, before any v16b job was submitted and before any v16b accuracy existed.
**Part of:** `V16_PLAN.md` (Part 2, as cut by amendment A1). **Follows:** v14 (`V14_RESULTS.md`), v16a (`PREREGISTRATION_v16a.md`).
**Protocol and package format copied from:** `PREREGISTRATION_v16a.md`.

## 1. Question

At w15 the ZZ QNN head is 12 fixed trig features of fc2's 2-D output z plus a linear layer (v13 §2, v16a). The phase contains the product 2(π−z₀)(π−z₁) with z unbounded, so its local frequency grows with |π−z|. The head costs 2.3 pp against the classical head at N = 500 and 13.5 pp (K = 5) / 17.1 pp (K = 10) at N = 50, almost all of it generalisation (v14 L2, L3).

**Q-v16b. If the quantum head's encoding is bounded (u = s·tanh z), has no product term, and enters as a gated residual on top of the classical head, does the head's cost disappear?** Concretely: is BFR far better than M2-w15, and equivalent to (or better than) M4?

## 2. Model BFR

**Definition (spec, V16_PLAN.md Part 2).** Backbone = M4 exactly: classical resize front-end (0 parameters), conv1, conv2, fc2(64 → 2) = z. Then

  logits = W_c z + b + g · W_q ⟨P⟩(u),  u_i = s · tanh(z_i)

- **Classical path** W_c z + b = M4's own head fc3(fc_mid(z)) with the linear head variant, fc_mid width trimmed to w(K) (A2 below).
- **Circuit.** 2 qubits from |00⟩, 2 re-uploading layers, each Ry(u₀)⊗Ry(u₁) → trainable Ry(θ_l0)⊗Ry(θ_l1) → CNOT(q0 → q1). No ZZ product term.
- **Readout.** Exact expectations of all 15 non-identity two-qubit Paulis, in the project's order (`closed_form_zz_v16.PAULI_NAMES`, first letter on q0).
- **Simulator.** Batched torch statevector, (B, 4), no per-sample loop: `bfr_head_v16b.py`.
- **Parameters and initialisation.**
  - s: learnable scalar, init π/2.
  - θ: 2 × 2, init 0 (the spec leaves this open; fixed here).
  - g: learnable scalar, init 0.
  - W_q: K × 15, no bias, init N(0, 0.01²).
  - At initialisation the logits equal the classical path exactly.
- **Invocation.** `--model M4 --v16b_head bfr`. Every M4 code path applies unchanged. The result JSON is named `result_M4_…` inside `results_v16b_bfr_*`; the sidecar `v16b_run_BFR_seed<S>.json` records `model = BFR`.

**Structural facts, stated now and verified by `verify_v16b.py` (not design choices):**
1. Every gate is real, so the state is real. The 6 Paulis with exactly one Y (YI, IY, XY, YX, YZ, ZY) are identically 0 for every input and every θ. BFR effectively reads 9 features. The 6·K W_q weights on the dead columns receive only weight decay. They are kept because the spec asks for all 15, and they are counted in the parameter budget.
2. Ry(u)·Ry(θ) = Ry(u + θ). The trainable rotations act as learned offsets on the encoded angles.
3. Every feature is a trigonometric polynomial of degree ≤ 2 in each u_i, and |u_i| < |s|. This is the "bounded frequency" property, checked by FFT.
4. At g = 0, only g receives gradient among the quantum tensors. s, θ and W_q start moving after the first step. This is the intended residual-gate behaviour (see D1 for the consequence in the trainer).

**Parameter counts** (verified in every job log; the gate checks them):

| K | M2-w15 | M4 | **BFR** | BFR − M4 | BFR fc_mid width |
|---|---|---|---|---|---|
| 5 | 1686 | 1719 | **1720** | +1 (+0.06%) | 5 (M4: 15) |
| 10 | 1766 | 1799 | **1799** | 0 (0.00%) | 3 (M4: 15) |

BFR = M4 − fc_mid/fc3 at width 15 + fc_mid/fc3 at width w + quantum branch (s 1, θ 4, g 1, W_q 15K) = M4 + 3(w − 15) + K(w − 15) + 6 + 15K.

**Initialisation and pairing.** BFR is built as M4 at the same seed, with every RNG stream in its original state. The surgery then runs in a forked RNG, so every global RNG stream is left exactly where building M4 leaves it.
- Backbone and fc2 are M4's tensors.
- fc_mid takes M4's first w rows.
- fc3 takes M4's first w columns, rescaled by √(15/w) so that it has PyTorch's default bound for fan-in w.
- W_q is drawn from a private generator seeded with the run seed.
- A BFR forward/backward consumes no RNG, and the gradient probe (D1) draws the same batch as M4's.

So at the same seed, BFR sees M4's data split and M4's batch order. Verify proves the RNG equality and the identical backbone/fc2. The gate checks that every BFR run's split hashes equal the stored v14 M4 log of the same cell and seed, where that log is found.

**What BFR and M4 still differ in (stated now, so the reading can't drift).**
- **Classical-path width.** With a linear head, fc3∘fc_mid is an affine map 2 → K of rank ≤ 2 for any width ≥ 2. The trimmed classical path therefore has exactly M4's function class (verify: it reproduces M4's logits). What differs is the parameterisation, through implicit regularisation under AdamW weight decay. This is the pre-registered caveat: **BFR's starting point is a slightly narrower-parameterised M4, not M4 exactly.**
- **The gated quantum branch.** This is the intended difference.
- **Scope of attribution.** Any effect of BFR is an effect of the whole design: bounding, no product term, gating and the residual form together. **Causal attribution to the product term needs BFR-ZZ, which A1 cut from the main grid.**

## 3. Protocol (unchanged from v14)

- **Fixed settings, identical to v14:**
  - w15, CE, linear head;
  - `--deterministic`, `--save_predictions`;
  - AdamW lr 1e-3, wd 1e-4, batch 32, dropout 0;
  - noise off (V15_ALPHA unset);
  - val 125/class, test 200/class (fixed);
  - patience 20;
  - max-epoch cap 200 × 500 / N (2000 at N = 50, 200 at N = 500).
- **Data split** comes from the unchanged `load_dataset`, so BFR and M4 see identical images at each (K, subset, N, seed).
- **Warmup decision (stated before the smoke).** The trainer's rule is `needs_warmup = model_name in (M2, M3, M5) or isinstance(back_end, QNNBackend)`.
  - BFR runs as `--model M4` with its own back-end class, so **min_epochs = 0, the same as M4**.
  - This is also the right setting by design: BFR starts as the classical head (g = 0), so there is no quantum plateau to protect against.
  - Giving BFR a 30-epoch warmup that M4 didn't have would itself confound the comparison (the lesson of v13.5 A1).
  - The gate checks `[V135] warmup: model=M4 min_epochs=0` in every log.
- **Walltime.** 1 h for every cell (K = 10: user decision). Trainer budget 0.85 h.
- **Logged per run.** g, s and θ at the best checkpoint (the reported model) and at the stop epoch, in the log (`[V16B] GS`) and the sidecar.

## 4. Design

| Arm | Model | K | Subsets | N per class | Seeds | Jobs |
|---|---|---|---|---|---|---|
| b | BFR | 5 | A {1,3,5,8,9}, B {0,2,6,8,9} | 50, 500 | 42–51 | 40 |
| b | BFR | 10 | all digits | 50, 500 | 42–51 | 20 |
| r | M4 (reproduction gate) | 5 | A | 500 | 42 | 1 |
| | | | | | **total** | **61** |

**Comparators are read, not re-run (A1):**
- M4 from v14 at N = 50 and N = 500: `qcq_cnn_v14/results_v14_k{K}_{TAG}_n{N}`.
- M2-w15 from v14 at N = 50 (`results_v14_k{K}_{TAG}_n50`) and from v13 at N = 500 (`qcq_cnn_v13/results_v13_k{K}_{TAG}_w15`).
- The smoke submission refuses unless all 120 stored files exist.
- Pooling the stored v13 N = 500 runs with v14 was validated by v14's bit-exact reproductions.

**Smoke (7 jobs, one submission, one gate):** seed 42 at each of the 6 BFR cells, plus the M4 reproduction. **Grid:** the other 54.

**Reproduction (gated):** M4 at K = 5 A, N = 500, seed 42 must equal the stored v14 run **exactly**: test_acc, best epoch, stop epoch and every per-image prediction. This proves the patched trainer, the environment and the stored comparators line up.

**Optional, not part of this plan:** BFR-ZZ at K = 10, N = 50, seeds 42–51 (10 jobs), only after the BFR grid is submitted and only on the user's go. If it is run, it is first logged below as a dated amendment, with its own definition and parameter count.

## 5. Analysis (`collect_v16b.py`)

- **Pairing.** Contrasts are paired over seeds 42–51.
- **K = 5 pooling.** K = 5 is pooled over subsets (per-seed mean of A and B, df = 9), with A and B reported alongside.
- **Test-set alignment.** Every model's stored `test_labels` must agree index-for-index at each (K, subset), or the script stops.
- **Nulls.** Every null is reported with its MDE (80% power, α = .05).

**Primary.** At each (K, N):
- **D4 = BFR − M4**, with 95% CI, a TOST at **±1 pp**, and a two-sided difference test.
- **D2 = BFR − M2-w15**, with 95% CI and a two-sided difference test.
- **Holm correction:** within each K, over the two N cells, per contrast. It is applied separately to the TOST p-values and to the difference p-values (v16a precedent).

**Secondary (uncorrected):**
- per-subset D4 and D2;
- g and s at the best checkpoint and at the stop epoch, per cell;
- the Spearman correlation between |g|·‖W_q‖ and the per-run D4;
- median best epoch, BFR vs M4.

**Exploratory:** per-image error agreement, comparing J(BFR_s, M4_s) with J(M4_s, M4_t) for s ≠ t, and J(BFR_s, M2_s).

## 6. Predictions (written before any v16b result)

| | Prediction | Operational test |
|---|---|---|
| **P1** | BFR − M2 ≥ **+1.2 pp** at K = 10, N = 500, and ≥ **+6 pp** at N = 50 (K = 5 and K = 10) | In each of the 3 cells: mean D2 ≥ threshold **and** D2 detected > 0 (Holm-adjusted two-sided p < .05) |
| **P2** | BFR ≡ M4 within ±1 pp at N = 500; BFR − M4 ≥ −1 pp at N = 50 | N = 500, both K: Holm-adjusted TOST p < .05. N = 50, both K: non-inferiority, H₀: D4 ≤ −1 pp rejected at one-sided p < .05 |
| **P5** | BFR seed SD ≤ 1.5 × M4's | Every (K, N) cell; at K = 5, the SD of the pooled per-seed means |

P3 (causal test of the product term) and P4 (twin) are **not tested** in v16b (A1).

## 7. How each outcome will be read (from V16_PLAN.md)

| Outcome | Reading |
|---|---|
| BFR ≫ M2 (P1) and ≈ M4 (P2) | Bounding the encoding removes the head's cost. This is consistent with the diagnosis; **causal attribution to the product term needs BFR-ZZ.** |
| BFR ≈ M2 | Bounding the phase is not enough; the cost is periodicity or optimisation. |
| BFR > M4 by ≥ 1 pp, detected | The first fair hybrid win in the project. **Confirm on fresh seeds before claiming it.** |
| BFR < M4, detected | **Check the implementation and the g/s logs first**, then read: the gated branch costs accuracy against the classical head it starts from. |
| P2 not shown, nothing detected | Underpowered at n = 10 in that cell; report CI and MDE. No equivalence claim there. |
| Branch switched off: effective weight \|g\|·‖W_q‖_F < 0.05 at the best checkpoint in more than half the runs of a cell | BFR collapsed onto its classical path there. Any BFR ≈ M4 result in that cell is "the quantum branch was switched off", and is reported as such. |

## 8. Run order, gates and stop rules

1. **Setup.** `setup_v16b.sh` builds `qcq_cnn_v16b` from `qcq_cnn_v16a` (code only, data linked), then patches and verifies. It must end in `ALL CHECKS PASSED`. Verify checks:
   - M1–M5 and M2c are bit-identical to the pre-patch (v16a) file;
   - the circuit matches an independent statevector and Qiskit;
   - the structural facts of §2;
   - the surgery, parameter table, RNG and init pairing, and the function-class check;
   - the gradient behaviour;
   - dry runs, including M4 through the patched file equalling the pre-patch file prediction by prediction, and BFR determinism.
2. **Smoke** (`submit_v16b_smoke.sh`, 7 jobs). It refuses without the verify log, this file in the repo, and all 120 stored comparator files.
3. **Gate** (`check_v16b.sh smoke`). Per run:
   - rc = 0, no continuation, loss = ce, n_classes, class list, head_variant linear, n_obs = 15;
   - trainer parameter check VERIFIED for M4, BFR HEAD line VERIFIED with the §2 table and initial values;
   - determinism, max_epochs = cap(N), warmup 0, noise off, no v16a line;
   - the gradient probe line and g restored bit-exact;
   - the GS line, RUN self-check MATCH, and the sidecar agreeing with the result JSON;
   - split audit;
   - the same images within (K, subset, N, seed), one test set per (K, subset), splits equal to the stored v14 M4 logs where found;
   - reproduction exact.
4. **Grid** (`submit_v16b_grid.sh`, refuses without the smoke gate). Then **the gate again** (`check_v16b.sh grid`) **before any accuracy is read**. The smoke gate prints epochs, job times and g/s only, no accuracies.

**Stop rules (fixed now):**
- The reproduction fails → stop. Nothing is read until the cause is found.
- Any smoke run stops at its epoch cap → stop before the grid; the cap is revisited as a dated amendment.
- Any continuation (time budget hit) in the smoke → stop; walltime is revisited as a resource amendment.
- In the grid, cap hits are reported per cell; a cell with > 2/10 is labelled budget-limited.

## 9. What would NOT change this plan

A high or low headline number. No grid is re-run to improve a result. Any change after the smoke is submitted is logged below, dated, **before** it is acted on.

## Amendments and defects

- **A1 (2026-10-21, in V16_PLAN.md, before any v16b code).** Minimal v16b. BFR only; M4 and M2-w15 reused; BFR-ZZ and BFR-twin cut; P3 and P4 not tested; the optional 10-job BFR-ZZ cell (K = 10, N = 50) only after the BFR grid and on the user's go.

- **A2 (2026-10-21, before the smoke; implementation of the parameter-budget rule).** The spec says "trim fc1's hidden width". The code has **no fc1**. M4 is resize → conv1 → conv2 → flatten(64) → fc2(64 → 2) = z → fc_mid(2 → 15) → fc3(15 → K), and the only hidden width in the head is fc_mid's. BFR therefore trims **fc_mid** to w = 5 (K = 5) and w = 3 (K = 10), giving 1720 vs 1719 and 1799 vs 1799 (table, §2). w is chosen by a fixed rule: minimise |BFR − M4| over 2 ≤ w ≤ 15, ties to the larger w. Because the head is linear, this keeps the classical path's function class identical to M4's (verified). Two alternatives were rejected:
  - trimming conv2's channels (16 → 15/14) would change the backbone, which the spec fixes as M4's;
  - a bare Linear(2 → K) classical path undershoots M4 by 1.7% (K = 5) and 1.1% (K = 10), outside ±1%.

  The spec's caveat holds with "fc_mid" in place of "fc1".

- **D1 (2026-10-21, before the smoke; defect, found by reading the trainer).** `train_model()` calls `assert_gradient_flow()`, which aborts the run if any trainable tensor gets exactly zero gradient. At g = 0, s, θ and W_q get exactly zero gradient by design (§2 fact 4), so every BFR job would have aborted before epoch 1.

  **Fix (no design change):** for BFR runs the probe runs with g temporarily set to 1, so it checks that the gradient chain is connected. g is then restored to exactly 0 (logged `bit-exact: True`). The probe draws the same batch as M4's, so the RNG sequence and batch order are unchanged. Verify checks both regimes on the real model: at g = 0 only g among the gated tensors gets gradient, and at g = 1 every tensor does. No job had been submitted.

- **A3 (2026-10-22, AFTER the v16b grid results were read and BEFORE any v16b-ext job; user's decision; design extension "v16b-ext").** Two 10-job controls at K = 10, N = 50, seeds 42–51, with the same protocol and pairing as the v16b grid. **Honesty note:** v16b's results are already known (`V16B_RESULTS.md`: BFR − M4 = −4.66 pp at K10 N50, detected). These predictions are therefore informed by them. They are fixed here before any ext data exists. Supersedes A1's "optional BFR-ZZ cell" note (BFR-ZZ now runs, alongside M4-trim).

  **Models.** Both run as `--model M4 --v16bx_head …`. They are built on M4 at the same seed in a forked RNG, so they get M4's split and batch order.
  - **M4-trim** (`m4trim`): M4 with fc_mid trimmed 15 → 3 exactly as in BFR (same row slicing, fc3 rescaled by √(15/3)), and no quantum branch. **1643 parameters** (= M4 − 156 = BFR − 156). It is not iso-parametric with M4, by design: it is BFR's classical path on its own (verify: tensor for tensor), so it isolates the trim.
  - **BFR-ZZ** (`bfrzz`): BFR with the ZZ product term restored in the encoding. In each re-uploading layer: Ry(u₀)⊗Ry(u₁) → CX(q0 → q1)·P(ζ) on q1·CX(q0 → q1) → Ry(θ_l0)⊗Ry(θ_l1) → CNOT(q0 → q1), with **ζ = 2(π − u₀)(π − u₁) on the bounded u = s·tanh z** (ZZFeatureMap convention, as `closed_form_zz_v16`).
    - Everything else equals BFR: s, θ, g, W_q, the trim, the 15-Pauli readout, and the initialisation (same private-generator draw: BFR-ZZ's init equals BFR's at every seed).
    - 0 new parameters: **1799**.
    - Verify: with ζ switched off it equals BFR exactly, so the product term is the only difference.
  - **Choice stated:** ζ uses the bounded u, not raw z. Restoring the unbounded term would change two factors at once (product and bounding). Within |u| ≤ s the product still raises the local frequency in u: |∂ζ/∂u₀| = 2|π − u₁| ≤ 9.4, against 1 per Ry upload. Verify measures Fourier content above degree 2 (BFR has none).
  - **Confound stated:** the ZZ phase makes the state complex, so the 6 single-Y Paulis are no longer 0. BFR-ZZ reads 15 non-zero features against BFR's 9. This can't be removed while keeping the readout identical.

  **Comparators (read, not re-run):** M4 and M2-w15 from v14 (`results_v14_k10_all_n50`), BFR from v16b (`qcq_cnn_v16b/results_v16b_bfr_k10_all_n50`).

  **Primary contrasts.** Paired over seeds, two-sided paired t, **Holm over the 3**:
  - C1 = BFR − M4-trim (the quantum branch's own effect on top of the trimmed head);
  - C2 = M4-trim − M4 (the effect of the trim);
  - C3 = BFR-ZZ − BFR (the effect of restoring the product term).

  C1 + C2 = BFR − M4 exactly (checked by collect).

  **Predictions (written now).** A prediction **holds** only if the point estimate meets the threshold **and** the contrast is detected in that direction (Holm p < .05), the same rule as P1. A threshold met without detection is reported as "direction only, not confirmed".
  - **E1.** M4-trim − M4 explains ≥ 50% of the BFR − M4 deficit at K10 N50: mean C2 ≤ −2.3 pp.
  - **E2.** Restoring the product term brings back a large part of the ZZ cost: mean C3 = BFR-ZZ − BFR ≤ −5 pp.

  **Readings.**
  - E1 holds → at least half of BFR's deficit is the narrower classical parameterisation, not the quantum branch. The branch's own effect is C1.
  - E1 fails and C1 is detected < 0 → the deficit belongs to the quantum branch.
  - E2 holds → the product term is causally implicated in the ZZ head's cost even on bounded inputs (subject to the Y-Pauli confound).
  - E2 fails with C3 near 0 → no evidence the product term carries the cost; v16b's recovery is attributed to bounding, gating and the residual form.
  - C3 detected > 0 → the product term helps.

  **Secondary (uncorrected):** fit/gap decomposition of each contrast (train_acc_at_best, eval mode, best checkpoint), seed SD, and g and s for BFR-ZZ vs BFR. MDE is reported with every null.

  **Run order.** `setup_v16bx.sh` builds a new repo, `qcq_cnn_v16bx`, from `qcq_cnn_v16b`'s code; the v16b repo and its results are untouched. The patch adds one block with its own flag and edits no line. Verify must pass: M1–M5, M2c and BFR are bit-identical to the v16b file, plus the circuit, surgery and dry-run checks.
  - **Smoke (3 jobs):** M4-trim s42, BFR-ZZ s42, and a **gated reproduction**: BFR K10 N50 s42 through the new file must equal the stored v16b run exactly.
  - Then the gate (`check_v16bx.sh smoke`), the grid (18 jobs), and the gate again before any accuracy is read.
  - The stop rules of §8 apply unchanged. Walltime is 1 h.

*(Date every further amendment and keep it.)*
