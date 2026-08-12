# QCQ-CNN Thesis — Models Source of Truth

**Last verified:** 2026-07-20
**Owner:** Yasser Arafat (z5477389), UNSW Bachelor of Engineering (Electrical Engineering) Honours
**Thesis:** End-to-End Trainable Quantum-Classical-Quantum CNNs for Image Classification
**Supervisor:** A/Prof Vidhyasaharan Sethu | **Assessor:** Dr Sirojan Tharmakulasingam

> **How to use this file:** This is the reference specification for the M1–M4 pipeline. If any other note contradicts this file, **this file wins** unless you've just re-verified something against a primary source (the actual paper PDF, the actual Long et al. GitHub notebook, or an uploaded circuit figure) — in which case, update this file first, then proceed.

---

## 0. Research Question

Does replacing Long et al.'s (2025, *Scientific Reports*) frozen random quanvolutional front-end with a trainable Circuit 4 ansatz (Hur et al. 2022) improve image classification — and does any improvement survive IBM-calibrated hardware noise?

---

## 1. The Four Models — Overview

| | Front-end | Backbone | Back-end/Head | Total params | Role |
|---|---|---|---|---|---|
| **M1** | Frozen `RandomLayers` (Henderson-style) | Shared CNN | QNN (ZZFeatureMap + RealAmplitudes) | **1,604** | Replication anchor — exempt from iso-parametric matching |
| **M2** | Trainable Circuit 4 (Hur et al.) | Shared CNN | QNN (ZZFeatureMap + RealAmplitudes) | **1,610** | **Core contribution** — reference point (P_M2) |
| **M3** | Trainable Circuit 4 | Shared CNN | Classical `Linear(64→2)` | **1,600** | Ablation — isolates QNN back-end contribution |
| **M4** | Bilinear resize (0 params) | Shared CNN | Classical `Linear(64→2)` | **1,594** | Fully classical baseline — isolates quantum front-end |

**Attribution logic (locked):**
- **M2 vs M1** → isolates front-end trainability (same QNN back-end both sides)
- **M2 vs M3** → isolates the quantum back-end's contribution (same trainable front-end both sides)
- **M3 vs M4** → isolates the quantum front-end's contribution (same classical head both sides)

All four share the exact same CNN backbone module, which is what makes the comparison iso-parametric. M1 is explicitly **not** part of the iso-parametric matching group — it's a faithful replication anchor, not a fairness comparand.

---

## 2. Full Data Flow, Stage by Stage

### Stage A — Front-end (differs per model)

Input to all front-ends: `(B, 1, 28, 28)` grayscale image.
Output of all front-ends: `(B, 4, 14, 14)` — 4 channels, 14×14 spatial grid.

**M1 — Frozen RandomLayers:**
- 2×2 non-overlapping patches taken directly off the raw 28×28 image, stride 2 → 14×14 grid of patches (**no downsampling step before patching**).
- Each patch (4 pixels) angle-encoded as `RY(π · pixel)` on a 4-qubit register.
- `qml.RandomLayers` applied with **fixed random weights** (registered as a buffer, not a `nn.Parameter` — no gradient flows through this layer).
- Measure `⟨Z⟩` on all 4 qubits → 4 output channels per patch.
- **0 trainable parameters.**

**M2 — Trainable Circuit 4:**
- Identical patching and angle-encoding to M1.
- After encoding, apply **Circuit 4** (see Section 3) instead of `RandomLayers`.
- Circuit 4 applied identically (weight-shared, translationally invariant) to qubit pairs (0,1) and (2,3) within each 4-qubit patch register.
- **6 trainable parameters** (θ1–θ6), shared across all patches and both qubit pairs.
- Gradients via PennyLane parameter-shift rule.

**M3 — same front-end as M2** (trainable Circuit 4, 6 params).

**M4 — Bilinear resize (classical, non-quantum):**
- `F.interpolate(image, size=(14,14), mode='bilinear')` on the raw 28×28 image → single-channel 14×14.
- Repeat the single channel 4× → `(B, 4, 14, 14)`, matching the shape the quantum front-ends produce.
- **0 trainable parameters.** No learned front-end — deliberately kept parameter-free to preserve symmetry with M1's "does nothing clever" frozen front-end. (An earlier design used a learned `Conv2d(1,4,k=2,s=2)` front-end for M4; this was explicitly rejected because it would give M4 a smarter front-end than any other model has, breaking the comparison.)

### Stage B — Shared CNN Backbone (identical module, all 4 models)

```
Input: (B, 4, 14, 14)
  → Conv2d(4, 8, kernel_size=3, no padding)   → (B, 8, 12, 12)   [296 params]
  → ReLU
  → MaxPool2d(2)                              → (B, 8, 6, 6)
  → Conv2d(8, 16, kernel_size=3, no padding)  → (B, 16, 4, 4)    [1,168 params]
  → ReLU
  → MaxPool2d(2)                              → (B, 16, 2, 2)
  → Dropout2d
  → Flatten                                   → (B, 64)
```

**Backbone total: 1,464 params** (296 + 1,168). Spatial trace: 14−3+1=12 → /2=6 → 6−3+1=4 → /2=2 → 16×2×2=64. Verified by direct arithmetic.

### Stage C — Back-end / Head (differs: M1/M2 get a QNN, M3/M4 get a linear head)

**M1 & M2 — QNN back-end:**
```
64-dim backbone output
  → Linear(64, 2)                              [130 params]  (classical projection into the QNN)
  → ZZFeatureMap(2)                             [0 params, fixed encoding, not trainable]
  → RealAmplitudes(2, reps=2)                   [6 params — SEE SECTION 4 for why reps=2]
  → (via EstimatorQNN + Qiskit TorchConnector)
  → Linear(1, 2)                                [4 params]  (final classification logits)
```
**Back-end total: 140 params** (130 + 0 + 6 + 4).

**M3 & M4 — Classical head (no quantum back-end):**
```
64-dim backbone output
  → Linear(64, 2)                               [130 params]
```
**Head total: 130 params.**

---

## 3. Circuit 4 (Hur et al. 2022, Fig. 2d) — ✓ VERIFIED 2026-07-18 against source figure

Six shared trainable parameters (θ1–θ6), applied identically to qubit pairs (0,1) and (2,3).

```
Column:        1           2            3           4
q_a: ──────  RY(θ1)  ──  RZ(θ3)  ──   RY(θ4)  ──    ●    ──────
                            │                        │
q_b: ──────  RY(θ2)  ──    ●    ──   RY(θ5)  ──  RZ(θ6)  ──────

Legend: ● = control qubit for that column's CRZ. The box (RZ(θ)) on the
other wire in the same column is the target.
Column 2 → CRZ(θ3): control = q_b, target = q_a
Column 4 → CRZ(θ6): control = q_a, target = q_b
```

| Order | Gate | Wires | Param | Note |
|---|---|---|---|---|
| 1 | RY | q_a | θ1 | independent rotation |
| 2 | RY | q_b | θ2 | independent rotation |
| 3 | CRZ | ctrl=q_b, tgt=q_a | θ3 | first entangling gate |
| 4 | RY | q_a | θ4 | **RY, not RZ** |
| 5 | RY | q_b | θ5 | **RY, not RZ** |
| 6 | CRZ | ctrl=q_a, tgt=q_b | θ6 | second entangling gate, reversed control |

Shorthand: **RY → RY → CRZ → RY → RY → CRZ**

**Sanity check when reading `train_gadi.py`:** search for `circuit4_block`. It must contain exactly four `qml.RY` calls and two `qml.CRZ` calls. **Zero** `qml.RZ` calls and **zero** `qml.CNOT`/`qml.CZ` calls anywhere in that function. If you see RZ, CNOT, or CZ in there, the implementation has regressed to an old, wrong version.

### ⚠️ History of wrong readings — do not reintroduce any of these

This circuit has been misread multiple times across sessions. Every one of the following is **confirmed wrong** and must never resurface:

- ~~RY→RY→CRZ→**RZ→RZ**→CRZ~~ (middle two gates were wrongly read as RZ instead of RY)
- ~~RY→RY→**CZ**→RY→CZ~~ (misread as unparameterized CZ instead of CRZ — impossible anyway, since CZ carries no parameter and Circuit 4 needs exactly 6 trainable θ's)
- ~~Ry–Rz–**CNOT**–Ry–Rz–**CNOT(reversed)**~~ (misread as CNOT entangling gates)

**Source of truth for this circuit is the uploaded Hur et al. figure image, never a text description carried over from a previous session.** If you need to re-verify, re-upload the figure and re-read it fresh.

Prior to this block: each of the 4 pixels in a flattened 2×2 patch is angle-encoded as `RY(π · pixel)` on its own wire (fixed encoding, identical for M1 and M2 — only the ansatz after encoding differs).

---

## 4. RealAmplitudes reps — ✓ LOCKED: `reps=2` (6 params)

This value flip-flopped across three sessions before being conclusively settled. Documenting the resolution so it never gets re-litigated on stale grounds:

- One session claimed `reps=1` (4 params) was correct, citing an unverified memory of "Long et al.'s notebook cell 28." **This was never actually checked against the paper and should be disregarded.**
- A later session correctly pointed out the published paper states depth d=5 (`reps=2` in Qiskit) as the chosen configuration, giving 6 trainable RY parameters.
- This was **conclusively confirmed** by directly pulling Long et al.'s text via project knowledge search: the paper's own ablation shows depth 3 (`reps=1`) "converges rapidly in early epochs but shows a premature plateau with suboptimal accuracy," while depth 5 (`reps=2`) "achieves the highest final accuracy with stable convergence."
- This exactly matched a real symptom already observed in a smoke test: M2 plateauing at 80–90% accuracy, well below M3/M4. That run was later found to have been misconfigured with `reps=1`. **The plateau was an artifact of the wrong config, not a real architectural finding about the trainable front-end.**

**Do not use `reps=1` anywhere in the final experimental pipeline.** `RealAmplitudes(2, reps=2)` → depth 5 → 6 trainable parameters is locked.

---

## 5. Exact Parameter Counts

| Component | M1 | M2 | M3 | M4 |
|---|---|---|---|---|
| Quantum front-end | 0 (frozen) | 6 (Circuit 4) | 6 (Circuit 4) | 0 (bilinear resize) |
| CNN backbone (Conv1 296 + Conv2 1,168) | 1,464 | 1,464 | 1,464 | 1,464 |
| Linear(64→2) pre-QNN projection | 130 | 130 | 130 (this **is** the head, not pre-QNN) | 130 (this **is** the head) |
| ZZFeatureMap(2) | 0 | 0 | — | — |
| RealAmplitudes(2, reps=2) | 6 | 6 | — | — |
| Linear(1→2) post-QNN | 4 | 4 | — | — |
| **Total** | **1,604** | **1,610** | **1,600** | **1,594** |
| Δ vs P_M2 (1,610) | −6 (exempt) | reference | −10 (0.62%) | −16 (0.99%) |

M2 (P_M2 = 1,610) is the reference point. M3 and M4 are both within 1% of P_M2 by construction (not by any additional binary-search step — the shared backbone + shared 130-param head naturally lands both within tolerance). M1 is exempt from this matching requirement entirely.

---

## 6. Training Protocol (Locked)

| Parameter | Full-scale (Gadi) value | Note |
|---|---|---|
| Batch size | **32 — all four models (M1–M4)** | Changed 2026-07-28 (v5). Was 1 for M1–M3 / 128 for M4. The bs=1 constraint was tied to PennyLane's parameter-shift *tape execution*, which no longer exists in the training path: the front-end circuits are now exact torch-native simulations and gradients come from autograd, verified numerically identical to parameter-shift (see Sec 10). 32 gives 156 optimiser steps/epoch at n_train=5,000 — ample for 1,600 parameters, retains useful gradient noise against the barren-plateau risk, and transfers lr=1e-3 from Long et al.'s bs=4 without retuning. **Matched across all four models** so that batch size is not a second uncontrolled variable inside the M3-vs-M4 comparison (the one that isolates the quantum front-end). **All M4 results prior to 2026-07-28 are superseded** — both the bs=32 and bs=128 generations. Validated by an explicit bs=1 vs bs=32 control run (M2, MNIST, seed 42) — see `pbs/run5_batchsize_control.pbs`. |
| Weight decay | **1e-4 — all four models, via AdamW** | Added 2026-07-28 (v5); previously absent entirely. Thesis A Sec 5.3 commits to weight decay applied uniformly across all four models; it had never been implemented (optimiser was plain `Adam`). `AdamW` is used rather than `Adam(weight_decay=)` because Adam couples the penalty to the adaptive learning rate, giving a different effective decay per parameter and breaking the 'uniform' requirement. **Measured effect at this model size: essentially none** — swept 0 to 1e-1 on M3/MNIST/seed42 at n_train=1000; `train_acc` stays pinned at 1.0000 at every value, `val_acc` moves ~2/250, and test accuracy is non-monotonic (noise at n=200). Kept as a standard, harmless default honouring the protocol commitment — **not** claimed as a tuned improvement. Re-check at full scale before reporting anything about regularisation. |
| Dropout | **0.0 — all four models** | Changed 2026-07-28 (v5); was 0.5 per Long et al. `Dropout2d` is the sole cause of the M1/M2 global decision inversion — it is the only module in the architecture whose behaviour differs between train and eval mode, and train/val are drawn from the same shuffled pool, so nothing else can explain ~0.96 train / ~0.04 val (see Sec 9). The QNN back-end's single-scalar bottleneck (`fc2 -> 2 -> QNN -> 1 -> fc3`) makes the threshold flip globally; M3/M4's linear head is immune. Applied to **all four** models to keep the backbone identical for the Sec 5 attribution. Zero trainable parameters, so no parameter count changes. Override with `--dropout`. |
| Max epochs | 100 | Matches Long et al. |
| Patience | 15 | Matches Long et al. |
| `min_epochs` warmup | 30, **M2/M3 only** | Deviation from Long et al. — trainable quantum front-end has a long early optimisation plateau; patience would fire prematurely without it |
| Seeds | 5: {42, 43, 44, 45, 46} | Deviation from Long et al.'s 10 seeds — documented, compute-budget driven |
| Train / Val / Test | 5,000 / 1,250 / 1,000 | Per task, matches Long et al. |
| Datasets | MNIST 3v5, Fashion-MNIST (T-shirt/top vs Trouser), MRI brain tumour (tumour vs no-tumour) | All three, matching Long et al. |
| Noise sweep | 0–200% of IBM-calibrated real device fidelity via `NoiseModel.from_backend()` | Longer-horizon phase, not yet run at full scale |

The batch-size choice and the min_epochs-warmup are **deliberate, documented deviations** from Long et al. — call both out explicitly in the thesis methods section, don't let them look like oversights.

> **Correction, 2026-07-28.** The thesis draft (§4.4.1) previously justified `batch_size=1` on the grounds that averaging parameter-shift gradients over a minibatch causes *systematic cancellation* from the *symmetric structure of the shift-rule terms*. **This does not hold.** The parameter-shift rule returns the exact per-sample gradient; averaging exact per-sample gradients gives the exact minibatch gradient, by linearity of differentiation. The ± shift terms are subtracted *within* each sample's evaluation, before any averaging — there is no cross-sample structure left to cancel. The observed symptom (bs=4 stalling at ~58%, near-random on a balanced binary task) is the signature of a **broken gradient chain**, and is almost certainly the same `input_gradients=False` `TorchConnector` bug later identified in M1: the shared CNN backbone stayed at random init while only `fc3`'s 4 parameters trained. A mis-averaged gradient degrades convergence; it does not pin a model to chance. Note also that Long et al.'s own stated reason for bs=1 is *"to reduce the simulation overhead in quantum layers"* and *"fine-grained gradient updates"* — a cost/granularity argument, not a correctness one; they used bs=4 for their classical and QC-CNN models. §4.4.1 needs rewriting before submission. Discuss with Sethu; the honest version ("I attributed a stall to minibatch gradient cancellation, then traced it to a missing gradient flag in the TorchConnector wrapper") is a stronger methodological contribution than the original claim.

---

## 7. Design Rationale (for defending under supervisor questioning)

- **Why Circuit 4 specifically:** selected via a tiebreaker against Circuit 2 and Circuit 6 — higher mean best accuracy, faster convergence (mean epoch 14.8 vs Circuit 6's 18.0), better cross-seed reliability (4/5 vs 3/5 seeds converging). Circuit 2 eliminated for insufficient expressibility. Circuit 6 eliminated for lower cross-seed reliability despite similar final accuracy.
- **Why hold the backbone constant across all 4 models:** isolates front-end trainability / quantum back-end / quantum front-end as the *sole* variable in each pairwise comparison (Section 1's attribution logic only works if everything else is identical).
- **Why quantum and classical params are reported separately in places, not just summed:** they represent fundamentally different computational resources. On hardware (and in the later noise phase) quantum params are updated via the parameter-shift rule, classical params via backprop. Two corrections to the old wording here: (a) "2 circuit evaluations per parameter" is right for RY but **wrong for CRZ**, whose generator has eigenvalues {0, 0, −½, +½} and therefore needs the **four-term** shift rule — Circuit 4 costs 16 tape evaluations per backward pass, not 12; (b) in the current *noiseless* phase (v5 onward) the front-end params are updated by autograd on an exact simulation, which is numerically identical to parameter-shift but not the same execution strategy. State the distinction plainly rather than implying parameter-shift is running when it isn't.
- **Why the QNN head bottlenecks through `Linear(64,2)`:** ZZFeatureMap(2) is built for exactly 2 input features (2 qubits). This is a real information bottleneck (97% compression from 64 dims) and is worth discussing honestly as a limitation, not glossing over.
- **Why M4's front-end is parameter-free bilinear resize, not a learned conv:** keeps M4's front-end "doing nothing clever," the same way M1's frozen random circuit does nothing clever. A learned front-end would make M4 the only model with two learned convolutional stages before the shared backbone, breaking the comparison.
- **Entanglement argument (core architectural claim):** entangled quantum circuits produce joint multiplicative terms (e.g. `sin(x_a)·sin(x_b)·sin(θ3)`) that are unreachable by any linear classical filter of the same size. This is the central "why should this work at all" argument.
- **Every plausible outcome is publishable** — including a finding that barren plateaus kill the trainable front-end, or that the classical baseline (M4) wins outright. Frame results honestly regardless of which way they land.

---

## 8. Known Issues / Infrastructure Notes (as of 2026-07-20)

- **Serial patch loop bottleneck:** the naive per-patch Python loop generates ~12.74M sequential circuit calls per epoch (5,000 samples × 196 patches × 13 parameter-shift evaluations), making epoch time impractical (~62–77 min on CPU simulators). Fix: (1) batched `unfold` call replacing the triple-nested loop, reducing to ~65K batched calls/epoch, and (2) `pennylane-lightning[gpu]` with `custatevec-cu11` for GPU-accelerated simulation. A `circuit4_batched.py` drop-in exists. Estimated post-fix epoch time: 2–5 min on V100, competitive or better on local RTX 3080.
- **M1 gradient barrier (unresolved):** M1 replication diverges sharply from Long et al.'s published 98.85%/99.41% (observed ~53%/62%). Leading diagnosis: gradient barrier at `TorchConnector` — if `input_gradients=True` silently fails, only the final 4-param layer trains rather than the full backbone. Needs direct diagnosis before trusting any M1 result.
- **Gadi compute quota:** has been exhausted at points during Phase 1 (97.8%+ used). Compute nodes also lack internet access — datasets must be pre-downloaded on the login node before job submission.
- **Long et al.'s own published numbers** (Table 1, full dataset, 10 seeds) for reference/sanity-checking replication runs:

  | Model | MNIST 3v5 | Fashion-MNIST |
  |---|---|---|
  | CNN (classical) | 98.79% | 99.19% |
  | QCCNN (quantum front-end, no QNN back-end) | 98.81% | 99.28% |
  | QCQ-CNN (full pipeline) | 98.85% | 99.41% |

  Note the QCCNN → QCQ-CNN gap is tiny (+0.04% MNIST, +0.13% Fashion) in Long et al.'s own numbers — the QNN back-end adds very little in their results. Worth keeping in mind when interpreting your own M2 vs M3 comparison.

---

## 9. Change Log

| Date | Change |
|---|---|
| 2026-07-18 | Circuit 4 gate sequence re-verified directly against Hur et al. source figure: corrected middle two gates from RZ back to RY |
| 2026-07-08 | RealAmplitudes reps=2 (not reps=1) conclusively confirmed against paper text + explains earlier plateau symptom |
| 2026-07-20 | This source-of-truth file created, consolidating all of the above after cross-checking an architecture verification report against full conversation history |
| 2026-07-20 | Fixed the Section 3 ASCII circuit diagram — column 2 had control/target backwards (showed q_a as control, q_b as target for CRZ θ3), contradicting the gate table directly below it. The table was already correct; only the ASCII art was wrong. Re-verified column-by-column against a freshly uploaded copy of the Hur et al. figure, zoomed in at pixel level. |
| 2026-07-27 | M4 batch size raised 32 → 128 (`--batch_size_m4` CLI override added) in response to supervisor feedback that GPU memory/utilisation was low. Infra-only — does not touch the iso-parametric comparison. M4 runs from before/after this date are not directly comparable (see Sec 6 batch size row). |

| 2026-07-28 | **v5.** Quantum front-ends reimplemented as exact batched torch tensor algebra (`quanv_torch.py`), replacing PennyLane `TorchLayer`. Root cause of the persistent 0% GPU: (a) 4-qubit circuits cannot occupy a V100 — cuStateVec launch overhead dominates, and `lightning.gpu` is *slower* than `lightning.qubit` below ~18 qubits; (b) `TorchLayer` re-serialised the v4 `unfold()` batch into B×196 sequential tapes, because the lightning devices don't support parameter broadcasting — so the v4 "batching fix" changed nothing. Verified: forward 8.9e-16 vs a brute-force 4-qubit reference, autograd gradients identical to the four-term parameter-shift rule, 282× measured speedup on CPU alone. Batch size unified at 32 across M1–M4 (Sec 6). `--broadcast_shift` and `verify_broadcast_shift.py` deleted as dead code. Permanent `[GRADCHECK]` tripwire added to `train_gadi.py`. |

| 2026-07-28 (later) | **v5 M1/M2 device bug found + fixed.** Smoke-scale run (500/125/100, seed 42) showed M3/M4 learning correctly (87%/94% test acc) but M1/M2 frozen at byte-identical accuracy every epoch despite healthy, comparable-magnitude gradients on every trainable tensor (`back_end.qnn.weight` ~3.3e-2, same order as `backbone.conv1.weight`). Confirmed both jobs ran on `Device: cuda`. Root cause: `EstimatorQNN`/`TorchConnector` has no GPU code path; v5's initial `model.to(device)` call moved `QNNBackend` onto `cuda` along with everything else, and the CUDA→CPU→numpy round trip inside `TorchConnector` silently failed to apply updated weights to what the Estimator actually evaluates, even though autograd computed correct gradients upstream. Fix: `QNNBackend` (fc2→qnn→fc3) is explicitly pinned to CPU after the top-level `.to(device)` call, and its `forward()` shuttles input/output across the CPU/GPU boundary — symmetric with how the front-end handled this in v4 before it became GPU-native. A per-epoch `qnn_weight_drift` diagnostic was added to `train_model`'s loop so this class of bug is visible in one run rather than needing a debug round-trip. **Not yet re-confirmed on Gadi hardware — this fix must be validated with a re-run before trusting any M1/M2 result.** |

| 2026-07-28 (later still) | **v5 M1/M2 device fix confirmed NECESSARY BUT NOT SUFFICIENT.** Re-ran M2 on Gadi with the CPU-pin fix in place: `qnn_weight_drift` confirmed real, substantial parameter movement every epoch (1.5e-2 down to 5.4e-3), yet `train_acc`/`val_acc` remained byte-identical across all 9 epochs before early stopping -- ruling out the device round-trip as the (sole) cause. Added `check_output_diversity()`, called before training and again after epoch 3, printing raw per-sample logits for 8 distinct images plus their std -- to directly test whether the model's decision depends on the input at all, rather than continuing to infer from indirect signals (gradients, weight drift). **Awaiting this run's output before further diagnosis** -- do not assume either the device fix or the underlying bug is resolved. |

| 2026-07-28 (final, this session) | **Reframed the M1/M2 investigation.** Output-diversity diagnostic showed the model always argmax-predicts class 0 (not literal input-blindness -- per-sample logits do vary, std nonzero, but never enough to flip the ranking), matching the observed accuracy exactly (0.506 x 500 = 253, the class-0 count). Between-sample spread actually *shrank* over 3 epochs, consistent with early-stage optimisation on a hard, bottlenecked landscape rather than a dead gradient. Cross-referenced against Long et al.'s own preliminary work (thesis draft Sec 4.3.2): their QCQ-CNN -- frozen front-end + trainable quantum back-end, architecturally identical to M1 -- is documented staying near 50% chance level for "roughly 10-15 epochs" before catching up. Found a real protocol gap that would explain the symptom: `min_epochs` warmup was gated on `model_name in ("M2","M3")` -- i.e. trainable FRONT-end -- but the plateau Long documents belongs to the trainable quantum BACK-end (M1, M2), which M1 was getting ZERO warmup for, including in the **locked full-scale protocol** (M1: min_epochs=0, patience=10 -- shorter than Long's own documented plateau length). **Fixed**: warmup condition widened to `model_name in ("M2","M3") or isinstance(model.back_end, QNNBackend)`, covering M1/M2/M3; M4 unaffected. Smoke-scale config bumped to min_epochs_quantum=15, patience=10, max_epochs=40 to give the documented plateau length room to resolve. **Still unconfirmed whether this alone explains the smoke-run symptom** -- awaiting a re-run with the corrected warmup before concluding this was a calibration issue rather than a deeper bug. If M1/M2 still don't move with 15 warmup epochs, return to the output-diversity/expressibility line of investigation (fc2 output range vs ZZFeatureMap encoding). |

| 2026-07-28 (resolution) | **Two findings, one confirmed hypothesis and one real bug.** (1) **The warmup hypothesis was CORRECT.** With min_epochs widened to cover M1/M2, the M2 smoke run's plateau broke at epoch 13 -- `train_acc` went 0.5060 -> 0.5120 -> 0.6060 -> 0.7500 across epochs 12-15, landing exactly in the "roughly 10-15 epochs" window Long et al. document for this architecture (thesis draft Sec 4.3.2). M1/M2 were never stuck; every earlier "frozen" observation was the documented plateau, and the device pin + gradient checks were chasing a symptom with a benign cause. (2) **A real early-stopping bug was exposed by the fix.** `patience_ctr` was incremented UNCONDITIONALLY from epoch 1, while `min_epochs` gated only the stop *check*. So the counter saturated during warmup and the run terminated on the very first epoch past `min_epochs`, every time -- discarding the model exactly as it began learning and returning the epoch-1 degenerate all-one-class state as "best" (hence the `FINAL test_acc=0.5000`). This contradicted this document's own Sec 6 wording ("warmup BEFORE patience counting"): the counting was never deferred, only the check. **At full scale this would have silently stopped M2/M3 at exactly epoch 30 on every seed, and M1 at epoch 30 once warmup was widened -- producing a full grid of degenerate results that looked like a legitimate architectural finding.** Fixed: `patience_ctr` now increments only once `epoch >= min_epochs`. Verified by replaying the observed val-accuracy trace through both code paths (old: stops epoch 15, best_val 0.4758; new: runs on, best_val 0.8500). |
| 2026-07-28 (open watch item) | **Val accuracy moved BELOW chance while train accuracy climbed** in the M2 smoke run's final epochs (train 0.5060->0.7500 while val 0.4758->0.4032->0.3226 on a balanced binary task; 0.3226 is worse than either constant-class baseline). Only 2-3 post-plateau epochs of evidence, so this may be nothing more than an early transient -- the corrected-warmup rerun will show whether val recovers once training continues past epoch 15. **If it does not recover, investigate the Dropout2d(0.5) / QNN interaction**: dropout is expectation-preserving, which is exact for M3/M4's linear head (E[f(x)] = f(E[x])) but NOT for M1/M2's nonlinear quantum back-end, so train-mode and eval-mode could be driving the ZZFeatureMap encoding into materially different regimes. This would explain why only the QNN-back-end models are affected. Do not act on this until the rerun data is in. |

| 2026-07-28 (ROOT CAUSE FOUND) | **M1/M2's failure is a globally INVERTED decision at eval time, not a failure to learn.** With the early-stopping bug fixed, the M2 smoke run trained to `train_acc=0.9600` while `val_acc` fell monotonically to **0.0323** (4/124). Inverting the validation predictions gives **0.9677 -- higher than the training accuracy.** The model learns the task correctly and then flips its decision at evaluation. **Mechanism:** `Dropout2d(0.5)` sits immediately before `flatten -> fc2 -> QNN`. Dropout is expectation-preserving (`E[dropout(x)] = x`), which is EXACT for M3/M4's linear head (`E[f(x)] = f(E[x])`) -- which is precisely why M3/M4 were unaffected and trained fine -- but NOT for M1/M2's nonlinear quantum back-end, where `E[QNN(x)] != QNN(E[x])`. The QNN's single scalar output therefore has a different mean in train vs eval mode, and `fc3 = Linear(1,2)` thresholds that scalar, so `argmax` flips when `(w0-w1)*q + (b0-b1)` changes sign -- moving EVERY sample's prediction in the same direction simultaneously. That is why the signature is a clean inversion rather than gradual degradation. **Mechanism verified numerically** in `scripts/verify_dropout_qnn_shift.py`, which simulates the real `ZZFeatureMap(2,reps=1) + RealAmplitudes(2,reps=2)` circuit with `Z(x)Z` exactly in torch: linear-head train/eval gap ~3e-3 (Monte Carlo noise), QNN-head shift 0.024 with 29/64 samples flipping argmax **at random init, before training amplifies it**. Added `--dropout` CLI override (dropout has zero trainable parameters, so this does NOT affect any parameter count or the iso-parametric comparison) and a `check_train_eval_shift()` diagnostic reporting the per-mode logit margin and flip count. **Awaiting the `--dropout 0.0` A/B confirmation on hardware (`pbs/submit_run5_dropout_ab.sh`) before selecting the permanent fix.** |
| 2026-07-28 (decision pending) | **If the A/B confirms dropout is the cause, the fix must preserve backbone identity across all four models** -- the iso-parametric attribution logic (Sec 5) requires M1-M4 share an identical backbone, so dropout cannot simply be disabled for M1/M2 only. Options, in order of preference: (a) **disable dropout for all four models** -- keeps the backbone identical, deviates from Long et al., defensible as a documented methodological finding in the same class as the batch-size correction; (b) reduce the rate for all four (e.g. 0.1) -- smaller deviation, but only shrinks the shift rather than eliminating it, and the shift magnitude scales with the rate; (c) replace `Dropout2d` (which zeroes whole channels) with standard `Dropout` -- lower variance per forward pass, but same expectation-preservation argument applies, so this likely only reduces severity. **Recommend (a)** pending confirmation. Note the finding itself is publishable: dropout is near-universal in hybrid QML architectures and this interaction does not appear to be widely documented. |

| 2026-07-28 (**HYPOTHESIS REFUTED — M2 WORKING**) | **The dropout/train-eval-mismatch hypothesis from the preceding two entries is WRONG. Do not cite it.** The A/B test (`--dropout 0.5` vs `0.0`, M2/mnist/seed42) had BOTH arms train successfully: dropout=0.5 -> best_val 0.9919, **test 0.9700**; dropout=0.0 -> best_val 0.9919, **test 0.9500**. The `check_train_eval_shift()` diagnostic added specifically to test the hypothesis refuted it directly: at dropout=0.5 the train-mode vs eval-mode logit margins were near-identical at every checkpoint (+0.4823/+0.4634 at init, +0.2329/+0.2435 at epoch 15, +0.1978/+0.1947 at epoch 25) with **0/32 predictions flipping** in every case. The predicted mechanism is not operating at the magnitude required. The `scripts/verify_dropout_qnn_shift.py` demo measured a real but SMALL shift (0.024) at random init; that was over-extrapolated to explain a catastrophic inversion, which the hardware data does not support. **What actually fixed M1/M2 was the early-stopping `patience_ctr` bug plus widening the `min_epochs` warmup to cover the QNN back-end** (preceding entries) -- nothing to do with dropout. |
| 2026-07-28 (open, lower priority) | **The single observed inversion (run 174891473: train 0.96 / val 0.032) remains unexplained and appears TRAJECTORY-DEPENDENT, not systematic.** Same seed, same dropout=0.5, same config as the successful run 174938392 -- the only difference is that `check_train_eval_shift()` performs 8 dropout-active forward passes before training, consuming RNG and shifting every subsequent dropout mask. So the two runs followed different stochastic trajectories from the same seed: one converged to ~0.99 val, the other to a catastrophically inverted solution. This means **the QNN back-end can converge to an inverted-polarity solution on some trajectories** -- `fc3 = Linear(1,2)` can represent either polarity of the QNN's scalar output, and the long plateau is where that commitment happens. Evidence base is n=1, so do NOT over-correct. **Action: run the full 5 seeds at the locked dropout=0.5 and count inversions.** An automatic detector is now in `train_model()` (see `[INVERSION]` warnings) -- it flags any epoch where val_acc < 0.5 while train_acc > 0.7, which is the unambiguous signature. Only switch to dropout=0.0 if inversions actually recur across seeds. |

| 2026-07-28 (**correction to the entry above; inversion is REAL and RECURRING**) | The preceding entry declared the dropout hypothesis "refuted." **That was an overcorrection -- the A/B test could not refute it.** Both A/B arms happened to train healthily, so there was no inversion present for `check_train_eval_shift()` to detect; the 0/32 flip readings were taken on HEALTHY runs. The A/B established only that dropout is not *sufficient* to cause inversion. It said nothing about whether it is *necessary*. **New data:** the 4-model smoke batch (seed 42, dropout 0.5) gave M2=0.9700, M3=0.9600, M4=0.9500 -- and **M1 INVERTED** (train_acc 0.9620 while val_acc fell to 0.0403; inverting val gives 0.9597; reported test_acc=0.5000). The `[INVERSION]` detector caught it across 11 consecutive epochs. This is the **2nd observed inversion**, and the pattern is now sharp: **both models that invert (M1, M2) have the nonlinear QNN back-end; M3/M4 (linear head) never invert**, on identical data, identical split, identical dropout. |
| 2026-07-28 (**confound identified — decisive test now in code**) | A confound was present in every inversion observation so far and should have been caught earlier: `train_acc` is measured in **TRAIN mode** (dropout active) via `run_epoch(train=True)`, while `val_acc` is measured in **EVAL mode** (dropout off). Comparing them therefore mixes *different data* with *different mode*, and cannot distinguish the two. **Added `[MODE CHECK]`**: when inversion is detected, it evaluates the TRAINING set in EVAL mode, isolating the mode effect on identical data. Interpretation: `tr_acc_eval ~= 1 - tr_acc` means the decision inverts between modes on the same data, and since dropout is the only train/eval difference, dropout is the cause via the nonlinear QNN (`E[QNN(x)] != QNN(E[x])`); `tr_acc_eval ~= tr_acc` means mode is NOT the cause and the val split needs investigating instead. Runs only when inversion is suspected, so healthy runs pay nothing. **Plausible mechanism if confirmed:** under dropout, `fc2`'s output has much higher variance, and the QNN maps that through a bounded oscillatory function, so `E[QNN(noisy)]` is pulled toward the function's mean (~0) while `QNN(clean)` can sit far from it. `fc3 = Linear(1,2)` thresholds that scalar at `q* = (b1-b0)/(w0-w1)`; if `q*` falls between the two means, EVERY prediction flips at once -- which also explains the trajectory-dependence, since where `q*` lands varies by run. **Do not treat this as established until `[MODE CHECK]` fires on hardware.** |

| 2026-07-28 (**DECISION: bs=32 uniform, locked**) | Considered and **rejected** adding a bs=1 arm for M1 to match Long et al. exactly. **Decision: batch size 32 for all four models, including M1.** Rationale: the iso-parametric attribution logic (Sec 5) is the thesis's primary contribution, and it requires that batch size not be a second uncontrolled variable across the M1/M2/M3/M4 comparisons. A mixed protocol (M1 at bs=1, others at bs=32) would contaminate the M2-vs-M1 comparison -- the one that isolates front-end trainability -- with a batch-size difference. Uniformity wins. **Consequence to document explicitly in the thesis methods section:** M1 is therefore NOT a strict replication of Long et al.'s QCQ-CNN. It differs in batch size (32 vs 1), gradient path (autograd on an exact simulation vs parameter-shift -- numerically identical, see Sec 10), early stopping with warmup (vs flat 100 epochs), and seed count (5 vs 10). M1 should be described as a **re-implementation under a harmonised protocol**, serving as the controlled baseline for the attribution comparisons -- not as a numerical reproduction of the published result. Do not claim replication of Long's 99.4% figure unless a separate faithful-protocol run is added later. **Open risk:** if the M1 polarity inversion turns out to be driven by the coarser plateau traversal at bs=32 (160-240 updates through the plateau vs 5,000-7,500 at bs=1), the fix must come from somewhere other than batch size -- candidate levers are longer warmup, a lower LR during warmup, or fc3 initialisation, since fc3 = Linear(1,2) is the layer free to commit to either output polarity. |

| 2026-07-28 (**RESOLVED — dropout is the cause, proved from the code**) | **`DROPOUT_RATE` locked to 0.0 for all four models** (was 0.5, Long et al.'s value). The M1/M2 global decision inversion is caused by dropout. Established by **elimination from the source, not by experiment** -- the empirical A/B was inconclusive because neither arm happened to invert, and an earlier entry wrongly recorded the hypothesis as "refuted" on that basis. The argument: (1) train and val are drawn from ONE balanced pool, shuffled by `rng.permutation` and then sliced (`load_dataset`) -- same distribution, no overlap, identical label mapping; (2) `run_epoch` derives loss and accuracy from the same `y`, and one-hot BCE agrees with `argmax` on polarity, so no flip can originate there; (3) the ONLY difference between how `train_acc` and `val_acc` are measured is `model.train(True)` vs `model.train(False)`; (4) `Dropout2d` is the ONLY module in the entire architecture whose behaviour depends on that flag -- there is no BatchNorm or LayerNorm anywhere. A model at ~0.96 train (dropout on) and ~0.04 val (dropout off) on same-distribution data therefore admits exactly one explanation. **Why M1/M2 only:** the QNN back-end funnels the whole network through ONE scalar (`64 -> fc2 -> 2 -> QNN -> 1 -> fc3 -> 2`), making the decision a threshold on a single number; dropout shifts that number's distribution between modes and, with no redundancy to absorb it, the threshold flips for every sample at once. M3/M4's `Linear(64->2)` head is linear and unbottlenecked, so `E[f(x)] = f(E[x])` holds exactly and they never invert. Trajectory-dependence follows: the flip occurs only when the shift crosses `fc3`'s learned threshold. **Applied to all four models**, not selectively, because Sec 5's attribution requires an identical backbone; dropout has zero trainable parameters so no counts change. **Documented deviation from Long et al.** -- state it in the methods section alongside the batch-size deviation. |
| 2026-07-28 (minor, non-blocking) | `(N_TRAIN + N_VAL) // 2` in `load_dataset` truncates when the sum is odd, so the val split can come up one short. At smoke scale `(500+125)//2 = 312` gives a 624-sample pool and therefore **124 val samples, not 125** -- which is why smoke-run val accuracies are multiples of 1/124. The locked full protocol is unaffected: `(5000+1250)//2 = 3125` gives exactly 6250, splitting cleanly into 5000/1250. Left as-is; only relevant when interpreting reduced-scale numbers. |

| 2026-07-28 (**Circuit 4 has a structurally dead parameter**) | **w[5] has exactly zero effect on the front-end output.** Circuit 4's final gate is `CRZ`, diagonal in the computational basis, followed immediately by `PauliZ` measurement, also diagonal -- so its phases cancel in `|psi|^2` and cannot alter the measurement statistics. Verified numerically: perturbing `w[5]` by 1.0 rad changes the output by 5.6e-16, and its gradient is exactly zero under both autograd and the four-term parameter-shift rule. Surfaced by the new per-epoch `[GRADVAR]` logging, which reported per-parameter variance of exactly 0.0 for that index. **This is not an implementation error** -- Circuit 4 is faithfully reproduced gate-for-gate against Hur et al. Fig. 2d. In Hur et al.'s QCNN the ansatz is followed by further layers, so the final CRZ does real work; it is the front-end role (ansatz then immediate Z measurement) that makes it inert. **Consequences:** (a) M2's front-end has **5 effective parameters, not 6** -- M2 = 1,609 effective / 1,610 nominal, M3 = 1,599 / 1,600; (b) the iso-parametric constraint still holds, `|1609-1594|/1609 = 0.93%`, inside Thesis A Sec 3.3's 1% bound; (c) the barren-plateau gradient-variance analysis (Thesis A Sec 5.1) **must exclude w[5] or flag its zero variance as structural**, otherwise it reads as false plateau evidence; (d) Thesis A Sec 3.2.2's "reduces the number of independent parameters to six" needs a footnote. **Not changed** -- measuring in X or appending a final RY would activate it, but fidelity to the published ansatz outweighs one extra parameter. Supervisor decision if it's ever revisited. |

| 2026-07-28 (**QNN back-end now GPU-native — last CPU bottleneck removed**) | M1/M2 remained 30-85x slower than M3/M4 after the front-end fix (8.5 s/epoch vs 0.1-0.3 s/epoch at n=500). The entire gap was qiskit's `EstimatorQNN`: it is a pure-Python/numpy reference implementation with **no CUDA code path at any circuit size**, and `TorchConnector` round-trips `tensor -> numpy -> parameter-bind -> simulate -> numpy -> tensor` on every call. With `input_gradients=True` (mandatory -- without it the shared CNN backbone gets no gradient) each backward also evaluates shift terms for the 2 inputs on top of the 6 weights, so at batch 32 that is ~500 sequential circuit simulations per step, ~17,000 per epoch, on one CPU core. **Why the obvious fixes fail:** Aer with `device='GPU'`/cuStateVec is pointless on a 2-QUBIT circuit (4 amplitudes; kernel-launch overhead dwarfs the work -- the same reason `lightning.gpu` failed for the 4-qubit front-end); batching Estimator calls helps constant factors but stays on CPU; dropping `input_gradients` reintroduces the M1 near-random-accuracy bug. **Fix:** `qnn_torch.py` simulates the identical circuit as batched torch tensor algebra. The state is 2 qubits = 4 complex amplitudes = a `(B,4)` tensor; fixed gates (H, CX, RY) are 4x4 matmuls over the whole minibatch, and -- the key enabling fact -- **every input-dependent gate in `ZZFeatureMap` is DIAGONAL**, so per-sample phase rotations are an elementwise multiply rather than per-sample matrix construction. Follows Qiskit's little-endian qubit ordering. **Verified against the real `EstimatorQNN`** (`scripts/verify_qnn_torch.py`, exact pinned stack): forward 1.0e-06 rel, weight gradients 9.7e-07 rel, **input gradients 2.9e-06 rel** -- all at float32 precision, which is `TorchConnector`'s internal dtype. **Measured: 181x faster at batch 32, 680x at batch 128** (the gap widens because torch does one matmul while qiskit scales per-sample); projected back-end cost at full scale drops from ~412 s/epoch to ~0.61 s/epoch. Selected with `--qnn_backend torch` (now default) / `qiskit`. Parameter counts unchanged (M1 1604, M2 1610). |
| 2026-07-28 (**noise phase constraint — do not lose this**) | The torch fast paths (`quanv_torch.py`, `qnn_torch.py`) simulate **ideal circuits only**. They cannot express Aer's `NoiseModel`, so **Aim 3 (the IBM-calibrated noise sweep, alpha > 0) must run with `--qnn_backend qiskit`** and, for the front-end, the PennyLane path. All the measured speedups above therefore apply to the **alpha = 0 column only**, which is 1 of 6 noise levels — i.e. the ideal column is now essentially free, and the remaining 5/6 of the experimental matrix carries the original cost. Budget the noise phase against the ORIGINAL timings (~412 s/epoch back-end at full scale, plus higher shot counts under noise, which Thesis A Sec 3.4.3 notes are needed to control gradient-estimator variance). Density-matrix simulation of a 2-qubit circuit is 16 amplitudes and could in principle be added to the torch path later; that is a genuine option for the noise phase but is NOT implemented and should not be assumed in planning. |

| 2026-07-28 (correction to an earlier overstatement) | An earlier assessment described the scale-1000 results as showing "clear overfitting" that "will matter more at 5,000 samples". **Both claims were overstated.** A train=1.00 / val=0.97 gap is modest, and the gap should SHRINK at full scale rather than grow, since 1,600 parameters against 5,000 samples is more underparameterised than against 1,000. Thesis A Sec 5.3's overfitting concern was specifically about M4 being **padded** with extra classical capacity to match M2's count -- but in this implementation M4 is 1,594 vs M2's 1,610, so it was never meaningfully padded and the confound the mitigation targets is largely absent. Weight decay was still added (it is a protocol commitment) but should not be presented as fixing a problem. |
| 2026-07-28 (**gradient-variance logging is NOT barren-plateau evidence — correction**) | The `fe_gradvar` metric added this session logs variance of the front-end gradient **across minibatch steps within an epoch**. That is minibatch gradient noise, and it falls naturally as training approaches a minimum -- M3's decay from 6.7e-04 to 1.1e-05 over 24 epochs is ordinary convergence. **It does not test for a barren plateau and must not be reported as if it does.** A barren plateau is the variance of the gradient over **random parameter initialisations**, measured as a function of **qubit count**, with exponential decay in n_qubits as the signature. Nothing in the per-epoch curves addresses that. Thesis A Sec 5.1 names this measurement as a deliverable, so it needs a dedicated script: sample many random inits of the Circuit 4 front-end, compute the gradient for each, take the variance, and repeat at 4/6/8/10 qubits to obtain the scaling curve. Cheap now that the front-end is torch-native; no training required. **Keep `fe_gradvar` -- it is useful as a convergence diagnostic -- but label it as such.** |

**When you make a future correction:** add a row here with the date and what changed, so the next session (or the next version of you) can see the provenance instead of re-discovering the same bug.

---

## 10. Front-End Evaluation Backend (added 2026-07-28, v5)

| | |
|---|---|
| **Production path** | `quanv_torch.py` — the circuits computed as exact batched torch tensor algebra, on the GPU, gradients via autograd. This is what every result from v5 onward uses. |
| **Reference path** | `pennylane_frontends_legacy.py` — the original PennyLane/`TorchLayer`/parameter-shift front-ends, reachable via `--backend pennylane`. Equivalence checking only; ~282× slower. |
| **Proof of equivalence** | `scripts/equivalence_check.py` (torch vs PennyLane, forward + gradient, both front-ends) and `scripts/verify_torch_quanv.py` (torch vs a brute-force 4-qubit reference built from explicit Kronecker products). Run both on Gadi once and keep the output — it is the methods-section evidence. |

**The circuits themselves are unchanged.** Circuit 4's gate sequence (Sec 3) is implemented gate for gate in `TrainableQuanvFrontEndTorch._block_unitary()`; M1's `RandomLayers` unitary is extracted once via `qml.matrix()` so its semantics are PennyLane's by construction. Only the *evaluation strategy* changed.

**Why the front-end is only a 2-qubit problem.** The RY encoding produces a product state, and `circuit4_block(0,1)` and `circuit4_block(2,3)` share no wires. The 4-qubit state therefore remains an exact tensor product of two 2-qubit states throughout, and ⟨Z₀⟩,⟨Z₁⟩ depend only on block A while ⟨Z₂⟩,⟨Z₃⟩ depend only on block B. The whole front-end is two independent dim-4 simulations — a batched 4×4 complex matmul, not a dim-16 one. This is worth a sentence in the thesis: it is a genuine structural property of the ansatz-per-patch design, not an implementation trick.

**What this does NOT change.** Parameter-shift remains required for the noisy/hardware phase, where it is a real hardware-faithfulness constraint rather than a simulation choice. Do not delete the reference path.
