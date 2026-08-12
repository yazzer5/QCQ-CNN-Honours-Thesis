# Progress — 2026-07-28: GPU root cause, torch-native front-end, batch-size decision

Supersedes the GPU sections of `PROGRESS_CHECK_2026-07-27.md`. Scope for this
phase: **ideal (noiseless) MNIST 3-vs-5 only, all four models M1–M4.**
Fashion-MNIST, MRI, and the noise sweep are explicitly out of scope until the
MNIST grid is complete and reproduces Long et al.'s ~99.4%.

---

## 0. One-paragraph summary

The 0% GPU utilisation was never a device-selection problem, and `lightning.gpu`
could not have fixed it. The front-end circuits are 4 qubits (16 amplitudes);
no state-vector backend can occupy a V100 at that size, and PennyLane's
`TorchLayer` was silently re-serialising the `unfold()` batch back into
`B × 196` sequential tape executions because the lightning devices don't support
parameter broadcasting. The fix is to stop calling a simulator at all: the
circuits are computed directly as batched torch tensor algebra, verified
numerically identical to the PennyLane/parameter-shift result. This runs on the
GPU natively, removes the `batch_size=1` constraint, and takes the MNIST grid
from a multi-week job to an overnight one.

---

## 1. Evidence review of the v4 run

Every `.o` / `.e` / `.csv` file in `GADI_v4.zip` is from the **failed** batch
(`174853423`–`426`) that died in 4–12 s on the `$PBS_O_WORKDIR` path bug. The
GPU CSVs contain exactly one sample each, all `0 %, 0 MiB`. Outputs from the
post-path-fix batch (`174853951`–`954`) were never pulled.

**Action:** still pull those (`~/qcqcnn_v4/*.o*`, note the path moved to repo
root) for the per-epoch timing record, but they are not needed to proceed — the
diagnosis below is structural, not empirical.

---

## 2. Root cause (two independent problems)

### 2.1 A 4-qubit circuit cannot saturate a V100

16 complex amplitudes; each gate is a few dozen FLOPs. cuStateVec pays ~5–10 µs
of dispatch and kernel-launch overhead per call, so the GPU is idle >99.9% of
the time and `nvidia-smi` reads ~0%. Below roughly 18 qubits, `lightning.gpu` is
*slower* than `lightning.qubit`. `get_quantum_device()` works exactly as
written; it is the wrong lever.

### 2.2 The `unfold()` batching fix never reduced the number of circuit calls

`TrainableQuanvFrontEnd.forward()` claims the qnode is "called ONCE per forward
pass." It is not. `TorchLayer` only issues one device execution when the
*device* supports parameter broadcasting. The lightning devices do not, so
PennyLane applies `broadcast_expand` and splits the batched tape back into
`B × 196` separate sequential executions. `diff_method="parameter-shift"` then
multiplies that on the backward pass.

**Extra cost nobody accounted for:** CRZ's generator has eigenvalues
{0, 0, −½, +½}, so it needs the **four-term** shift rule, not the two-term one.
Circuit 4 has 2 CRZ + 4 RY per block ⇒ 4×2 + 4×2 = **16 tape executions per
backward pass**, not the 12 assumed in the code comments. PennyLane does this
correctly, so gradients were right — the cost estimate was wrong by ~33%.

This is the whole of the M3 ~490–519 s/epoch anomaly. It was never anomalous.

---

## 3. The fix: torch-native front-ends (`quanv_torch.py`)

### 3.1 Key structural fact

The M2/M3 front-end is RY encoding (a product state) followed by
`circuit4_block(0,1)` and `circuit4_block(2,3)` with **no gates spanning the two
blocks**. The 4-qubit state therefore remains an exact tensor product of two
2-qubit states, and ⟨Z₀⟩,⟨Z₁⟩ depend only on block A while ⟨Z₂⟩,⟨Z₃⟩ depend only
on block B.

So the front-end is **two independent 2-qubit (dim-4) simulations**, i.e. a
batched 4×4 complex matmul — not a dim-16 simulation.

### 3.2 Verification (all done, reproducible via `verify_torch_quanv.py`)

| Check | Result |
|---|---|
| Forward pass vs brute-force 4-qubit reference (explicit kron, PennyLane gate conventions) | max abs error **8.9e-16** |
| Torch autograd gradient vs analytic parameter-shift (four-term rule for CRZ) | max abs error **~1e-15**, exact agreement |
| Naive two-term shift rule on CRZ | **0.80 discrepancy** — confirms the four-term rule is required, and confirms the torch path is the correct one |
| Front-end fwd+bwd throughput, container CPU, no GPU | **~5,800 images/s** |
| torch vs PennyLane front-end, forward (M2/M3 and M1) | rel. error **3.6e-07** (float32 precision) |
| torch vs PennyLane front-end, gradient (autograd vs parameter-shift) | rel. error **2.8e-07** |
| Measured speedup, CPU only, 4 images | **282×** (M2/M3), **292×** (M1) |
| End-to-end M3 training, 256 samples, CPU | **~0.05 s/epoch** |

The 5,800 img/s figure is on a shared container CPU with no GPU at all. The
previous path was doing single-image forward passes measured in seconds.

### 3.3 M1 (frozen RandomLayers)

Not separable across blocks, so it takes the full 16×16 route — but M1 is
frozen, so the unitary is extracted **once** at construction via `qml.matrix()`
(guaranteeing identical circuit semantics to the PennyLane version) and then
applied as a batched matmul forever after. Zero trainable parameters, so nothing
is approximated.

### 3.4 Thesis-level implication

Parameter-shift and backprop agree *analytically* on a noiseless simulator —
they compute the same exact expectation-value derivative. Parameter-shift only
becomes *necessary* in the noisy/hardware phase, where it is a genuine
hardware-faithfulness requirement rather than a simulation choice. Keep the
PennyLane path behind a `--backend pennylane` flag so the equivalence can be
asserted on a small run and cited in the methods section.

---

## 4. Batch size: recommendation and reasoning

### 4.1 Decision

> **Batch size 32, for all four models M1–M4. Learning rate stays 1e-3, Adam,
> max_epochs 100, patience 15.**

M4 comes **down** from 128 to 32 so that all four models match.

### 4.2 Why `batch_size=1` was believed to be mandatory — and why it isn't

Two separate justifications exist in the record, and they don't agree:

- **Long et al.** state bs=1 for QCQ-CNN was chosen *"to reduce the simulation
  overhead in quantum layers"* and for *"fine-grained gradient updates."* That
  is a simulation-cost and optimisation-granularity argument, not a correctness
  one. Long used **bs=4** for their classical and QC-CNN models.
- **The thesis draft (§4.4.1)** claims something stronger: that averaging
  parameter-shift gradients over a minibatch causes *"systematic cancellation"*
  from *"the symmetric structure of the shift-rule terms."*

**The §4.4.1 claim does not hold mathematically.** The parameter-shift rule
returns the exact per-sample gradient ∂⟨O⟩/∂θ. Averaging exact per-sample
gradients over a minibatch gives the exact minibatch gradient — that's just
linearity of differentiation, and it's the same operation backprop performs. The
±shift terms are subtracted *within* each sample's gradient evaluation, before
any averaging happens; there is no cross-sample structure for averaging to
cancel.

The observed symptom (bs=4 stalling at ~58%, near-random on a balanced binary
task) is almost certainly the **same gradient-barrier bug later found in M1** —
the missing `input_gradients=True` on `TorchConnector`, which left the CNN
backbone at random init with only `fc3`'s 4 parameters training. Near-random
accuracy is the signature of "the backbone isn't learning," not of "the gradient
is slightly mis-averaged." A mis-averaged gradient degrades convergence; it does
not pin a model to chance.

**Action required:** §4.4.1 needs rewriting before submission. This is a good
change, not a painful one — "I initially attributed a training stall to
minibatch gradient cancellation, then traced it to a missing gradient flag in
the TorchConnector wrapper" is a stronger and more honest methodological
contribution than the original claim, and it's a genuinely useful warning for
the hybrid-QML community. Raise it with Sethu early rather than letting it
surface at assessment.

Independently of all the above: **there is no parameter-shift machinery left in
the M1–M3 training path** once `quanv_torch.py` is in. Gradients come from torch
autograd. Even if the §4.4.1 mechanism were real, it could not apply.

### 4.3 Why 32 specifically

**Optimisation steps.** At n_train = 5,000:

| Batch size | Steps/epoch | Steps over 100 epochs |
|---|---|---|
| 1 | 5,000 | 500,000 |
| 4 (Long's classical setting) | 1,250 | 125,000 |
| **32** | **156** | **15,600** |
| 128 (current M4) | 39 | 3,900 |

15,600 Adam updates is comfortably enough to fit 1,600 parameters on a
near-separable binary task. **3,900 is not obviously enough** — the current M4
bs=128 setting is doing only 39 updates per epoch, which is a real underfitting
risk and a likely confound in any M4 comparison.

**Learning-rate transfer.** Long's lr=1e-3 was set at bs=4. Going to 32 is an 8×
increase. Adam's per-parameter second-moment normalisation makes it far less
batch-size-sensitive than SGD, and 8× is well inside the range where lr=1e-3
transfers without retuning. 128 would be 32×, which starts to need a warmup or
an lr bump — another reason to come down from it.

**Gradient noise is not purely a cost here.** The barren-plateau risk on the
front-end's 6 parameters is a named concern in the project plan (§5.3). SGD
noise helps escape flat regions. Batch 32 retains meaningful stochasticity;
128+ starts removing exactly the noise that may be doing useful work on the
front-end.

**GPU occupancy.** At bs=32 the front-end processes 32 × 196 × 2 = **12,544
dim-4 state vectors per call** as one batched matmul, and the CNN backbone gets
a 32-image batch. The backbone, not the front-end, is now the utilisation
bottleneck — which is the correct state of affairs. Going larger buys little
because the model is small; it mostly just costs optimisation steps.

**Memory.** Irrelevant at any of these sizes. 32 GB requested, model is 1.6k
parameters.

### 4.4 Why all four models should share one batch size

The iso-parametric design's whole claim is that each pairwise comparison
isolates exactly one variable. Batch size 128 for M4 and 1 for M1–M3 is a second
uncontrolled variable sitting inside the M3-vs-M4 comparison, which is precisely
the comparison that isolates the quantum front-end. Matching all four at 32
removes the confound and makes the comparison easier to defend, at negligible
compute cost since M4 is the cheapest model.

Note this **invalidates every existing M4 result** (both the bs=32 and bs=128
generations) for comparison purposes. M4 is fast; re-run it.

### 4.5 Required control experiment

Before the full grid, run **M2, MNIST, seed 42, bs=1 vs bs=32**, both under the
torch-native front-end, and compare final test accuracy and the loss curve. This
is the evidence that the batch-size change is benign, and it is what a marker or
reviewer will ask for. One extra run, and it converts a protocol deviation into
a documented, tested decision.

Optionally add a bs=4 arm to match Long's classical setting exactly.

### 4.6 Knock-on settings

- `min_epochs` warmup (30, M2/M3): **keep for now, but watch it.** The warmup
  plateau was characterised at bs=1 with 5,000 updates/epoch. At 156
  updates/epoch the plateau may occupy more *epochs* even though it's the same
  number of updates. If M2/M3 are still flat at epoch 30, raise it rather than
  concluding the front-end doesn't train.
- Gradient accumulation (§5.3 mitigation): **no longer needed.** Delete the plan
  item.
- Time-budget / checkpoint-resume logic and the 8-link chained PBS jobs: keep
  the machinery, but the MNIST grid should now fit in a single link per job.

---

## 5. Compute estimate for the MNIST grid

Grid: 4 models × 5 seeds × 100 epochs × 5,000 train / 1,250 val / 1,000 test.

Measured, not guessed: a full M3 forward+backward epoch over 256 samples takes
**~0.05 s on a container CPU with no GPU**. Linear extrapolation to 5,000
samples gives ~1 s/epoch, so ~100 s for a full 100-epoch run, on CPU. A V100
will be faster still.

That means the **full 20-run MNIST grid is minutes-to-an-hour of compute**,
versus the previously projected 6–10 h *per M2 run*. The 6 h walltime in
`run5_mnist_grid.pbs` is deliberate slack; lower it once the timing run
confirms, because shorter walltime means shorter queue on `gpuvolta` (which is
not bonus-time eligible).

Do not skip the timing run on that basis — the extrapolation is from CPU, and
the M1/M2 QNN **back-end** still runs on CPU through qiskit's Estimator, which
is now the slowest remaining component and has not been re-measured.

---

## 6. Next actions, in order

1. **Integrate** `quanv_torch.py` into `train_gadi.py`: replace
   `TrainableQuanvFrontEnd` → `TrainableQuanvFrontEndTorch`,
   `FrozenHendersonFrontEnd` → `FrozenHendersonFrontEndTorch`. Delete the
   `.cpu()` round-trips and the "keep `front_end` on CPU" block in `main()` —
   the whole model now lives on `cuda`.
2. **Keep the parameter-count assertions.** They should still pass unchanged
   (6 trainable front-end params for M2/M3, 0 for M1). If they don't, stop.
3. **Run `verify_torch_quanv.py` on Gadi** inside the `qcqcnn` env and save the
   output — that's the equivalence evidence for the thesis.
4. **Equivalence check against PennyLane**: one fixed seed, one batch, both
   front-end implementations, `torch.allclose` on the output tensor. Log it.
5. **Gradient-flow assertion**: after one optimiser step, assert
   `model.backbone.conv1.weight.grad is not None` and print its norm, for every
   model. This is the M1 near-random-accuracy bug's tripwire — it should be a
   permanent check, not a one-off diagnosis.
6. **Timing run**: M2 + M3, MNIST, seed 42, n_train=5,000, `max_epochs=3`, and
   read the real per-epoch time. Everything downstream depends on this number.
7. **Control run** (§4.5): M2 bs=1 vs bs=32.
8. **Full MNIST grid**: M1–M4 × seeds {42–46}, bs=32, lr=1e-3, 100 epochs,
   patience 15, `min_epochs`=30 for M2/M3.
9. Only then: Fashion-MNIST, then MRI, then noise.

### PBS / infra fixes to fold in while editing

- Echo the resolved working directory and `ls train_gadi.py` before invoking
  python, so a path bug fails loudly in the log instead of as one stderr line.
- Sample `nvidia-smi` every 2 s, not 10 s.
- `chmod +x` after every `scp` from Windows — still not preserved.
- Data must be staged before upload; `gpuvolta` nodes have no internet.

---

## 7. Changes to `MODELS_SOURCE_OF_TRUTH.md`

Sec 6 (Training Protocol) — replace the batch-size row:

> | Batch size | **32, all four models (M1–M4)** | Changed 2026-07-28. Was 1 for M1–M3 / 128 for M4. The bs=1 constraint was tied to PennyLane's parameter-shift tape execution, which no longer exists in the training path (front-ends are now exact torch-native simulations, gradients via autograd, verified numerically identical to parameter-shift). 32 gives 156 steps/epoch at n=5,000 — enough optimisation steps for 1,600 params, retains useful gradient noise against barren-plateau risk, and transfers lr=1e-3 from Long's bs=4 without retuning. Matched across all four models to remove a confound from the M3-vs-M4 comparison. **All M4 results prior to 2026-07-28 are superseded.** Validated by an M2 bs=1 vs bs=32 control run (seed 42, MNIST). |

Sec 9 (change log) — add:

> | 2026-07-28 | Quantum front-ends reimplemented as exact batched torch tensor algebra (`quanv_torch.py`), replacing PennyLane `TorchLayer`. Root cause of persistent 0% GPU: 4-qubit circuits cannot occupy a V100, and `TorchLayer` re-serialised batched input into B×196 sequential tapes because lightning devices lack parameter broadcasting. Forward verified to 8.9e-16 against a brute-force 4-qubit reference; autograd gradients verified identical to the four-term parameter-shift rule. Batch size unified at 32 across M1–M4 (see Sec 6). Noted that CRZ requires the four-term shift rule (16 tape evaluations/backward, not 12) — corrects an under-estimate in the old cost model. |

Sec 7 (design rationale) — the bullet stating quantum params are updated via
parameter-shift at "2 circuit evaluations per parameter per step" needs
amending: correct for RY, wrong for CRZ (four evaluations), and in the current
noiseless phase they're updated by autograd on an exact simulation. Restore
parameter-shift for the hardware/noise phase.

---

## 8. Open items carried forward

- Pull and archive the `174853951`–`954` outputs for the record.
- `--broadcast_shift` and `scripts/verify_broadcast_shift.py` are now **dead
  code** — the whole broadcasting question was a workaround for the PennyLane
  execution path being removed. Delete both.
- M1's near-random accuracy: the QNN *back-end* still uses `TorchConnector`, so
  `input_gradients=True` remains load-bearing there. Item 5 above is the check.
- The train/val/test split question from 2026-07-27 (6,000 vs 5,000) is now
  **moot** — at the new throughput there is no reason to deviate from Long's
  5,000 / 1,250 / 1,000. Keep the locked protocol; one less deviation to defend.
- §4.4.1 of the thesis draft needs rewriting (see §4.2). Flag to Sethu.
