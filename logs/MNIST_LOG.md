# MNIST LOG — MNIST phase (3-vs-5 grid and 10-class diagnostic)

**Student:** Yasser (z5477389) · **Supervisor:** A/Prof Vidhyasaharan Sethu
**Thesis:** End-to-End Trainable Quantum-Classical-Quantum CNNs for Image Classification
**Code version:** v6 (grid) / v7 (diagnostic) · **Last updated:** 2026-08-03

Companion document: `FMNIST_LOG.md` (Fashion-MNIST phase, current work).

---

## 1. The four models

Iso-parametric by design — all within 0.93% of 1,600 parameters, so accuracy
differences cannot be attributed to capacity.

| Model | Front-end | Back-end | Params | Role |
|---|---|---|---|---|
| M1 | frozen Henderson quanvolution | QNN head | 1,604 | replication anchor |
| M2 | trainable Circuit 4 | QNN head | 1,610 | **core contribution** |
| M3 | trainable Circuit 4 | classical head | 1,600 | isolates the quantum head |
| M4 | bilinear resize (no quantum) | classical head | 1,594 | fully classical control |

Shared backbone: `Conv2d(4,8,3)/ReLU/MaxPool → Conv2d(8,16,3)/ReLU/MaxPool →
Dropout2d → Flatten` → 64-dim (1,464 params).
QNN head: `ZZFeatureMap(2) + RealAmplitudes(2, reps=2)` → 6 variational params,
`Linear(64,2)` = 130, `Linear(1,2)` = 4 → 140 total.

The three adjacent contrasts isolate one factor each:

- **M2 − M1** → front-end trainability (trainable Circuit 4 vs frozen random)
- **M2 − M3** → the quantum back-end (QNN head vs classical head)
- **M3 − M4** → the quantum front-end (Circuit 4 vs bilinear resize)

---

## 2. MNIST 3-vs-5 grid, α = 0 — complete (20/20)

4 models × 5 seeds {42–46}, splits 5000/1250/1000. No failures, no inversions.

| Model | Test acc | SD |
|---|---|---|
| M1 | 99.22% | 0.26 |
| M2 | 99.18% | 0.23 |
| M3 | 99.18% | 0.13 |
| M4 | 99.14% | 0.21 |

All three attribution contrasts null, **p > 0.57**. Total spread across all four
architectures: **0.08 pp**.

### This is a ceiling effect, not a result

At 99.2% there is ~0.8 pp of headroom, while the design resolves ~0.25–0.40 pp.
With 1000 test images, one image is 0.1 pp. When total spread is smaller than
measurement resolution, the experiment cannot distinguish the architectures
**whether or not a difference exists**.

The decisive detail: **M4 contains no quantum components at all and is
statistically indistinguishable from every quantum variant.** If the task cannot
separate a fully classical model from a quantum one, it cannot evaluate the
research question.

**Do not present this as evidence that quantum layers do not help.** The correct
statement is that the task is saturated and the measurement uninformative — which
is what motivated the Fashion-MNIST phase.

### Why this chapter still matters

It is the control that explains why the literature reports success on
MNIST-class tasks: at saturation, architectural differences are invisible. That
is a direct, evidenced comment on the evaluation regime in Long et al. (2025).

---

## 3. Three silent correctness defects found during v6 validation

All three would have produced plausible-looking but wrong results. Found by
reduced-scale validation (500 → 1000 → 3000 samples) before committing to the
full grid.

**Defect 04 — early stopping fired at epoch 1.** The patience counter
incremented unconditionally from epoch 1; `min_epochs` gated only the stop
*check*, not the counter. Every run would have terminated returning the
degenerate epoch-1 state, producing a full grid of ~50% results that looked like
a legitimate architectural finding.

**Defect 06 — `Dropout2d(0.5)` globally inverted M1/M2.** Train 96% / val 4%.
Cause: the QNN head's single-scalar bottleneck means the decision threshold
flips for *every sample simultaneously* under train/eval mode switching.
Established by code elimination; fixed by setting dropout to 0.0 uniformly. This
is the first appearance of the single-scalar bottleneck as a real mechanism —
see §4.

**Defect 07 — warm-up keyed to the wrong component.** The condition was tied to
trainable front-ends (M2/M3) when the plateau actually belongs to the QNN
back-end. M1 has a frozen front-end and plateaus; M3 has a trainable front-end
and does not.

**Related:** M1's near-random accuracy in early runs was a gradient barrier at
`TorchConnector` when `input_gradients=True` silently fails — only `fc3`
(4 params) trained.

**Also established:** the QNN back-end plateaus for ~180 optimiser updates
regardless of dataset size, confirmed at four scales.

---

## 4. 10-class MNIST diagnostic (2026-08-03, Gadi job 175317142)

Run to answer a supervisor question: would multiclass give more headroom than
harder binary pairs?

### Status: DIAGNOSTIC, not a thesis experiment

Three caveats that must accompany any number from it:

1. **Not iso-parametric.** Ten classes forces the output layers to widen
   unequally — M1/M2 back-end 156–176 params, M3/M4 650. M3/M4 carry ~480 more
   parameters, so their advantage is confounded by capacity.
2. **Single seed** (42). No error bars, no significance tests.
3. Training protocol otherwise **matches the binary grids exactly**
   (5000/1250/1000, batch 32, lr 1e-3, wd 1e-4, dropout 0.0, max 100 epochs,
   patience 10, min_epochs 30).

### Results

| model | readout | test acc | distinct classes predicted | params |
|---|---|---|---|---|
| M1 | 1 observable | 33.50% | 7/10 | 1,620 |
| M2 | 1 observable | 29.60% | 8/10 | 1,626 |
| M1 | 3 observables | 66.50% | 10/10 | 1,640 |
| M2 | 3 observables | 72.90%† | 10/10 | 1,646 |
| M3 | classical head | 95.60% | 10/10 | 2,120 |
| M4 | classical head | 95.80% | 10/10 | 2,114 |

Chance = 10.00%.
† **M2 hit the 100-epoch cap without early-stopping** (best val 76.48% at epoch
94, still rising). 72.90% is a **lower bound**, not a converged value. Re-run
with `EPOCHS=200`.

### Why M1/M2 fail — two separate reasons

**Reason 1 — the readout (fixable, and demonstrated).**
With one observable the network's entire output is a single scalar in [−1,+1].
`fc3(1,10)` then produces ten logits, each affine in that one value — ten
straight lines in one variable, of which only the **upper envelope** can ever
win. Classes whose line sits below the others everywhere are **mathematically
unreachable**, which is exactly why the narrow models emitted only 7–8 of 10
classes. Even reachable classes are forced into a fixed 1-D ordering that cannot
express 10-way similarity structure.

Widening to three observables — `⟨Z0⟩`, `⟨Z1⟩`, `⟨ZZ⟩` — gives `fc3` a 3-D space
to partition. **This adds zero circuit parameters**: same 6-parameter state, same
circuit, just two more measurements off it (20 extra classical weights in
`fc3`). Accuracy doubles and all 10 classes become reachable.

**Reason 2 — the 64 → 2 compression (not fixed).**
`fc2` maps 64 features to 2 numbers before the circuit, because the circuit has
only 2 qubits to encode into. Information destroyed there cannot be recovered by
any amount of rotation or entanglement downstream. M3/M4 keep all 64 features
and feed `Linear(64,10)` directly.

The narrow → wide jump proves reason 1 was dominant. What remains is a mix of
reason 2 and the parameter confound.

### Two conclusions

**1. The multiclass route is confirmed rejected.** Not because quantum models
are bad at multiclass, but because 10-class MNIST reaches ~95–96% for the
classical control — the *same ceiling* as 3-vs-5 — after an architectural change
that breaks the iso-parametric bound. It buys no headroom and costs the
controlled comparison.

**2. The QCQ architecture is binary-shaped by construction.** Neither limitation
binds for 2 classes: one scalar thresholds two classes perfectly, and 64 → 2 is
far less lossy with only two classes to separate. This is why M1/M2 sit at ~84%
on Fashion 0,6 rather than ~30%.

### Secondary observation (weak)

M2 beat M1 by 6.4 pp on the wide readout (72.9 vs 66.5) — the first clear
advantage for the trainable front-end anywhere in this project, against +0.62 pp
(p = 0.45) on Fashion 0,6 and +0.04 pp on MNIST 3-vs-5. Plausible reading: 10
classes demand more of the front-end than a binary split does. **Hypothesis
only** — single seed, M2 was the truncated run, and the configuration is not in
the thesis spec.

---

## 4b. 10-class MNIST grid, code v10 (2026-08-13) — 20/20 clean, **reduced-scale/feasibility run**

> **Amendment 2026-08-14:** every job in this grid ran under
> `protocol.reduced_scale = true` (`N_TEST=2000, MAX_EPOCHS=200, PATIENCE=20`),
> not the locked full protocol (`N_TEST=1000, MAX_EPOCHS=100` used by the
> binary grids). This was missed in the original write-up below because it
> was drafted from fallback constants before the raw job logs were available;
> the raw JSONs and job logs now confirm it directly (0/20 `FULL PROTOCOL`,
> 20/20 `REDUCED-SCALE`). **The numbers below are unchanged and match the
> raw JSONs exactly** — only the label changes. Full account:
> `docs/AMENDMENT_2026-08-14_v10_reduced_scale.md`. Treat every finding in
> this section as a feasibility result pending a locked-protocol confirmation
> run, not as the final word on 10-class M2 vs M1.

The single-seed diagnostic in §4 (+6.4 pp for M2 over M1) motivated running
the real thing: 4 models × 5 seeds properly, not a truncated single-seed
comparison. Code v10 is v9 plus: a 3-observable QNN readout (`<Z0>`, `<Z1>`,
`<Z0Z1>`, same circuit, zero extra circuit parameters), an iso-parametric
re-pad for 10 classes (M1 1640 / M2 1646 / M3 1649 / M4 1643, spread 0.55% —
**not** the binary 1604/1610/1600/1594), `--deterministic`, and
`--save_predictions` for McNemar's test. Full changelog:
`docs/CHANGELOG_v10.md`.

**Loss function amendment, made before the grid ran:** the pre-registration
(`docs/PREREGISTRATION_MNIST10.md`) specified BCE-on-one-hot retained for
continuity with v6–v9. A smoke test triggered the convergence-at-cap
criterion described in `docs/AMENDMENT_2026-08-13_loss.md` — at 10 classes,
BCE drives 9 of 10 logits toward a strongly negative target every step, and
that pressure falls asymmetrically on M1/M2's bounded QNN outputs. The grid
switched to softmax cross-entropy before any full run was submitted. Disclose
this in the thesis: it is a genuine protocol deviation from v6–v9, made for a
documented and pre-registered reason, not a post-hoc fix.

### Results

| Model | Front-end | Head | Test acc | SD | Seed spread |
|---|---|---|---|---|---|
| M1 | frozen Henderson | QNN | 70.59% | 5.17 pp | 12.0 pp |
| M2 | trainable Circuit 4 | QNN | 70.92% | 5.40 pp | 15.0 pp |
| M3 | trainable Circuit 4 | classical | 81.96% | 2.33 pp | 6.2 pp |
| M4 | bilinear resize | classical | 82.41% | 1.98 pp | 5.0 pp |

Per-seed M1/M2 (seeds 42–46): M1 = 64.70, 74.35, 71.15, 76.70, 66.05. M2 =
77.75, 62.75, 72.55, 70.30, 71.25.

**Primary contrast, M2 − M1 = +0.33 pp, p = 0.94, SD(diff) = 9.67 pp, MDE =
16.08 pp.** Far underpowered relative to every binary grid (MDE 1.4–2.8 pp).
Per-seed differences: +13.05, −11.60, +1.40, −6.40, +5.20 — sign flips across
seeds, no consistent direction.

**QNN-head models are highly seed-unstable at 10 classes** (12–15 pp spread)
while classical-head models are not (5–6 pp). This is the primary finding of
this grid — a substantive architectural result, not noise, and it is the
reason the primary contrast is unresolvable at 5 seeds regardless of the true
effect size.

**Classical head beats QNN head by 11.4 pp** (Welch p = 2.2×10⁻⁵) — the same
grouped-contrast pattern seen at every Fashion-MNIST pair, now much larger.
**M3 − M4 = −0.45 pp, MDE = 1.38 pp** — null; the quantum front-end
contributes nothing detectable at this power level, reversing the sign of
the one nominally-significant attribution contrast in the project (+1.24 pp
at Fashion 2,4, itself flagged as marginal there).

**The diagnostic that motivated this grid does not hold up.** The +6.4 pp
single-seed result from §4 sits well inside the ±13 pp range observed once
all five seeds ran — very likely noise, not signal. This is the second time
in the project a single-seed or single-configuration result motivated a
follow-up experiment that then failed to reproduce at proper power (the
first being the withdrawn gap metric, `FMNIST_LOG.md` §6.1); worth naming as
a pattern when writing up methodology.

### Pre-registered predictions — scored

Full scoring with commentary in `docs/PREREGISTRATION_MNIST10.md` §Outcome.
Summary: of 5 predictions, 1 falsified outright (M2−M1 magnitude/significance),
1 falsified on sign (M3−M4), 1 mostly held (accuracy range), 1 held (McNemar
sign agreement, see below), 1 not yet evaluated (determinism check — see
§Data provenance note).

### McNemar's test — now computed (raw JSONs arrived 2026-08-14)

Using `test_predictions`/`test_labels` from the raw result JSONs (`--save_predictions`
was enabled for every v10 run, confirmed in the job logs).

**Pooled across all 5 seeds (n = 10,000 image-level predictions per model):**
M1 70.59%, M2 70.92%, M2-only-correct 1,632 vs M1-only-correct 1,599,
McNemar χ² = 0.317, p = 0.574. **Null, same direction as the accuracy-mean
t-test (+0.33 pp, p = 0.94) — prediction 2 holds:** the two tests agree in
sign, so there is no discrepancy to chase down.

**Per-seed McNemar tells a different story than the pooled test:**

| seed | M1 acc | M2 acc | M1-only correct | M2-only correct | χ² | p |
|---|---|---|---|---|---|---|
| 42 | 64.70% | 77.75% | 182 | 443 | 108.16 | <0.0001 (M2 wins) |
| 43 | 74.35% | 62.75% | 474 | 242 | 74.53 | <0.0001 (M1 wins) |
| 44 | 71.15% | 72.55% | 332 | 360 | 1.05 | 0.305 (ns) |
| 45 | 76.70% | 70.30% | 321 | 193 | 31.38 | <0.0001 (M1 wins) |
| 46 | 66.05% | 71.25% | 290 | 394 | 15.51 | 0.0001 (M2 wins) |

**Four of five seeds show a highly significant per-image difference between
M1 and M2 — this is a much more sensitive test than the 5-point paired
t-test on aggregate accuracy, because it uses all 2,000 test images per seed
instead of one number per seed.** But the sign flips: M2 wins decisively at
seeds 42/46, M1 wins decisively at 43/45, and seed 44 is the one genuine
near-tie. Pooling across seeds cancels these out and produces the misleadingly
quiet p = 0.574 above.

**Read together, this sharpens rather than resolves the central question.**
The seed-to-seed accuracy swings in §4b are not measurement noise around a
stable small effect — McNemar shows each seed's M1-vs-M2 difference is itself
highly non-random, just non-random in *opposite directions* depending on the
seed. This looks less like "front-end trainability has a small, hard-to-detect
effect" and more like "which architecture wins depends on some seed-coupled
factor (initialisation, optimisation trajectory) that this design cannot yet
identify." Worth flagging to the supervisor as a genuinely new framing, not
just an underpowered version of the original question — and worth checking
whether a similar per-seed McNemar split shows up in the binary Fashion-MNIST
grids (§5.7 note in `FMNIST_LOG.md`, where predictions weren't saved).

### Data provenance note

This section, `docs/PREREGISTRATION_MNIST10.md`'s outcome, and
`results/results_v10_mnist10/` were originally built (2026-08-14, earlier
same day) from code, docs, and four summary figures — **not from raw
`result_M*.json` files**, which were not included in that upload. Raw
per-run JSONs and job logs arrived later the same day and have now been
added directly (`results/results_v10_mnist10/result_*.json`,
`results/job_logs/v10_mnist10/`). Their per-seed test accuracies match the
previously-used fallback constants exactly, to the digit. **The job logs
also revealed the reduced-scale protocol issue — see the amendment banner at
the top of §4b and `docs/AMENDMENT_2026-08-14_v10_reduced_scale.md`.**
Determinism is confirmed enabled (`deterministic=True` printed in every job
log); a live re-run to verify bit-exact agreement at fixed seed has not been
done and remains open (prediction 5).

---

## 5. Figures

| file | content |
|---|---|
| `figA_readout_width.png` | narrow vs wide readout, plus validation trajectories |
| `figB_param_confound.png` | accuracy vs parameter count — the confound, stated honestly |

Show figB alongside figA so the ~95% classical bars are not read as evidence
against quantum layers.

**v10 grid figures (2026-08-13):** `code/v10_mnist10/figures/` and mirrored
at `results/results_v10_mnist10/analysis/plots/`, with a standalone report at
`results/results_v10_mnist10/analysis/report.html`:

| file | content |
|---|---|
| `v10_fig1_seed_spread.png` | every seed, every model — the headline figure showing QNN-head instability |
| `v10_fig2_paired_differences.png` | per-seed paired differences for each attribution contrast |
| `v10_fig3_effect_vs_mde.png` | observed effect size vs MDE for each contrast |
| `v10_fig4_head_comparison.png` | grouped QNN-head vs classical-head comparison |

---

## 6. M2 improvement candidates

Ranked by expected value. None is guaranteed to close the 1.08 pp gap to M4 on
Fashion 0,6; test one at a time, 5 seeds each, against the existing baseline.

**1. Widen the QNN readout — highest expected value, zero circuit parameters.**
Directly evidenced by §4: `⟨Z0⟩, ⟨Z1⟩, ⟨ZZ⟩` instead of `⟨ZZ⟩` alone. `fc3(3,2)`
costs 4 extra params, trivially re-paddable to preserve the iso-parametric
bound.
*Caveat:* for 2 classes one scalar is already sufficient, so the gain may be much
smaller than the 10-class result suggests. Worth testing, not assuming.

**2. Fix the front-end connectivity deficit.**
Circuit 4 acts on wire pairs (0,1) and (2,3) independently, so two of four output
channels are single-pixel functions and no channel crosses the top/bottom row
boundary. M1's `RandomLayers` entangles all four wires — **M2 is strictly less
connected than M1 despite being trainable**, a candidate explanation for
M2 ≈ M1. A mid-layer CZ fixes this for zero parameters. It must be mid-layer: at
the end of the layer it is diagonal before a diagonal measurement and therefore
inert.

**3. Recover the dead parameter.** `w[5]` is structurally dead in the front-end
role (final CRZ diagonal before a Z measurement), so M2 has **5 effective
parameters, not 6**. Fixing (2) likely fixes this too.

**4. Separate learning rate for the front-end.** Six circuit parameters share an
optimiser group with 1,464 backbone parameters at lr 1e-3; `fe_gradvar` ~3e-3
suggests possible undertraining. One-line change.

**5. Light regularisation.** Worth ~1.9 pp against M4 on gap@best. Weight decay
above 1e-4 and augmentation are safe; dropout needs care given defect 06.

---

## 7. Reference: Circuit 4 gate sequence

Verified 2026-07-18 **directly against Hur et al. Fig 2d**:

```
RY(θ1) q_a → RY(θ2) q_b → CRZ(θ3, ctrl=q_b, tgt=q_a)
→ RY(θ4) q_a → RY(θ5) q_b → CRZ(θ6, ctrl=q_a, tgt=q_b)
```

⚠️ The middle two single-qubit gates are **RY, not RZ**. A prior session's
"confirmed" note recorded them as RZ; that was wrong and has been corrected in
`circuit4_block()`.

**Standing rule:** always verify gate sequences against a freshly viewed source
figure. Text descriptions and prior "confirmed" notes have been wrong repeatedly
across sessions (RZ vs RY; CZ vs CRZ; CNOT vs CRZ; control/target direction).

Applied identically to wire pairs (0,1) and (2,3) in each 4-qubit patch
register. Front-end output: **14×14×4** (not 7×7×4 — earlier notes were wrong).

---

## 8. M1 vs Long et al. — verification COMPLETE (2026-08-03)

**Why this matters.** M1 is meant to *be* Long's QCQ-CNN. The thesis
contribution is Long's architecture with the front-end replaced by a trainable
Circuit 4 ansatz. If M1 is not Long's model, it is not a baseline and M2 − M1
does not isolate front-end trainability against the published architecture.

Long's source is public:
`github.com/ChrisRyu2101/hybrid_quantum-classical-quantum_cnn`
(notebook `qcq_cnn_mnist3_5.ipynb`). M1 was compared against it line by line.

### 8.1 Verdict

**M1 reproduces Long's architecture exactly, with one discrepancy that traces to
a contradiction inside Long's own publication — and that discrepancy has been
measured and shown to be immaterial (§8.4).**

### 8.2 Layer-by-layer comparison

Long's `Net` class:

```python
self.conv1   = Conv2d(4, 8, 3)
self.conv2   = Conv2d(8, 16, 3)
self.dropout = Dropout2d()
self.fc2     = Linear(64, 2)        # created lazily in forward()
self.qnn     = TorchConnector(qnn)  # ZZFeatureMap(2) + RealAmplitudes(2, reps=1)
self.fc3     = Linear(1, 2)
```

| Component | Long's code | Our M1 | |
|---|---|---|---|
| Front-end | `Quanvolutional_Layer(2,1,2,0)` — 2×2, 1 layer, stride 2, 4 qubits | same | ✔ |
| Encoding | RY(π·xᵢ) | same | ✔ |
| Front-end params | random, frozen | same | ✔ |
| Front-end output | 14×14×4 | same | ✔ |
| conv1 | `Conv2d(4, 8, 3)` | same | ✔ |
| conv2 | `Conv2d(8, 16, 3)` | same | ✔ |
| Pooling | `MaxPool2d(2)` ×2 | same | ✔ |
| Flatten | → 64 | same | ✔ |
| fc2 | `Linear(64, 2)` | same | ✔ |
| Feature map | `ZZFeatureMap(2)` (reps=1 default) | same | ✔ |
| Ansatz | `RealAmplitudes(2, **reps=1**)` → 4 params | `reps=2` → 6 params | ✖ |
| fc3 | `Linear(1, 2)` | same | ✔ |
| **Total** | **1,602** | **1,604** | 2 apart (0.1%) |

An earlier draft of this section claimed Long had two dense layers before the
QNN and we had one. **That was wrong and is retracted.** Eq. 14's
`Dense2 ∘ Dense1` refers to `fc2` and `fc3`, one either side of the QNN.

### 8.3 The one discrepancy: Long's paper and code disagree

- Long's **paper**: ansatz depth d = 5 → **6** trainable RY parameters
- Long's **code**: `RealAmplitudes(2, reps=1)` → **4** trainable RY parameters

Our M1 follows the paper. This is not a defect in our implementation — it is an
inconsistency in the source, and we matched the citable artefact.

### 8.4 The discrepancy was measured, not assumed

A dedicated ablation was run (job IDs 175895589–175895609, 2026-08-03) on
Fashion-MNIST 0,6 — the task with the most headroom — for M1 **and** M2, since
both carry the QNN back-end. Prediction registered in writing before running
(`ABLATION_README.md`): expect null, because the diagnostic in §4 shows this
stage is readout-limited rather than parameter-limited.

| model | reps | params | test acc (5 seeds) |
|---|---|---|---|
| M1 | 1 (Long's code) | 1,602 | 83.76 ± 0.85% |
| M1 | 2 (Long's paper) | 1,604 | 83.74 ± 0.71% |
| M2 | 1 | 1,608 | 84.48 ± 0.56% |
| M2 | 2 | 1,610 | 84.34 ± 1.40% |

Paired across seeds: **M1 Δ = −0.02 pp (p = 0.975), M2 Δ = −0.14 pp
(p = 0.773).** MDE ≈ 1.76 pp.

Both differences are **smaller than the fixed-seed GPU non-determinism floor**
(~0.8 pp, `FMNIST_LOG.md` §5.5). This is not an effect too small to detect; it
is no effect. M1 and M2 agree in sign and magnitude, so there is no interaction
between front-end trainability and back-end depth either.

**Decision: keep reps=2.** It matches Long's paper, preserves M1 at 1,604 params
and the iso-parametric bound, and validates every completed grid. Nothing needs
re-running.

**Secondary value:** adding 50% more variational parameters to the ansatz does
nothing, while widening the readout 1 → 3 observables doubled 10-class accuracy
(§4). Both halves of the "measurement-limited, not parameter-limited" claim are
now evidenced.

### 8.5 Two properties of Long's code worth reporting

**Dropout2d() with default p = 0.5.** We use 0.0, set that way to suppress
defect 06. Long's reference implementation therefore combines dropout 0.5 with
the single-scalar QNN bottleneck — the exact configuration that produced our
train 96% / val 4% inversion.

**`fc2` appears never to be optimised.** `__init__` sets `self.fc2 = None`; the
optimiser is constructed from `model.parameters()` immediately afterwards; `fc2`
is only instantiated during the first `forward()`. As written, those 130 weights
are not in Adam's parameter list, leaving the 64→2 projection at random
initialisation. **Confirm against their exact torch version before asserting
this publicly**, but if it holds it is a substantive observation about the
reported 99.4% baseline.

### 8.6 Protocol differences (deliberate, documented)

Architecture matches; the training protocol does not, by design:

| | Long | Our M1 |
|---|---|---|
| Loss | CrossEntropyLoss | BCE (one-hot) |
| Optimiser | Adam | AdamW, wd 1e-4 |
| Batch size | 1 | 32 |
| Dropout2d | 0.5 | 0.0 |
| Epochs | 100 flat | max 100, patience 10, min 30 |
| Validation split | none | 1,250 held out |
| Model tested | final epoch | best-val checkpoint |
| Seeds | 1 | 5 |

**Correct framing: M1 is Long's architecture trained under a harmonised, more
rigorous protocol.** Do not claim to reproduce their 99.4%.

### 8.7 Bottom line

Architecturally, M1 and Long's implementation are indistinguishable in any way
that affects results. The only difference is 2 parameters out of 1,604, arising
from a contradiction in Long's own publication, measured at −0.02 pp with
p = 0.975.

---

## 9. Where we are up to

### Completed
- **MNIST 3-vs-5 grid, 20/20 clean, α = 0** — valid; no re-run needed
- Three silent correctness defects found and fixed during v6 validation
- Batched torch tensor algebra: 214× at batch 32, 803× at batch 128, ~1e-6
  relative error
- **10-class diagnostic complete** — multiclass route rejected on evidence
- **M1 vs Long verification COMPLETE (§8) — M1 matches, reps discrepancy
  measured and immaterial**
- **reps=1 vs reps=2 ablation complete** — 20 runs, null (§8.4)
- **Fashion-MNIST 0,6 / 2,6 / 2,4 grids complete** — 60 runs, all clean
  (`FMNIST_LOG.md`)
- **Full MNIST 10-class grid complete (v10) — 20/20 clean** (§4b): the primary
  M2−M1 contrast is null at MDE 16.08 pp; QNN-head models show 12–15 pp
  seed-to-seed instability. The single-seed +6.4 pp diagnostic that motivated
  this grid did not reproduce.

### Established
- MNIST 3-vs-5 is saturated: 0.08 pp spread against ~0.25–0.40 pp resolution.
  Uninformative by construction — and the control showing why MNIST-class
  results in the literature should be read cautiously.
- **M1 is architecturally equivalent to Long's published implementation.**
  Difference is 2 params of 1,604, from a paper-vs-code contradiction in the
  source, measured at −0.02 pp (p = 0.975).
- The QNN back-end is **measurement-limited, not parameter-limited**: +50%
  ansatz params changes nothing; widening the readout 1 → 3 observables doubles
  10-class accuracy.
- The single-scalar readout is a real architectural limit, visible independently
  in defect 06 and in the 10-class diagnostic (7–8/10 classes reachable).
- **QNN-head models are seed-unstable at 10 classes in a way classical-head
  models are not** (12–15 pp vs 5–6 pp spread, §4b). This, not the M2−M1
  point estimate, is the strongest 10-class finding and the main obstacle to
  resolving the central hypothesis there.
- Across all binary and 10-class grids together, every M2−M1 point estimate
  has sat inside its own MDE. The hypothesis remains open on power grounds,
  not refuted.

### NEXT EXPERIMENTS (agreed with supervisor 2026-08-03; updated 2026-08-14)

Two arms per dataset. **Arm (a) is the controlled experiment; arm (b) is the
replication.** They answer different questions and neither replaces the other.

- **Arm (a) — iso-parametric.** All four models matched to ~1,600 params.
  Answers: *is the quantum contribution attributable once capacity is
  controlled?* This is the current model set, unchanged.
- **Arm (b) — Long-faithful.** M1 exactly as Long implemented it, M2/M3/M4
  following that structure with only the intended substitutions. Not
  parameter-matched, and should not be. Answers: *does this replicate Long, and
  does the front-end swap help his model?*

**Arm (b) is a config change, not a rebuild** (§8.6): `--be_reps 1`,
`--dropout 0.5`, `--batch_size 1`, CrossEntropyLoss, plain Adam, 100 flat
epochs, no validation split, test the final model.

**Workflow rule: one full-scale smoke test before every grid. Non-negotiable.**

| # | Experiment | Runs | Notes |
|---|---|---|---|
| 1 | FMNIST **0,6** arm (a) | — | ✅ DONE (`FMNIST_LOG.md` §4) |
| 2 | FMNIST **2,6** arm (a) | — | ✅ DONE (`FMNIST_LOG.md` §5, v8) |
| 3 | FMNIST **2,4** arm (a) | — | ✅ DONE (`FMNIST_LOG.md` §5, v9) |
| 4 | FMNIST 0,6 / 2,6 / 2,4 arm (b) | 60 | Long-faithful config — not started |
| 5 | MNIST 10-class arm (a) | — | ✅ DONE (§4b, v10) — null at MDE 16 pp |
| 6 | MNIST 10-class arm (b) | 20 | needs the arm (b) config above; not started |

Priority now follows the six-step plan in `FMNIST_LOG.md` §9: per-image
predictions + McNemar for the completed 10-class grid comes before spending
more compute on arm (b), since McNemar on the existing v10 predictions may
resolve the primary contrast more cheaply than more seeds would.

**Decisions already made for the 10-class grid (v10), recorded for arm (b)
and any re-run:** wider QNN readout (3 observables, zero extra circuit
parameters) and a re-padded iso-parametric budget (M1 1640 / M2 1646 / M3
1649 / M4 1643, spread 0.55%) — see §4b and `docs/CHANGELOG_v10.md`.

**Caution for arm (b):** batch size 1 with 100 flat epochs is dramatically
slower — let the smoke test measure it before committing. And dropout 0.5 plus
the single-scalar bottleneck is exactly the defect-06 configuration, so **expect
arm (b) to be capable of inverting.** Keep the inversion check running; if it
fires there it is a property of the faithful config, not a bug — and it is a
finding.

### Deferred
- Noise sweep α ∈ {0.25, 0.5, 1.0, 1.5, 2.0} — reframed as a regularisation
  question, since M1/M2 show no accuracy advantage
- Barren-plateau measurement (`fe_gradvar` currently logs minibatch noise)
- MRI experiments
- M2 improvement variants (§6); candidate 1 is now partly subsumed into
  experiments 5–6

### Standing corrections to carry
- Thesis A §4.4.1 minibatch/parameter-shift claim is mathematically incorrect
- Generalisation gap must be computed at the same checkpoint
  (`FMNIST_LOG.md` §5.1)
- Fixed-seed runs vary ~0.8 pp from GPU non-determinism (`FMNIST_LOG.md` §5.5)
- `scripts/verify_reps.py` has not yet been run to completion — it confirms
  reps=1 against Qiskit's forward and gradients. The ablation result stands on
  the training behaviour, but run it before the §8.4 numbers go into the thesis.

---

## 10. v13.5: M5, the untrained-front-end control (2026-10-12 → 10-14)

> **Repo gap:** v11–v13 (and v14/v15) are not yet written up in this log. v13.5 uses
> v12's class-count cells (K=3/4/5, two digit subsets each, plus K=10) and v13's
> readout widths (w3/w6/w15). Full write-up: `docs/V13_5_RESULTS.md`.

**Why.** Supervisor request: every experiment from v12 to v15 must include **M5 = M3
with the front-end not trained**. M2 − M1 (§4b) changes two things at once, because
M1's frozen front-end is Henderson RandomLayers, not Circuit 4. M3 − M5 changes only
whether the Circuit-4 front-end is trained, with a classical head.

**Design (pre-registered, `docs/PREREGISTRATION_v13_5.md`).** M5 only, at all 13
v12+v13 cells × 10 seeds = 130 runs, plus 1 M3 reproduction. M5 = `build_model("M3")`
+ front-end `requires_grad=False`: same init as M3, same params, same 30-epoch warmup.
Primary: M3 − M5 (+TOST ±1) at every width; M5 − M4 (+TOST ±1) at w3. Holm within K.

**Integrity.** All gates passed. M1–M4 bit-identical to pre-patch code (verify script,
8 cells). M3 reproduction bit-exact vs v12 (test_acc 0.935, best 58, stop 78, 1000
predictions). Every M5 front-end unchanged at exit. 0 runs at the 200-epoch cap, 0
continuations.
- **A1:** the warmup rule `needs_warmup = model_name in ("M2","M3") or …` would have
  given M5 0 warmup epochs; M5 added before the smoke, resolved value now logged per run.
- **A2:** K=10 walltime 4 h → 1 h (resource only).
- **D1:** `config135.sh` defined `MODEL=M5` and overwrote the reproduction job's
  `MODEL=M3`. Caught by the gate, fixed (`GRID_MODEL`), rerun. The stray run matched
  the M5 smoke bit-for-bit, a free determinism check. Lesson: a sourced config must
  never define a name a job script takes as input.

**Results.**

| K | w | M3 − M5 (pp) | p_holm | M5 − M4 (pp) |
|---|---|---|---|---|
| 3 | 3 | +0.61 | 0.12 | −0.51 (TOST eq.) |
| 4 | 3 | +0.67 | 0.12 | −1.10 |
| 5 | 3 / 6 / 15 | +1.45 / +1.61 / +1.59 | 0.24 / 0.24 / 0.18 | −1.56 |
| 10 | 3 / 6 / 15 | +2.96 / +2.63 / +3.08 | 0.079 / **0.014** / **0.046** | −2.63 |

- **Training the front-end helps,** in 13/13 cells, growing with K. Detected at K=10.
  P1 (M3 ≈ M5) not supported.
- **An untrained Circuit-4 front-end is slightly worse than none** (M5 − M4 < 0 at every
  K; detected at K=4 subset A). With v12's M3 ≈ M4: training repairs what a random
  front-end loses; it adds nothing beyond no front-end.
- **Exploratory: seed 46 is a bad random front-end in all 13 cells** (training gains
  4–11 pp there). Without it, K=3–4 are TOST-equivalent; K=10 keeps ≈ +2.2 pp
  [+0.9, +3.5]. M5 is 2–6× less seed-stable than M3 (P4 refuted). Width control holds
  at K=5 (TOST).
- **Corrects §4b's framing:** "training the front-end adds nothing" (M2 − M1) was
  confounded by circuit type.

Figures: `results/results_v135/analysis/plots_deep/` (D1–D8); lead with D2 and D4.

---

## 11. v15: IBM-calibrated noise on the quantum front-end (2026-10-17 → 10-18)

> **Repo gap:** v14 (low data) ran 2026-10-15 → 10-17 and is not yet in this repo.
> v15 reuses v13 (M2/M3, w15), v13.5 (M5, w15) and v14 (M1/M4, N=500, w15) as its
> α = 0 cells. Full write-up: `docs/V15_RESULTS.md`.

**Why.** Supervisor brief: "V15 is noise analysis, so noisy input alpha from real IBM
calibrated noise." The last condition in the thesis aims where a front-end advantage
could appear.

**Design (pre-registered, `docs/PREREGISTRATION_v15.md`).** Noise on the **front-end
only** (the QNN head stays exact by design), from the ibm_brisbane calibration
(FakeBrisbane snapshot 2025-02-26, hash-checked by every job). Each logical gate is
charged its native IBM gates: RY → 2 SX, CRZ → 2 ECR, CNOT → 1 ECR. Each native gate gets
depolarizing + thermal relaxation (the channels `NoiseModel.from_backend` builds), and
readout error is applied to ⟨Z⟩. α scales every error rate. M1, M2, M3, M5 at α ∈ {0, 0.5,
1, 2, 4}; M4 at 0 only. K=5 A/B and K=10, N=500, w15, seeds 42–51 = **630 runs**
(63 smoke + 567 grid). Every trained model is also cross-evaluated at α ∈ {0 … 16}, so
there are two arms: **matched** (trained and tested at α) and **deployed** (trained
clean, tested at α). Primary (Holm within K): NC3@1 = M3(1) − M3(0), F@1, T@1, slope(F).

**Integrity.** All gates passed.
- The noise model equals Qiskit Aer / IBM `from_backend` channels to ≤ 2e-15, and full
  circuits to ≤ 5e-8. Against a full transpiled device simulation it is 9% optimistic.
- **150/150 α = 0 runs are bit-exact** vs the stored v13/v13.5/v14 runs.
- Every model and α at a (K, subset, seed) saw identical images.
- Every job's cross-evaluation reproduced its own saved predictions.
- 0 runs at the cap (latest stop 122), 0 continuations.
- **Patch defect (caught at setup, before any job):** the real `train_gadi.py`
  defines `build_model` and `load_dataset` twice (the v13.5 / v14 wrappers). The
  first patch required exactly one definition and refused to run. It was fixed to wrap the
  last (effective) definition and to refuse only a by-reference capture after it.
- **A1** (before the smoke, resource only): gdata had ~10.3K inodes free, so each job
  deletes its checkpoint folder after a MATCH cross-evaluation.
- K=10 kept: 19.79 KSU available, v15 ≈ 4–6 KSU.

**Results.**

| | K=5 pooled | K=10 |
|---|---|---|
| NC3@1 | −0.05 [−0.60, +0.50], **TOST eq. (p=.002)** | +0.10 [−0.80, +0.99], **TOST eq. (p=.024)** |
| F@1 = M3 − M4 | −0.54 [−1.26, +0.18], n.s. | +0.09 [−1.78, +1.95], n.s. |
| T@1 = M3 − M5 | +1.34, p_holm 0.29 | +2.88, p_holm 0.085 |
| slope(F) | −0.01 pp/α, n.s. (MDE 0.11) | +0.08, n.s. |

- **Device noise costs nothing when the model trains under it.** Every model's
  matched accuracy is flat to 4× device noise.
- **No front-end advantage under noise.** F is never detected and its slope is null. This
  closes the fifth condition: class count, readout width, training vs freezing, data
  size, hardware noise.
- **Exploratory: the QNN head amplifies front-end noise.** Trained clean, with the same
  front-end and same noise, M2 loses 7.5 / 13.8 pp at α=4 vs M3's 0.3 / 3.2. The
  noise level that costs 2 pp has a median of about 8–9× device for the classical head
  vs 2–3× for the QNN head (K=5).
- **Exploratory: seed 46 again.** It is the worst M5 in all 15 cells, and the only seed
  whose clean-trained model drops at device noise (85 → 76% at K=5). Noise-aware
  training protects it.
- **Corrects the earlier project framing:** "QCQ-CNN noise-robustness advantage" is not
  supported. Write "noise-tolerant under noise-aware training; no robustness
  advantage; the QNN head is the least robust component".

Figures: `results/results_v15/analysis/plots/` (standard) and `plots_deep/` (D1–D6);
lead with `noise_curves.png`, `deploy_curves.png`, then D2 and D3.
