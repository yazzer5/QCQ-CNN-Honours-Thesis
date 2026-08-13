# CHANGELOG v10 — MNIST 10-class multiclass support

Base: v9 (byte-identical to v8 except version tags). All v10 changes are
gated on class count, so the binary path is untouched and v6–v9 results
remain valid. Verified by `scripts/verify_v10_multiclass.py`.

## 1. Three-observable QNN readout  (`qnn_torch.py`)
`QNNBackendTorch(n_obs=1|3)`. n_obs=3 returns `<Z0>`, `<Z1>`, `<Z0Z1>` read
from the same final statevector — the circuit, its depth, and its 6
variational parameters are unchanged, so this costs zero circuit parameters.
Motivation: the 10-class diagnostic showed the back-end is measurement-
limited, not parameter-limited (1→3 observables doubled accuracy; +50% more
variational parameters changed nothing, p = 0.975).
`<ZZ>` is bit-identical between n_obs=1 and n_obs=3.

## 2. Iso-parametric re-padding for multiclass  (`train_gadi.py`)
`EXPECTED_PARAMS` dict → `expected_params(model, n_classes, n_obs)`.
Binary returns the v6–v9 values unchanged.

A naive `Linear(64,10)` classical head is 650 params against the QNN head's
176 — a 474-param gap on a ~1600 budget. M3/M4's multiclass head instead
mirrors the quantum head's shape: `fc2(64→2) → fc_mid(2→n_obs) →
fc3(n_obs→n_classes)`. The only structural difference between M2 and M3 is
whether the 2→3 map is a quantum circuit or a linear layer.

    M1 1640   M2 1646   M3 1649   M4 1643      spread 0.55%
    M2 − M1 = 6   M3 − M4 = 6      (front-end params only)

`--head_variant {linear,tanh}` controls the classical mid-map activation.
Default `linear`, for continuity with the binary head. See the
pre-registration for why this is a limitation rather than a neutral choice.

## 3. `--deterministic`
`torch.use_deterministic_algorithms(True)` + cuDNN determinism +
`CUBLAS_WORKSPACE_CONFIG`. Costs ~10–20% throughput. Removes the ~0.8 pp
fixed-seed run-to-run variance that inflates SD(diff) and weakens every
paired test. The run prints whether it is on; if off, that must be disclosed
with any result derived from it.

## 4. `--save_predictions`
Per-image test predictions and labels into the result JSON, collected on the
restored best-validation checkpoint in a fixed (shuffle=False) order so runs
line up index-for-index. A tripwire asserts the recomputed accuracy matches
the reported `test_acc` to 1e-6 — the same class of defect that produced the
retracted gap metric would otherwise be invisible.
Consumed by `scripts/mcnemar.py`.

## 5. Dataset path generalised
`--classes` now works for MNIST as well as Fashion-MNIST, so the same code
serves 3-vs-5 and the full 10-class task. Per-class subset sizes divide by
`len(classes)` rather than a hardcoded 2. Asserts full class coverage in the
training pool and prints the realised counts.

## 6. Inversion detector gated to binary
The signature (high train accuracy, below-chance validation accuracy) is only
meaningful for 2 classes; at 10 classes "below chance" is 10% and legitimate
early-training states sit there. Gated off for multiclass, not deleted.

## 7. New tooling
- `scripts/verify_v10_multiclass.py` — 24 offline checks: binary regression
  guard, multiclass budget, forward/backward and gradient flow for all four
  models, observable properties, tanh variant.
- `scripts/mcnemar.py` — McNemar's test on pooled per-image predictions,
  with a per-seed table and an explicit caveat that pooling overstates
  certainty because the seeds share a test set.

## Not changed
Circuit 4 front-end, Henderson quanvolution, CNN backbone, loss
(BCE-on-one-hot), optimiser, early stopping, gap@best definition,
checkpointing, time-budget/continuation logic.
