# reps=1 vs reps=2 ablation — QNN back-end ansatz depth

## Why

Long et al.'s **paper** states ansatz depth d = 5, giving 6 trainable RY
parameters. Long et al.'s **published code**
(github.com/ChrisRyu2101/hybrid_quantum-classical-quantum_cnn) uses
`RealAmplitudes(2, reps=1)`, giving 4.

**The paper and the code disagree.** Our M1 follows the paper (reps=2, 6
params, 1,604 total); Long's code would give 1,602. Every other layer matches
exactly — verified line by line against their `Net` class.

This ablation measures whether the disagreement is worth caring about, so the
choice can be justified by measurement instead of argument.

## Prediction (registered before running)

**Expected: null.** `RealAmplitudes` is RY + CNOT, so the ansatz is real
orthogonal — SO(4) for two qubits, 6 parameters. reps=1 reaches a 4-dimensional
slice; reps=2 can reach all of SO(4). There is a real expressivity difference at
the unitary level, but the back-end collapses to a single scalar ⟨ZZ⟩, and the
10-class diagnostic already showed this stage is **readout-limited, not
parameter-limited** (1 → 3 observables doubled accuracy at zero change to the
ansatz).

So the extra 2 parameters push on the slack constraint. Expect |Δ| below the
~1.8 pp MDE.

**If the prediction fails** and reps=2 is materially better, that is a genuine
finding: it means Long's published code understates their own architecture.

## Scope

M1 and M2 × reps {1,2} × 5 seeds = **20 runs**, Fashion-MNIST 0,6.

M3/M4 have no QNN back-end and are unaffected — their existing results stand.
Fashion 0,6 is used because it has 15.8 pp of headroom; MNIST 3-vs-5 is
saturated and could not show a difference either way (`--dataset mnist` is
supported if a saturated control is wanted).

Both arms are re-run in the same batch rather than reusing the existing reps=2
grid, because fixed-seed runs vary ~0.8 pp from GPU non-determinism. A
same-batch comparison keeps the pairing meaningful.

## Correctness

`qnn_torch.py` now builds the ansatz as a loop:

```
RY layer, then reps x [CNOT, RY layer]
```

matching Qiskit's `RealAmplitudes(n, reps)`.

**reps=2 is bit-identical to the previously validated implementation**
(max|diff| = 0.0 over random inputs), so the completed grids remain valid.

Before running the grid, confirm reps=1 against real Qiskit:

```bash
python scripts/verify_reps.py
```

Checks forward, weight gradients, and input gradients for both depths against
`EstimatorQNN`. **Do not run the grid if this fails.**

## Output

Result filenames carry the depth (`..._reps1.json` / `..._reps2.json`) so the
two arms cannot overwrite each other. `protocol.be_reps` is recorded in every
JSON.

## Reading the result

A null means the ansatz depth does not matter *at this scale on this task*. It
is not evidence that the two are equivalent in general — report it with the MDE,
as with every other null in this project.
