# 10-class MNIST diagnostic (M1–M4)

**This is a diagnostic, not a thesis experiment.** It answers one question:
what happens to the four architectures when the task has 10 classes?

## Run it

Runs on the **login node** in ~2 minutes, CPU only. No PBS job needed.

```bash
cd /g/data/wa66/yh2481/qcq_cnn_v7/v7        # or wherever smoke10/ is placed
source /g/data/wa66/yh2481/miniconda3/etc/profile.d/conda.sh
conda activate qcqcnn
export PYTHONPATH=""

python scripts/preflight_data.py --data_root ./data --classes 0,6   # once (v7 script)
python smoke10_mnist.py
```

Options: `--head narrow|wide|both` (default both), `--epochs`, `--n_train`,
`--seed`. Writes JSON per run to `results_smoke10/`.

## Three caveats that must be stated with any number

**1. Not iso-parametric.** Ten classes forces the output layers to widen, and
they widen unequally:

| | back-end params |
|---|---|
| M1/M2 narrow | 130 + 6 + 20 = **156** |
| M1/M2 wide | 130 + 6 + 40 = **176** |
| M3/M4 | **650** |

M3/M4 carry ~500 more parameters. Any M3/M4 advantage here is confounded by
capacity. The binary grids do not have this problem — all four are matched to
within 0.93%.

**2. The QNN head is bottlenecked by construction.** The locked design squeezes
the network through `fc2(64→2)` → 2-qubit circuit → a **single** expectation
value `<ZZ>` ∈ [−1,1], then `fc3` maps that one scalar to the class logits. For
2 classes, thresholding one scalar is sufficient. For 10 classes it is not:
`fc3(1,10)` produces logits affine in one number, so the prediction is decided
by where a single scalar falls on the real line — a 1-D ordering of 10 classes.

**A low M1/M2 score is evidence about the readout width, not about quantum
layers.** Do not report it as "quantum models are worse at multiclass".

**3. One seed.** The training protocol otherwise **matches the binary grids
exactly**: 5000/1250/1000, batch 32, lr 1e-3, weight decay 1e-4, dropout 0.0,
max 100 epochs, patience 10, min_epochs 30 warm-up. The only reduction is a
single seed (42), so there are no error bars and no significance tests. Every
JSON is tagged `REDUCED_SCALE: true` and `ISO_PARAMETRIC: false`.

*An earlier draft used a flat 15 epochs with no early stopping. Withdrawn: the
wide-head models were still improving at epoch 15 (val 0.254 → 0.489, no
plateau), so the fixed budget understated them and biased the comparison
against the quantum models. The narrow head was saturated by epoch 2 either
way.*

## What is shared with the real grids

`quanv_torch.py` and `qnn_torch.py` are the **verified, unmodified** modules
from v7. The front-ends, the CNN backbone, and the circuit are identical. Only
the class-count-dependent output layers differ. The `wide` head reads three
observables `<Z0>`, `<Z1>`, `<ZZ>` off the **same** 6-parameter state — it adds
no circuit parameters, only more measurements.

## Reference result (seed 42, grid protocol, CPU)

Narrow head — complete:

| model | head | test acc | stop epoch | distinct classes | params |
|---|---|---|---|---|---|
| M1 | narrow | 32.30% | 58 | 7/10 | 1620 |
| M2 | narrow | 28.30% | 48 | 7/10 | 1626 |
| M3 | — | **95.60%** | 39 | 10/10 | 2120 |
| M4 | — | **96.50%** | 57 | 10/10 | 2114 |

Chance = 10.00%. Wide-head reference numbers are produced by the Gadi run.

**Reading it.** With a single `<ZZ>` readout the QNN models reach roughly a
third of the classical heads' accuracy and never learn to emit more than 7 of
the 10 classes, despite training to convergence under the full protocol. Under
the earlier 15-epoch budget the same models scored ~22%, so the deficit is not
an artefact of undertraining — it persists at convergence.

Widening the readout to three observables adds **zero** circuit parameters (the
same 6-parameter state, two extra measurements) and lifted accuracy from ~22%
to ~47% under the short budget. If that ratio holds at convergence, it isolates
the limiter as the **readout width** rather than the quantum layer. Both remain
below the classical heads, but M3/M4 also hold ~500 more parameters, so the
residual gap is not cleanly attributable.

**The useful conclusion for the thesis:** multiclass would require redesigning
the QNN back-end, which breaks the iso-parametric bound that makes the M2−M3
and M3−M4 contrasts interpretable. Confirms the decision to stay binary and
obtain headroom from harder class pairs instead.
