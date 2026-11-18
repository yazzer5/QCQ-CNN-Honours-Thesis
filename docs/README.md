# Documentation

| Folder / file | What it holds |
|---|---|
| `MODELS_SPEC.md` | The architecture specification for M1–M4, verified against Long et al.'s paper and code |
| `EXPERIMENT_INVENTORY.md` | One row per experiment, v1–v16: question, runs, key result, where it is written up |
| `preregistrations/` | Written before each experiment ran: question, design, contrasts, predictions, stop rules, and dated amendments |
| `results/` | One results write-up per experiment: integrity, accuracy tables, primary contrasts, prediction verdicts, findings, caveats |
| `analysis/` | Cross-experiment work: the v16 synthesis, the closed-form front-end derivation, the v15.5 noise mechanism and theory, and the deep analysis of v6–v12 |
| `process/` | Changelogs (v7, v8, v10), protocol amendments (v10 loss, v10 reduced scale) and the GPU root-cause note |

## Reading order

1. `analysis/V16_SYNTHESIS.md`: the whole argument in one place.
2. `EXPERIMENT_INVENTORY.md`: what was run, in order.
3. The results file for any experiment of interest, with its pre-registration next to it.

## Results by experiment

| Exp | Pre-registration | Results |
|---|---|---|
| v7–v9 Fashion-MNIST | `PREREGISTRATION_2_6.md`, `PREREGISTRATION_2_4.md` | `../logs/FMNIST_LOG.md` |
| v10 MNIST 10-class | `PREREGISTRATION_MNIST10.md` | `../logs/MNIST_LOG.md` §4b |
| Ansatz depth ablation | `ABLATION_README.md` | `../logs/FMNIST_LOG.md` |
| v12 class count | `PREREGISTRATION_v12.md` | `V12_RESULTS.md` |
| v13 readout width | `PREREGISTRATION_v13.md` | `V13_RESULTS.md` |
| v13.5 frozen front-end (M5) | `PREREGISTRATION_v13_5.md` | `V13_5_RESULTS.md` |
| v14 low data | `PREREGISTRATION_v14.md` | `V14_RESULTS.md` |
| v15 hardware noise | `PREREGISTRATION_v15.md` | `V15_RESULTS.md` |
| v15.5 pixel noise | `PREREGISTRATION_v15_5.md` | `V15_5_RESULTS.md` (+ `analysis/V15_5_MECHANISM.md`, `analysis/V15_5_THEORY.md`) |
| v15.6 noise geometry | `PREREGISTRATION_v15_6.md` | `V15_6_RESULTS.md` |
| v15.7 bubble during training | `PREREGISTRATION_v15_7.md` | `V15_7_RESULTS.md` |
| v15.8 classical cos control (M6) | `PREREGISTRATION_v15_8.md` | `V15_8_RESULTS.md` |
| v16a closed-form head | `PREREGISTRATION_v16a.md` | `V16A_RESULTS.md` |
| v16b redesigned head (+ ext) | `PREREGISTRATION_v16b.md` | `V16B_RESULTS.md` |
| v16c diagnostics | `PREREGISTRATION_v16c.md` | `V16C_RESULTS.md` |

Older results files are left as written. Where a later experiment changed the
reading of an earlier one, the correction is recorded in `analysis/V16_SYNTHESIS.md`
§4 or in a short note at the top of the affected file.
