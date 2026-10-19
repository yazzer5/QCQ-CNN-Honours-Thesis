# v13.5: M5, the untrained-front-end control (MNIST, K = 3/4/5/10)

**M5 = M3 with the Circuit-4 front-end frozen at its seed initialisation.** Same
initial weights as M3 (same seed), same 30-epoch warmup, same parameter count; only
the front-end's 6 parameters are never trained. M3 − M5 isolates front-end training.

| Folder | Runs | What |
|---|---|---|
| `k3_{A,B}_w3/`, `k4_{A,B}_w3/` | 4 × 10 | v12 cells (digit subsets drawn with seed 20260928) |
| `k5_{A,B}_w{3,6,15}/` | 6 × 10 | v12 + v13 cells (readout widths 3, 6, 15) |
| `k10_all_w{3,6,15}/` | 3 × 10 | v12 + v13 cells |
| `repro_k5_A_w3_M3/` | 1 | M3 rerun in the patched code: bit-exact vs v12 (test_acc, best/stop epoch, all 1000 predictions) |
| `analysis/` | | `v135_{runs,summary,contrasts}.csv`, `plots/` (standard set), `plots_deep/` (D1–D8 + `deep_numbers.json`), `deep_v135.py` |

All 130 M5 runs include per-image test predictions. Job logs: `../job_logs/v135/`
(14 smoke `v135s_*`, 117 grid `v135_*`, plus setup `patch_v135.log` / `verify_v135*.log`).
Code: `code/v13_5_m5/`. Pre-registration with amendments A1, A2 and defect D1:
`docs/PREREGISTRATION_v13_5.md`. Write-up: `docs/V13_5_RESULTS.md`.

**Comparators are not duplicated here.** M1–M4 come from v12 (K=3/4/5, w3), v10 +
v13 top-up (K=10, w3) and v13 (K=5/10, w6/w15 M2/M3). Those grids are not yet in this
repo; they live on Gadi under `qcq_cnn_v12/` and `qcq_cnn_v13/`. The `file` column in
`v135_runs.csv` gives each row's original Gadi path.

## Headline (K=3/4/5 pooled over subsets; 10 seeds; Holm within K)

| K | w | M3 − M5 (pp) | 95% CI | M5 − M4 (pp) |
|---|---|---|---|---|
| 3 | 3 | +0.61 | [−0.03, +1.24] | −0.51 (TOST ±1 equivalent) |
| 4 | 3 | +0.67 | [−0.22, +1.56] | −1.10 |
| 5 | 3 / 6 / 15 | +1.45 / +1.61 / +1.59 | | −1.56 |
| 10 | 3 / 6 / 15 | +2.96 / **+2.63** / **+3.08** | detected at w6, w15 | −2.63 |

Training the front-end helps in 13/13 cells and the gain grows with K; an untrained
front-end is slightly worse than none. Exploratory: seed 46 is a bad random front-end
in every cell (training rescues it). Without it, K=3–4 are equivalent within ±1 pp and
K=10 keeps a ~2 pp gain.
