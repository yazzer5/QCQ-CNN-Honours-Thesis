# v14: low data (sample efficiency), M1–M5 at full readout (w15), MNIST K = 5 / 10

Training set shrunk to **N = 50, 100, 250, 500 images per class** (val 125/class, test
200/class fixed). Every model at readout width 15 (all 15 Paulis). Max-epoch cap scaled
as 200 × 500 / N; early stopping (patience 20) ended every run. 10 seeds (42–51).

| Folder | Runs | What |
|---|---|---|
| `k5_{A,B}_n{50,100,250}/` | 6 × 50 | M1–M5 × 10 seeds (v12 digit subsets, seed 20260928) |
| `k10_all_n{50,100,250}/` | 3 × 50 | M1–M5 × 10 seeds |
| `k5_{A,B}_n500/`, `k10_all_n500/` | 3 × 20 | M1, M4 only (never run at w15 before) |
| `repro_k5_A_n500_M3/`, `repro_k5_A_n500_M5/` | 2 | Patched-code reruns: bit-exact vs v13 (M3) and v13.5 (M5) — test_acc, best/stop epoch, all 1000 predictions |
| `analysis/` | | `v14_{runs,summary,contrasts}.csv`, `collect_v14.log`, `plots/` (standard set), `plots_deep/` (D1–D5) |

All 512 runs include per-image test predictions and `train_acc_at_best`. Job logs:
`../job_logs/v14/` (53 smoke `v14s_*`/`v14r_*`, 459 grid `v14g*`, plus `patch_v14.log`,
`verify_v14*.log`). Every log carries a `[V14] SPLIT` audit (per-class counts, image
hashes, overlap). Code: `code/v14_lowdata/`. Pre-registration: `docs/PREREGISTRATION_v14.md`.
Write-up: `docs/V14_RESULTS.md`.

**Comparators not duplicated here:** N = 500 M2/M3 come from v13 (`results_v13_k*_w15`,
on Gadi, not yet in this repo) and M5 from v13.5 (`../results_v135/k*_w15/`). The `file`
column in `v14_runs.csv` gives each row's source path.

## Headline (K=5 pooled over subsets; 10 seeds)

| | N = 50 | 100 | 250 | 500 |
|---|---|---|---|---|
| F = M3 − M4, K=5 / K=10 | −1.47 / −1.74 | −1.41 / −2.89 | −0.57 / −0.87 | −0.49 / −0.01 |
| BE = M3 − M2, K=5 / K=10 | **+13.5 / +17.1** | +9.7 / +10.8 | +4.2 / +5.3 | +2.3 / +2.4 |

No sample-efficiency advantage from the quantum front-end (primary tests all null; an M3
advantage over M4 above +0.8 pp (K=5) / +1.1 pp (K=10) at N = 50 is excluded). The quantum
head's cost grows as data shrinks (slope −3.5 / −4.4 pp per halving, p ≤ 5×10⁻⁵), and it
is almost entirely a generalisation gap (`plots_deep/D2`). Seed 46 is again the worst
frozen front-end in every cell.
