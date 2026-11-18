# Experiment inventory, v1 → v16

One row per experiment. Built 2026-10-31 from the Gadi export, this repository and the
write-ups in `docs/`; v15.6–v15.8 added 2026-11-18.

**Totals:** ~2,750 training runs to v16 · 2,891 PBS jobs · ~30,700 SU (summed from PBS
resource-usage footers, unique job IDs). v15.6–v15.8 add 273 runs.

| Phase | Ver | Date | Question | Runs | Key result | Write-up |
|---|---|---|---|---|---|---|
| 0 Replication & infra | v1 | Mar 2026 | Can Long et al.'s QCQ-CNN be reproduced locally? | notebook | 600-sample smoke test (Thesis A) | `code/v1_v4_replication/qcq_cnn_mnist3_5.ipynb` |
| | v2 | 9 Apr | Full replication on Gadi (5000 samples, seed 42) | 1 job, 236 SU | QCQ-CNN 98.4% · QC-CNN 98.3% · Classical CNN 99.3% · Q+Dense 96.6% · Dense 96.0%. GPU util 0% | `results/results_replication/` |
| | (—) | Jun | Circuit selection: Hur Circuits 2 / 4 / 6 as M2 front end (MNIST 3v5, 100 train / 30 test, bs 1) | screen + 5-seed tiebreak | Phase 1: C2 stuck at chance. Phase 2 (seeds 7, 21, 32, 42, 55, 50 ep): **C4 93.3% best / 4 of 5 seeds ≥ 90% vs C6 86.0% / 3 of 5.** Rerun after the CRZ wiring fix. Tiny test set: selection is weak evidence, backed by the Hur et al. balance argument | `code/v1_v4_replication/m2_circuit_selection.ipynb` |
| | (—) | Jun | Four-model smoke test (50 train / 20 test, 3 seeds, 50 ep) | 12 | M2 96.7% best vs M1 85.0%; M3/M4 100% best. First sign the QNN head is the bottleneck. 20-image test set: one image = 5 pp | Thesis B report |
| | v3 | Jul | First `train_gadi.py` M1–M4 grid (MNIST 3v5 + FMNIST T-shirt/Trouser + MRI), batch size 1, chained jobs | 17 complete of 60 | M4 ≈ 99% (MNIST), 98.7% (FMNIST). M1 48–80% (near chance, later traced to dropout inversion). M2/M3 never finished: walltime, then SU quota exhausted. RZ→RY Circuit 4 fix made here (18 Jul) | `results/results_v3_grid/` |
| | v4 | late Jul | `unfold()` patch batching to fix 0% GPU | — | Did not help: TorchLayer re-serialised the batch into B×196 tapes | `docs/process/PROGRESS_2026-07-28_gpu_rootcause_and_batchsize.md` |
| | v5 | 28 Jul | Batched torch simulator; validate correctness | 20 (smoke / 1000 / 3000) | 214–803× speedup, 8.9e-16 vs reference. Defects found: QNN on CUDA silently not updating; dropout 0.5 inverts single-scalar head (locked to 0); early-stopping bug; w[5] structurally dead | `code/v5_validation/`, `docs/MODELS_SPEC.md` §9 |
| 1 Binary benchmarks | v6 | 28 Jul | MNIST 3v5, M1–M4 × 5 seeds | 20 | 99.14–99.22%, 0.08 pp spread: saturated | `logs/MNIST_LOG.md` §2 |
| | v7 | 2 Aug | FMNIST 0,6 | 20 (+20 depth ablation, +6 10-class diagnostic) | Contrasts null (MDE 1.8). QNN head −1.18 pp vs classical, **paired p = 0.027** (0.011 was an unpaired Welch test, superseded). Ablation: reps 1→2 null (p = .975). Diagnostic: 1→3 observables doubles 10-class accuracy | `logs/FMNIST_LOG.md` §5 |
| | v8 | 11 Aug | FMNIST 2,6 + gap@best fix | 20 | M3 best (87.70). M2−M1 +1.04 (p = .075) | `logs/FMNIST_LOG.md` |
| | v9 | 12 Aug | FMNIST 2,4 | 20 | M3 best (88.42). M2−M1 −0.30. Difficulty gradient collapsed (retracted) | `logs/FMNIST_LOG.md` §5.7 |
| 2 Multiclass & real data | v10 | 13 Aug | MNIST 10-class, 3 observables, CE loss (amended after loss smoke) | 20 + 2 smoke | Reduced-scale. M2−M1 +0.33, MDE 16.08. Classical head +11.4 pp (p = 2.2e-5). QNN seed spread 12–15 pp | `logs/MNIST_LOG.md` §4b, `docs/process/AMENDMENT_*` |
| | v11 | 7–9 Sep | Brain tumour MRI (Kaggle 7,200; glioma vs meningioma+pituitary; 2240/560/772 balanced; 28×28 z-scored) | 20 | **M3−M4 −3.68 pp, Holm p = .002**: quantum front end harmful. No pre-registration doc; protocol and sanity band fixed in README beforehand | `code/v11_brain_tumor/README.md`, `docs/analysis/DEEP_ANALYSIS_ALL_GRIDS.md` |
| | v12 | Sep–Oct | Class count K = 3, 4, 5 (two subsets each), 10 seeds | 240 | Front end ≡ 0 (TOST ±1 pp). Back-end cost grows with K: 1.4 → 2.7 → 4.7 pp | `docs/results/V12_RESULTS.md` |
| 3 Back end & front-end controls | v13 | Oct | Readout width w = 3 / 6 / 15 | 142 | Cost falls to ~2.3 pp at w15, still detected at K=5 (p_holm .0024). At w ≥ 6 the circuit is redundant | `docs/results/V13_RESULTS.md` |
| | v13.5 | 12–14 Oct | M5 = M3 with front end frozen | 132 | M3−M5 > 0 in all 13 cells; M5−M4 < 0. Mostly seed 46 | `docs/results/V13_5_RESULTS.md` |
| 4 Stress conditions | v14 | 15–16 Oct | Low data N = 50–500 per class | 512 | Front end no help (>1 pp excluded). QNN head cost 13.5 / 17.1 pp at N=50 | `docs/results/V14_RESULTS.md` |
| | v15 | 16–18 Oct | IBM-calibrated circuit noise α ≤ 4 (FakeBrisbane) | 630 | Noise-trained: cost ≡ 0. Noise-naive deployment: QNN head amplifies noise (7.5–13.8 pp) | `docs/results/V15_RESULTS.md` |
| | v15.5 | 19–21 Oct | Gaussian pixel noise σ ≤ 0.5 | 750 | **Only positive front-end result:** M3−M4 +9.1 pp at σ=0.3 deployed (Holm p = .001). Needs training (M5 not robust). Mechanism plausibly classical (flat cos at 0/1) | `docs/results/V15_5_RESULTS.md`, `docs/analysis/V15_5_MECHANISM.md`, `docs/analysis/V15_5_THEORY.md` |
| | v15.6 | 4–5 Nov | Noise geometry (shift, spread, Fisher ratio, bleed) per layer, M3/M4/M5 | 180 | Noise cloud not smaller with the circuit (2×2 averaging is tighter), but displaced less from conv1 onward; bleed 10% vs 23% at L1 (K5). Fisher ratio does not separate the models | `docs/results/V15_6_RESULTS.md` |
| | v15.7 | 6 Nov | Does the noise bubble tighten earlier in training for M3? | 60 | M3 overtakes M4 by about epoch 3 and holds; M4's bubble loosens as it fits clean data. Pre-registered early-window primary not detected (epoch-label offset, A1) | `docs/results/V15_7_RESULTS.md` |
| | v15.8 | 9 Nov | Classical control M6: fixed cos(πx) per pixel, 0 parameters | 33 | **M6 ≥ M3:** at K=5 deployed σ=0.3, M6 − M4 +10.6 pp and M3 − M6 −4.2 pp (Holm p = .026 both). The robustness is the encoding, not the trained circuit | `docs/results/V15_8_RESULTS.md` |
| 5 Diagnose → redesign | v16a | 21 Oct | Is the QNN head's cost classical? (closed-form replica M2c) | 73 | M2c reproduces cost, collapse, per-image errors. Circuit worth ≤ 0.75 / 0.88 pp | `docs/results/V16A_RESULTS.md` |
| | v16b | 22 Oct | BFR head (bounded, gated, residual) vs M4 | 61 | Recovers 75–95% of the cost; never beats M4. K10 N50 −4.7 pp (Holm p = .024) | `docs/results/V16B_RESULTS.md` |
| | v16b-ext | 22 Oct | What drove the recovery? | 21 | Not the product term. Learned gain fell 4–5×. Half the residual is the fc_mid trim | `docs/results/V16B_RESULTS.md` §11 |
| | v16c | 22 Oct | Mechanism diagnostics (z-dumps, init conditioning), analysis only | 0 new runs | Seed 46 = exact dead channel. Head = **fixed high gain (~9 rad/unit)**, not unbounded frequency (H1–H3 failed) | `docs/results/V16C_RESULTS.md` |

## Never run

- **Arm (b), the Long-faithful protocol.** Planned from Aug, dropped when the scope changed.
- **Circuit selection (Circuits 2 / 4 / 6) as a Gadi batch job.** Attempted 10 Jun, crashed in 1 min; the selection ran interactively instead.
- **Locked full-protocol 10-class confirmation.** v10 stayed reduced-scale; v12/v13 at 10 seeds partly supersede it.
- **Phase-sweep control** (fixed cos(πx + δ) at several δ, `V15_5_THEORY.md` P-T4) and a raw-pixel control separating "cosine" from "four pixels kept" (v15.8 §7).

## Resolved during the audit

- v7 grouped p: **0.027, paired by seed, df = 4.** 0.011 was an unpaired Welch test on 10 vs 10 runs, which ignores the shared seeds. Conclusion unchanged.
- Optimiser for v6–v11: AdamW, lr 1e-3, weight decay 1e-4, batch 32, dropout 0, max 100 epochs, patience 10, BCE. v10 and later: CE, max 200, patience 20.
- Environment exports: `code/v12_class_count/env_qcqcnn_v10.yml`, `code/v12_class_count/pip_freeze_v10.txt` (and the v11 exports in `code/v11_brain_tumor/`).
