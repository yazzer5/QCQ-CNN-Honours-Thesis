# Pre-registration v15.5: noisy input images (pixel noise), M1–M5

**Written:** 2026-10-20, before any v15.5 job was submitted and before any v15.5 accuracy existed.
**Follows:** v12 (`POST_CORE_RESULTS.md`), v13, v13.5, v14 (`V14_RESULTS.md`), v15 (`V15_RESULTS.md`). Runs in parallel with v16 (`V16_PLAN.md`); separate repo, job names and result folders, no shared files.
**Why:** v15 put IBM-calibrated noise *inside the quantum circuit*. The images stayed clean and M4 (no quantum part) was untouched by construction. The supervisor brief ("noisy input") may instead mean **corrupted images**. v15.5 tests that reading: noise is added to the **pixels**, so all five models see it, M4 included.

## 1. Question

**Q-v15.5. When the input images are corrupted by additive Gaussian pixel noise, does a quantum front-end make the model more robust than no front-end (M4) or a frozen one (M5): (a) when the model is trained clean and deployed on noisy images, and (b) when it is trained on noisy images?**

## 2. Models (unchanged code, all at readout width 15, N = 500/class)

| | Front-end | Head | Params K=5 / K=10 |
|---|---|---|---|
| M1 | Henderson RandomLayers, frozen | QNN | 1680 / 1760 |
| M2 | Circuit 4 (Hur et al.), trained | QNN | 1686 / 1766 |
| M3 | Circuit 4, trained | classical | 1725 / 1805 |
| M4 | none (bilinear resize) | classical | 1719 / 1799 |
| M5 | Circuit 4, frozen at seed init | classical | 1725 / 1805 |

Protocol identical to v13/v13.5/v14/v15: CE loss, linear head, `--deterministic`, `--save_predictions`, batch 32, AdamW lr 1e-3, wd 1e-4, dropout 0, max 200 epochs, patience 20, 30-epoch warmup for M1/M2/M3/M5 (0 for M4), `--noise_alpha 0.0`, val 125/class, test 200/class. Code: `qcq_cnn_v155`, built from `qcq_cnn_v14` (code only; v15's circuit-noise patch is **not** included).

## 3. The noise (fixed now)

**Corruption:** additive Gaussian noise of standard deviation σ on each pixel of the [0, 1] image, then clipped to [0, 1]:
`x̃ = clip(x + σ·ε, 0, 1)`, ε ~ N(0, 1) i.i.d. per pixel. Clipping keeps the RY(πx) encoding angles in [0, π] for the quantum front-ends.
**Where:** on the 28×28 image `_TensorImgDS` returns, before the model, so before the quantum front-end (M1, M2, M3, M5) and before the bilinear resize (M4). Identical for all five models.
**Levels:** training (and matched-test) σ ∈ {0, 0.1, 0.2, 0.3, 0.5}. Cross-evaluation σ ∈ {0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0}.

**How it enters the code.** `patch_v155_input_noise.py` inserts one block above `if __name__ == "__main__":` that wraps the module-global `load_dataset` (and `build_model`, only to keep a reference to the model for the cross-evaluation) by name. No model, data, split or training line is edited. The v14 split audit runs *inside* the wrapped function, so its `[V14] SPLIT` hashes stay on the **clean** splits. With `V155_SIGMA` unset or 0 the datasets are returned untouched.

**Determinism (exact):**
- **Test noise:** one fixed noise field per σ, generated once from `torch.Generator().manual_seed(15 500 000 + round(1000σ))`. The test images are already fixed per (K, subset), so **every model and seed is tested on identical noisy images** at each σ. Every contrast is paired on the same corrupted images. Each job logs the sha256 of its noisy test set; the gate requires one hash per (K, subset, σ).
- **Validation noise:** fixed per σ, seed `15 600 000 + round(1000σ)`. Early stopping in the matched arm uses a fixed noisy val set.
- **Training noise:** fresh every time an image is fetched (so new every epoch: noise augmentation), from a private `torch.Generator` seeded `155 000 000 + 10 000·seed + round(1000σ)`. It never touches the global RNGs (python, numpy, torch, CUDA), so shuffling and model initialisation are exactly those of v13/v13.5/v14. Logged per job; `DataLoader` uses `num_workers = 0` (a worker process would duplicate the generator; the wrapper refuses to run in a worker).
- **Side effect, stated:** every pass over the training loader sees noisy images, including the pre-training checks and `train_acc_at_best`. In the matched arm the generalisation gap is therefore measured against noisy training images.

**Size of the corruption (context, not a result).** MNIST background pixels are 0. After clipping, background pixels have mean ≈ 0.40σ (half-normal), strokes near 1 lose ≈ 0.40σ. At σ = 0.3 digits are clearly readable; at 0.5 they are heavily corrupted; 0.7 and 1.0 are stress levels.

## 4. Design

| K | Subsets (v12, seed 20260928) | σ (training = matched test) | Models | Seeds | Jobs |
|---|---|---|---|---|---|
| 5 | A {1,3,5,8,9}, B {0,2,6,8,9} | 0, 0.1, 0.2, 0.3, 0.5 | M1–M5 | 42–51 | 500 |
| 10 | all digits | 0, 0.1, 0.2, 0.3, 0.5 | M1–M5 | 42–51 | 250 |
| | | | | **total** | **750** |

**Cross-evaluation (no extra jobs).** At the end of every job, the trained (best-checkpoint) model is first evaluated on the test set at its own σ and must reproduce, image for image, the predictions `train_gadi.py` saved (self-check MATCH; only a MATCH is used). It is then evaluated at every cross-evaluation σ, and, **exploratory**, at salt-and-pepper noise with p ∈ {0.05, 0.1, 0.2} (fraction of pixels set to 0 or 1 with equal probability; fixed per p, seed `15 700 000 + round(1000p)`).

**Two arms from one grid:**
- **Matched:** trained, validated and tested at σ (noise-aware training). = the job's own test accuracy.
- **Deployed:** trained clean (σ = 0), tested at σ. = the σ = 0 job's cross-evaluation. The classic robustness question.

**The σ = 0 jobs are reproductions.** With σ = 0 the patch returns the datasets untouched, so each σ = 0 run must equal the stored run exactly (test_acc, best and stop epoch, every prediction): M1, M4 from v14 (`results_v14_k*_n500`), M2, M3 from v13 (`results_v13_k*_w15`), M5 from v13.5 (`results_v135_k*_w15`). The σ = 0 runs must be re-run anyway: v15's α = 0 sidecars hold circuit-noise cross-evaluations, not σ, and v15 deleted its checkpoints.

**One smoke test** (75 jobs, one submission, one gate): seed 42 at every (K, subset, σ, model) cell, kept and reused by the grid. Pre-registered reproduction gate: K = 5 A, σ = 0, seed 42, all five models, exact (stop rule). Every other σ = 0 run (smoke and grid) that has a stored counterpart must also reproduce exactly, or its gate fails (v15 reproduced 150/150). The grid is the other 675 jobs.

**Resources (decided 2026-10-20, before the smoke, no result read).** `nci_account`: 18.40 KSU available of 25. v13–v15 used about 2–7 SU per job, so 750 jobs are about 2–5 KSU. `lquota` gdata: 1,010,470 of 1,018,000 inodes (7,530 free), shared with v16. Each job leaves 3 files (log, result JSON, cross-eval sidecar); its checkpoint folder is deleted at the end of the job after a MATCH (v15 amendment A1, included from the start). About 2.3K inodes for the grid plus the repo. **Full design kept (σ = 0.1 and K = 10 kept).** The submit scripts refuse to submit if gdata has fewer free inodes than the submission needs plus a 1,500 margin. K = 10 or σ = 0.1 may be dropped only before the smoke, on resource grounds alone, as an amendment.

## 5. Analysis

Paired over seeds. K = 5 primary pooled over subsets (per-seed mean of A and B, df = 9), with A and B reported alongside. **Every null is reported with its MDE** (80% power, two-sided α = .05). Equivalence only via TOST, margin **±1 pp**.

Contrasts (each at every σ, both arms): **F** = M3 − M4, **T** = M3 − M5, **R** = M5 − M4, **BE** = M3 − M2, **Q** = M2 − M1.
Noise costs: **NC_m(σ)** = m(σ) − m(0), matched arm; **DEP_m(σ)** = m deployed at σ − m(0); **REC_m(σ)** = matched − deployed at σ.
**slope(X)** = per-seed OLS slope of X on σ over σ = 0, 0.1, 0.2, 0.3, 0.5, in **pp per 0.1 σ**, one-sample t over seeds. "slope_dep(X)" uses the deployed arm at the same five σ.

**Primary, Holm within each K (4 tests):**

| ID | Contrast | Question |
|---|---|---|
| F_dep@0.3 | deployed M3 − M4 at σ = 0.3, + TOST | Trained clean, is the quantum front-end more robust to noisy images than no front-end? |
| slope_dep(F) | trend of deployed F over σ | Does the robustness gap grow with noise? |
| F_mat@0.3 | matched M3 − M4 at σ = 0.3, + TOST | Trained on noisy images, does the quantum front-end beat no front-end? |
| T_dep@0.3 | deployed M3 − M5 at σ = 0.3, + TOST | Trained clean, does training the front-end make it more robust than freezing it? |

**Secondary (uncorrected, with MDE):** F, T, R, BE, Q at every σ, both arms; NC_m, DEP_m, REC_m for all five models; slopes of F, T, R, BE, Q (both arms) and of NC_m, DEP_m; per-subset results; seed SD vs σ; runs at the cap.
**Exploratory (labelled as such):** salt-and-pepper deployed robustness; the full train-σ × test-σ transfer matrix; T with and without seed 46 (worst M5 in every cell of v13.5, v14, v15); anything found after reading results.

## 6. Predictions (written before any v15.5 result)

- **P1 (F_dep, slope_dep(F)).** Deployed, the quantum front-end degrades **faster** than no front-end: F_dep@0.3 < 0 (point estimate) and slope_dep(F) < 0, **detected at both K**. Reason: the encoding is RY(πx), and cos(πx) is flat at x = 0, so the response to the clipped noise on the black background is quadratic and large. At σ = 0.3, the half of background pixels the noise pushes up move ⟨Z⟩ = cos(πx) by about (π·0.3)²/2 ≈ 0.44 on a [−1, 1] scale on average (≈ 0.22 averaged over all background pixels, since clipping leaves the other half at 0). This is a first-order single-qubit picture; Circuit 4 then entangles neighbouring pixels. M4's bilinear resize is linear and averages neighbouring pixels, shrinking the noise. Both see the same images; the quantum feature map turns a mean background shift into a large feature shift.
- **P2 (F_mat).** Trained on noisy images, F_mat@0.3 is not detected and its point estimate is ≤ 0, as in every clean condition (v12–v15). Noise augmentation lets every model learn the shifted background.
- **P3 (T_dep).** T_dep@0.3 ≥ 0 (point estimate). Not predicted to survive Holm. Seed 46's frozen front-end is predicted to be the worst deployed M5 at σ ≥ 0.2 (exploratory, as in v15 L4).
- **P4 (noise costs).** Unlike v15, pixel noise removes information, so it costs every model in both arms: NC_m(0.5) < 0 for all five models, detected for at least M4 and M3. DEP_m ≤ NC_m (deployed loses more than matched) at every σ ≥ 0.2, i.e. REC_m ≥ 0.
- **P5 (head, secondary).** As in v15 L3, the QNN head amplifies input perturbations: deployed BE grows with σ (slope_dep(BE) > 0, M2 degrades more than M3), and DEP_M1, DEP_M2 are the most negative at σ = 0.3.

## 7. How each outcome will be read

| Outcome | Reading |
|---|---|
| F_dep < 0 detected, slope_dep(F) < 0 | The quantum front-end is *less* robust to noisy images than no front-end. The trig encoding amplifies input noise. Strongest negative robustness result. |
| F_dep ≈ 0 (TOST), slope null | Same robustness as no front-end: the front-end is neutral to input noise. |
| F_dep > 0 detected | Contradicts P1: the front-end is more robust. Check T and R first (a frozen random circuit doing the same means fixed-feature smoothing, not training), then both subsets. Worded "more robust than no front-end", never "quantum advantage" (no matched classical front-end). |
| F_mat ≈ 0 while F_dep < 0 | A deployment message (train with noise augmentation), not an architecture one, as in v15. |
| T_dep > 0 | Training the front-end makes it more robust than freezing. Report with and without seed 46. |
| NC_m ≈ 0 at σ ≤ 0.3 for all models | MNIST at N = 500 is robust to this noise with augmentation; read the deployed arm and σ = 0.5 for differences. |

## 8. Run order, gates and stop rules

1. `setup_v155.sh`: build `qcq_cnn_v155` from `qcq_cnn_v14`, patch, verify. Must end in `ALL CHECKS PASSED`.
2. **Smoke** (`submit_v155_smoke.sh`): 75 jobs, one submission.
3. **Gate** (`check_v155.sh smoke`), per log: rc = 0; no continuation; loss = ce; n_classes and class list; n_obs = 15; params VERIFIED with the table above; determinism; max_epochs = 200; warmup; M5 freeze banner and exit self-check; `[V14] SPLIT` sizes 500·K / 125·K / 200·K, all classes, zero overlap; the job's σ printed in the header, in the noise line and in its result folder, all equal; noise-generator seeds equal to the formulas in §3; global-RNG self-check PASS; cross-evaluation self-check MATCH with every σ and salt-and-pepper level present, and its accuracy at the job's σ equal to the result JSON. Across logs: clean split hashes identical across models and σ at each (K, subset, seed); one clean test set per (K, subset); **one noisy test-set hash per (K, subset, σ)** across all models and seeds, in both the logs and the cross-evaluation sidecars; reproductions exact.
4. **Grid** (`submit_v155_grid.sh` refuses without the smoke gate's `.smoke_ok`), then **gate again** (`check_v155.sh grid`, writes `.grid_ok`) before any accuracy is read. `collect_v155.py` refuses to run without `.grid_ok`.

**Stop rules (fixed now):**
- A K = 5 A reproduction fails → stop. Nothing runs until the cause is found.
- Any σ = 0 grid run that has a stored counterpart and differs from it → the grid gate fails; investigated before any result is read.
- Any noise self-check fails, any noisy test-set hash differs within (K, subset, σ), or any cross-evaluation is not MATCH in the smoke → stop.
- Any smoke run stops at the 200-epoch cap → stop before the grid; revisited as a dated amendment.
- Any continuation (time budget hit) in the smoke → stop; walltime revisited as a resource amendment.
- In the grid: cap hits reported per cell; a cell with > 2/9 grid runs at the cap is labelled budget-limited.

## 9. What would NOT change this plan

A low or high headline number. No grid is re-run to improve a result. The noise seeds and levels are fixed above. Any change after the first smoke result is read is logged below as a dated amendment with its reason.

## Amendments

(none)
