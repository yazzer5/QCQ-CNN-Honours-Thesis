# v15: IBM-calibrated front-end noise, M1–M5. How to run it

Read `PREREGISTRATION_v15.md` first. Pattern: **setup → ONE smoke → gate → grid → gate → collect.**

| Stage | Jobs |
|---|---|
| Smoke (one submission) | 63 = seed 42 at every (K, subset, alpha, model) cell. The 15 alpha=0 jobs are the reproductions (K=5 A gated, the rest reported) |
| Grid | 567 = seeds 43–51 at every cell |
| alpha | 0, 0.5, 1, 2, 4 for M1, M2, M3, M5; M4 at 0 only (no quantum front-end). Every job also cross-evaluates its model at 0, 0.5, 1, 2, 4, 8, 16 |

Each job writes 3 files (log, result JSON, cross-eval sidecar) plus a checkpoint dir: roughly 2.5–3K inodes for the whole grid.

## 0. Local (PowerShell), in the folder holding GADI_v15.zip
```powershell
Expand-Archive GADI_v15.zip -DestinationPath . -Force
scp -r v15 yh2481@gadi-dm.nci.org.au:/g/data/wa66/yh2481/v15_pkg
```

## 1. Gadi: build, patch, verify
```bash
ssh yh2481@gadi.nci.org.au
source /g/data/wa66/yh2481/miniconda3/etc/profile.d/conda.sh && conda activate qcqcnn && export PYTHONPATH=""
lquota                                   # gdata inodes must be well under 1,018,000
nci_account -P wa66                      # SU left
cd /g/data/wa66/yh2481/v15_pkg && bash setup_v15.sh
```
Must end in `ALL CHECKS PASSED` (5–10 min). It builds `qcq_cnn_v15` from `qcq_cnn_v14` (code only, data linked), patches `train_gadi.py`, re-checks the calibration and the Aer reference in `qcqcnn_noise` if that env has Qiskit Aer, then runs `scripts/verify_v15.py`.

📋 **PASTE #1: the whole output** (`lquota`, SU, setup).

## 2. Cheap checks, then the smoke
```bash
cd /g/data/wa66/yh2481/qcq_cnn_v15
python train_gadi.py --help | grep -A1 -- '--model'
ls /g/data/wa66/yh2481/qcq_cnn_v14/results_v14_k5_A_n500/result_M{1,4}_mnist_seed42_alpha0.0.json \
   /g/data/wa66/yh2481/qcq_cnn_v13/results_v13_k5_A_w15/result_M{2,3}_mnist_seed42_alpha0.0.json \
   /g/data/wa66/yh2481/qcq_cnn_v135/results_v135_k5_A_w15/result_M5_mnist_seed42_alpha0.0.json
ls -d /g/data/wa66/yh2481/qcq_cnn_v14/results_v14_k{5_A,5_B,10_all}_n500 \
      /g/data/wa66/yh2481/qcq_cnn_v13/results_v13_k{5_A,5_B,10_all}_w15 \
      /g/data/wa66/yh2481/qcq_cnn_v135/results_v135_k{5_A,5_B,10_all}_w15
```
If any `ls` fails, fix the `*_ROOT` paths at the top of `pbs/config15.sh` (or tell me the right folder names).
To drop K=10 (SU only, **before** the smoke): set `GRID_KS="5"` in `pbs/config15.sh` and log an amendment.

```bash
cd /g/data/wa66/yh2481/qcq_cnn_v15/pbs
./submit_v15_smoke.sh
qstat -u $USER | tail -3
```
📋 **PASTE #2: the `ls` output and the submit line** (must say `Submitted 63 smoke jobs`).

## 3. Smoke gate
When `qstat -u $USER` is empty:
```bash
cd /g/data/wa66/yh2481/qcq_cnn_v15/pbs && ./check_v15.sh smoke
```
Must show 5 × 4 `MATCH` lines for K=5 A and end in `ALL CHECKS PASSED`.
📋 **PASTE #3: the whole gate output.** Don't submit the grid until I've read it.

## 4. Grid
```bash
cd /g/data/wa66/yh2481/qcq_cnn_v15/pbs
./submit_v15_grid.sh          # resumable: re-run it if qsub stops at a queue limit
qstat -u $USER | wc -l ; lquota
```
📋 **PASTE #4: the submit output** (`Submitted 567 grid jobs`, or how many before a queue limit).

While it runs, check now and then:
```bash
qstat -u $USER | grep -c " Q \| R "        # jobs left
lquota                                      # inodes
ls /g/data/wa66/yh2481/qcq_cnn_v15/pbs/v15g_*.o* | wc -l     # logs written (567 at the end)
```

## 5. Grid gate
When `qstat` is empty:
```bash
cd /g/data/wa66/yh2481/qcq_cnn_v15/pbs && ./check_v15.sh grid
```
📋 **PASTE #5: the whole gate output.**

## 6. Results (only after `ALL CHECKS PASSED` on the grid)
```bash
cd /g/data/wa66/yh2481/qcq_cnn_v15
python3 scripts/collect_v15.py | tee collect_v15.log
python3 scripts/plot_v15.py
rm -rf ckpt_v15_*                      # checkpoints are only used inside each job; frees inodes
tar czf v15_results.tgz v15_*.csv collect_v15.log figures_v15 results_v15_*/ pbs/*.o* patch_v15.log verify_v15.log calib/
```
```powershell
scp yh2481@gadi-dm.nci.org.au:/g/data/wa66/yh2481/qcq_cnn_v15/v15_results.tgz .
```
📋 **PASTE #6: `collect_v15.log`**, and upload `v15_results.tgz`.

## If something goes wrong
- A job leaves `qstat` in seconds: `cat pbs/<jobname>.o*` and paste it.
- `[V15] FATAL`, a self-check `FAIL`, or `selfcheck=MISMATCH`: stop and paste the log. Pre-registered stop rule.
- `models/alphas saw DIFFERENT ... sets`: stop and paste. Noise must never touch the data.
- A K=5 A reproduction `DIFFERENT`: pre-registered stop rule. Paste; nothing else runs.
- `stopped at the 200-epoch cap` or `CONT=1` in the smoke: stop and paste (amendment).
- `qsub` refuses part-way through the grid: wait for jobs to drain, then `./submit_v15_grid.sh` again.
- Held jobs ("H"): run `lquota` (the v13 inode incident).

## Files
| File | Role |
|---|---|
| `quanv_noise_v15.py` | The noise model: ibm_brisbane channels as Pauli-transfer matrices on the two front-ends; alpha = 0 runs the original forward. |
| `calib/ibm_brisbane_calib.json` | Pre-registered calibration (FakeBrisbane snapshot 2025-02-26), hash-checked by every job. |
| `calib/aer_reference_v15.npz` | Qiskit Aer reference used by verify (setup regenerates it on Gadi if Aer is available). |
| `patch_v15_noise.py` | Wraps `build_model` / `load_dataset` by name: installs the noise, cross-evaluates at exit. No model, data or training line edited. Backup `.v14bak`. |
| `scripts/verify_v15.py` | Models bit-exact with noise off; noise model == Aer / IBM `from_backend`; install invisible to RNG and weights; dry runs incl. alpha=0 invisible to training. |
| `scripts/verify_noise_v15.py` | The noise-model part of verify (torch only). |
| `scripts/extract_ibm_calibration.py`, `scripts/make_aer_reference_v15.py` | Qiskit-only: make / re-check the calibration and the Aer reference. |
| `pbs/config15.sh` | Pre-registered design and paths. |
| `pbs/run15.pbs`, `submit_v15_smoke.sh`, `submit_v15_grid.sh` | One job; one smoke; grid refuses without a passed smoke gate. |
| `pbs/check_v15.sh` → `scripts/gate_v15.py` | Integrity gate. |
| `scripts/collect_v15.py`, `scripts/plot_v15.py` | Contrasts (Holm, MDE, TOST, slopes, both arms) and the plot set. |
