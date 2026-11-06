# v15.5: noisy input images (Gaussian pixel noise), M1–M5. How to run it

Read `PREREGISTRATION_v15_5.md` first. Pattern: **setup → ONE smoke → gate → grid → gate → collect → plots.**
Never read an accuracy before the gate prints `ALL CHECKS PASSED`.

| Stage | Jobs |
|---|---|
| Smoke (one submission) | 75 = seed 42 at every (K, subset, σ, model) cell. The 15 σ = 0 jobs are the reproductions (K = 5 A is the stop rule; the rest must also match) |
| Grid | 675 = seeds 43–51 at every cell |
| σ | 0, 0.1, 0.2, 0.3, 0.5 for **all five models** (the noise is on the image, so M4 is affected too). Every job also cross-evaluates at σ = 0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0 and salt-and-pepper p = 0.05, 0.1, 0.2 |

Each job leaves 3 files (log, result JSON, cross-eval sidecar); its checkpoint folder is deleted after a MATCH. The submit scripts read `lquota` and refuse if gdata has fewer free inodes than `jobs × 3 + 1500`.

## 0. Local (PowerShell), in the folder holding GADI_v155.zip
```powershell
Expand-Archive GADI_v155.zip -DestinationPath . -Force
scp -r v155 yh2481@gadi-dm.nci.org.au:/g/data/wa66/yh2481/v155_pkg
```

## 1. Gadi: build, patch, verify
```bash
ssh yh2481@gadi.nci.org.au
source /g/data/wa66/yh2481/miniconda3/etc/profile.d/conda.sh && conda activate qcqcnn && export PYTHONPATH=""
lquota
cd /g/data/wa66/yh2481/v155_pkg && bash setup_v155.sh
```
Must end in `ALL CHECKS PASSED` (5–15 min, CPU). It builds `qcq_cnn_v155` from `qcq_cnn_v14` (code only, data linked), patches `train_gadi.py`, then runs `scripts/verify_v155.py`.

📋 **PASTE #1: the whole output** (`lquota` and setup).

## 2. Cheap checks, then the smoke
```bash
cd /g/data/wa66/yh2481/qcq_cnn_v155
grep -c "V155_INPUT_NOISE" train_gadi.py
ls -d /g/data/wa66/yh2481/qcq_cnn_v14/results_v14_k{5_A,5_B,10_all}_n500 \
      /g/data/wa66/yh2481/qcq_cnn_v13/results_v13_k{5_A,5_B,10_all}_w15 \
      /g/data/wa66/yh2481/qcq_cnn_v135/results_v135_k{5_A,5_B,10_all}_w15
cd pbs && ./submit_v155_smoke.sh
qstat -u $USER | tail -3
```
If an `ls` fails, the `*_ROOT` paths at the top of `pbs/config155.sh` need fixing; tell me before submitting.

📋 **PASTE #2: the `grep`/`ls` output and the submit lines** (must say `inodes: ... OK` and `Submitted 75 smoke jobs`).

## 3. Smoke gate
When `qstat -u $USER` shows no v155 jobs:
```bash
cd /g/data/wa66/yh2481/qcq_cnn_v155/pbs && ./check_v155.sh smoke
```
Must show 5 × 4 `MATCH` lines for K = 5 A, `15/15`-style reproduction counts, and end in `ALL CHECKS PASSED`.

📋 **PASTE #3: the whole gate output.** Don't submit the grid until I've read it.

## 4. Grid
```bash
lquota
cd /g/data/wa66/yh2481/qcq_cnn_v155/pbs
./submit_v155_grid.sh          # refuses without .smoke_ok; resumable if qsub stops at a queue limit
qstat -u $USER | grep -c v155g
```
📋 **PASTE #4: `lquota` and the submit output** (`Submitted 675 grid jobs`, or how many before a queue limit).

While it runs, check now and then:
```bash
qstat -u $USER | grep -c v155g                                   # jobs left
lquota                                                           # inodes
ls /g/data/wa66/yh2481/qcq_cnn_v155/pbs/v155g_*.o* | wc -l       # logs written (675 at the end)
```

## 5. Grid gate
When no v155 jobs are left in `qstat`:
```bash
cd /g/data/wa66/yh2481/qcq_cnn_v155/pbs && ./check_v155.sh grid
```
📋 **PASTE #5: the whole gate output.**

## 6. Results (only after `ALL CHECKS PASSED` on the grid)
```bash
cd /g/data/wa66/yh2481/qcq_cnn_v155
python3 scripts/collect_v155.py | tee collect_v155.log
python3 scripts/plot_v155.py
python3 scripts/deep_v155.py
rm -rf ckpt_v155_*                      # normally already gone (deleted after each MATCH)
tar czf v155_results.tgz v155_*.csv collect_v155.log figures_v155 figures_v155_deep results_v155_*/ \
    pbs/*.o* patch_v155.log verify_v155.log noise_examples_v155.npz
```
```powershell
scp yh2481@gadi-dm.nci.org.au:/g/data/wa66/yh2481/qcq_cnn_v155/v155_results.tgz .
```
📋 **PASTE #6: `collect_v155.log`**, and upload `v155_results.tgz`.

## If something goes wrong
- A job leaves `qstat` in seconds: `cat pbs/<jobname>.o*` and paste it.
- `[V155] FATAL`, a self-check `FAIL`, or `selfcheck=MISMATCH`: stop and paste the log. Pre-registered stop rule.
- `noisy test set differs between runs` or `models/sigmas saw DIFFERENT clean ... sets`: stop and paste.
- A σ = 0 reproduction `DIFFERENT`: pre-registered stop rule. Paste; nothing else runs.
- `stopped at the 200-epoch cap` or `CONT=1` in the smoke: stop and paste (amendment).
- `REFUSED: gdata has N free inodes`: free inodes (old `ckpt_*` folders, scratch tarballs) or tell me; don't override.
- `qsub` refuses part-way through the grid: wait for jobs to drain, then `./submit_v155_grid.sh` again.
- Held jobs ("H"): run `lquota` (the v13 inode incident).

## Files
| File | Role |
|---|---|
| `patch_v155_input_noise.py` | Wraps `load_dataset` / `build_model` by name: adds the pixel noise to the datasets it returns; cross-evaluates at exit. No model, data or training line edited. Backup `.v14bak`. |
| `scripts/verify_v155.py` | Models bit-exact; noise wrapper deterministic, RNG-neutral, right statistics; dry runs incl. σ = 0 invisible to training and training noise reproducible. Writes `noise_examples_v155.npz`. |
| `pbs/config155.sh` | Pre-registered design and paths. |
| `pbs/run155.pbs`, `submit_v155_smoke.sh`, `submit_v155_grid.sh` | One job; one smoke; grid refuses without a passed smoke gate. Both refuse if inodes are short. |
| `pbs/check_v155.sh` → `scripts/gate_v155.py` | Integrity gate (stdlib only, Python ≥ 3.6). |
| `scripts/collect_v155.py` | Contrasts (Holm, MDE, TOST, slopes, both arms, exploratory s&p and seed-46). Refuses without `.grid_ok`. |
| `scripts/plot_v155.py` | Standard plot set → `figures_v155/` (PNG 200 dpi). |
| `scripts/deep_v155.py` | Exploratory D0–D8 → `figures_v155_deep/` with a what-and-why table. |
