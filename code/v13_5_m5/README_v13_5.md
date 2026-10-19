# v13.5: M5 (M3 with frozen front-end) at every v12 + v13 cell. How to run it

Read `PREREGISTRATION_v13_5.md` first. Pattern: **setup → smoke → gate → grid → gate.**
All K go in one smoke and one grid.

## 0. Local (PowerShell), from the folder you unzipped GADI_v13_5.zip into
```powershell
scp -r v13_5 yh2481@gadi-dm.nci.org.au:/g/data/wa66/yh2481/v135_pkg
```

## 1. Gadi: build the repo, patch, verify
```bash
ssh yh2481@gadi.nci.org.au
source /g/data/wa66/yh2481/miniconda3/etc/profile.d/conda.sh
conda activate qcqcnn
export PYTHONPATH=""
cd /g/data/wa66/yh2481/v135_pkg
bash setup_v135.sh                     # copies code from qcq_cnn_v13, patches, runs verify
cd /g/data/wa66/yh2481/qcq_cnn_v135
```
It must end in `ALL CHECKS PASSED`. **Paste the whole output the first time.**
The patch prints every line it changed (the `--model` choices and any
min-epochs rule), and the verify script prints the M3/M5 parameter table for
MNIST_LOG.md. If the patch aborts, paste what it asks for. Nothing is written
when it aborts.

Section 4 of the verify script runs two tiny training runs (M3 and M5, 2 epochs,
CPU) on the login node. It takes a few minutes. If it fails, paste the last
lines it prints; the full logs are in `verify_v135_dry_M3.log` / `_M5.log`.

## 2. Cheap checks on the login node
```bash
python train_gadi.py --help | grep -A1 -- '--model'          # must list M5
ls /g/data/wa66/yh2481/qcq_cnn_v12/results_v12_k5_A/result_M3_mnist_seed42_alpha0.0.json
ls -d /g/data/wa66/yh2481/qcq_cnn_v12/results_v12_k{3,4,5}_{A,B}
ls -d /g/data/wa66/yh2481/qcq_cnn_v13/results_v13_k*_w*
nci_account -P wa66                                          # SU left: 131 jobs, 30 of them K=10 (up to 4 h)
```
If any `ls` fails, fix `V12_ROOT` / `V13_ROOT` / `V10_RES` at the top of `pbs/config135.sh`.

## 3. Smoke (14 jobs: M5 seed 42 at all 13 cells + the M3 reproduction)
```bash
cd /g/data/wa66/yh2481/qcq_cnn_v135/pbs
./submit_v135_smoke.sh
qstat -u $USER
```
When all have left `qstat`:
```bash
./check_v135.sh smoke
```
Paste it. It must show the reproduction `MATCH` lines, the M3 vs M5
min-epochs line, and end in `ALL CHECKS PASSED`.

K=3/4/5 smoke jobs finish well before K=10. To start their grid early, gate them
on their own (K=5 must be included, it carries the reproduction):
```bash
./check_v135.sh smoke 3 4 5
./submit_v135_grid.sh 3 4 5          # 90 jobs
# later, when the K=10 smoke is done:
./check_v135.sh smoke 10
./submit_v135_grid.sh 10             # 27 jobs
```

## 4. Grid (117 jobs, all K at once)
```bash
./submit_v135_grid.sh
```
When done:
```bash
./check_v135.sh grid
```

## 5. Results and plots (after the grid gate passes for every K)
```bash
cd /g/data/wa66/yh2481/qcq_cnn_v135
python3 scripts/collect_v135.py
python3 scripts/plot_v135.py
tar czf v135_results.tgz v135_*.csv figures_v135 results_v135_*/ pbs/*.o* patch_v135.log verify_v135*.log
```
Local (PowerShell):
```powershell
scp yh2481@gadi-dm.nci.org.au:/g/data/wa66/yh2481/qcq_cnn_v135/v135_results.tgz .
```

## If something goes wrong
- A job vanishes from `qstat` in seconds: `cat pbs/v135s_k5Aw3.o*` (or whichever) and paste it.
- `FATAL: train_gadi.py is not the v13.5-patched file`: setup didn't finish. Rerun step 1 into a fresh folder.
- `M5 FRONT-END CHANGED DURING TRAINING`: stop. Paste the log. Do not submit the grid.
- `M5 min-epochs differ from M3`: stop and paste. The patch missed the rule; it's a one-line fix.
- A `CONT=1` job appears: the time budget was hit. Stop and look (`qdel` the chain).
- Inode quota: the v13 incident. Check `lquota` before submitting; `conda clean --all` freed ~33K files last time.

## Files
| File | Role |
|---|---|
| `patch_v135_m5.py` | Adds M5 to `train_gadi.py`. Anchored, refuses to patch twice, backup `.v13bak`. |
| `scripts/verify_v135_m5.py` | M1–M4 bit-exact vs pre-patch; M5 = M3 init, front-end frozen, never moves; dry run. |
| `pbs/config135.sh` | Pre-registered design: subsets, widths, seeds, paths. |
| `pbs/run135.pbs` | One job. |
| `pbs/submit_v135_smoke.sh`, `submit_v135_grid.sh` | Submit; the grid refuses without a passed smoke gate. |
| `pbs/check_v135.sh` | Integrity gate. |
| `scripts/collect_v135.py`, `plot_v135.py` | Contrasts (Holm, MDE, TOST) and the standard plot set. |
