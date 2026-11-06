# v13: QNN readout width. How to run it

Read `PREREGISTRATION_v13.md` first. Pattern per K: **smoke → gate → grid → gate.**
Do K=5 first (cheap, ~1 h jobs), then K=10 (up to 4 h jobs).

## 0. Local (PowerShell), from the folder you unzipped GADI_v13.zip into
```powershell
scp -r v13 yh2481@gadi-dm.nci.org.au:/g/data/wa66/yh2481/v13_pkg
```

## 1. Gadi: build the repo, patch, verify
```bash
ssh yh2481@gadi.nci.org.au
source /g/data/wa66/yh2481/miniconda3/etc/profile.d/conda.sh
conda activate qcqcnn
export PYTHONPATH=""
cd /g/data/wa66/yh2481/v13_pkg
bash setup_v13.sh                      # copies code from qcq_cnn_v12, patches, runs verify
cd /g/data/wa66/yh2481/qcq_cnn_v13
```
It must end in `ALL CHECKS PASSED`. **Paste the whole output the first time**:
section 4 prints the parameter table for MNIST_LOG.md. If the patch aborts,
paste what it asks for. Nothing is written when it aborts.

## 2. Cheap checks on the login node
```bash
python train_gadi.py --help | grep -A1 -- '--n_obs'          # must list 1, 3, 6, 15
ls /g/data/wa66/yh2481/qcq_cnn_v12/results_v12_k5_A/result_M2_mnist_seed42_alpha0.0.json
ls /g/data/wa66/yh2481/v10_grid/results_v10_mnist10/result_M2_mnist_seed42_alpha0.0.json
# K=10 walltime: v13 uses 4 h. Confirm v10's 10-class jobs fit in that:
grep -h 'walltime' /g/data/wa66/yh2481/v10_grid/pbs/run10_mnist10_grid.pbs
grep -h 'Job start\|Job end' /g/data/wa66/yh2481/v10_grid/pbs/v10_m10*.o* | head -6
nci_account -P wa66                                          # SU left
```
If either `ls` fails, fix `V12_ROOT` / `V10_RES` at the top of `pbs/config13.sh`.

## 3. K=5: smoke (5 jobs)
```bash
cd /g/data/wa66/yh2481/qcq_cnn_v13/pbs
./submit_v13_smoke.sh 5
qstat -u $USER
```
When all have left `qstat`:
```bash
./check_v13.sh 5 smoke
```
Paste it. It must show the reproduction `MATCH` lines and end in `ALL CHECKS PASSED`.

## 4. K=5: grid (76 jobs)
```bash
./submit_v13_grid.sh 5
```
When done:
```bash
./check_v13.sh 5 grid
```

## 5. K=10: smoke (3 jobs), then grid (58 jobs)
You can start this as soon as the K=5 grid is submitted.
```bash
./submit_v13_smoke.sh 10
./check_v13.sh 10 smoke          # after the jobs finish; paste it
./submit_v13_grid.sh 10          # only after ALL CHECKS PASSED
./check_v13.sh 10 grid           # after the jobs finish
```
**If the K=10 reproduction says DIFFERENT:** stop, paste the output. That is
the pre-registered fallback A1-v13 (the top-up becomes all 10 seeds). Don't
change anything yourself.

## 6. Results and plots (after both grids pass)
```bash
cd /g/data/wa66/yh2481/qcq_cnn_v13
python3 scripts/collect_v13.py
python3 scripts/plot_v13.py
tar czf v13_results.tgz v13_*.csv figures_v13 results_v13_*/ pbs/*.o* verify_v13.log
```
Local (PowerShell):
```powershell
scp yh2481@gadi-dm.nci.org.au:/g/data/wa66/yh2481/qcq_cnn_v13/v13_results.tgz .
```

## If something goes wrong
- A job vanishes from `qstat` in seconds: `cat pbs/v13s_k5Aw15.o*` (or whichever) and paste it.
- `FATAL: qnn_torch.py is not the v13-patched file`: setup didn't finish. Rerun step 1 into a fresh folder.
- A `CONT=1` job appears: the time budget was hit. Stop and look (`qdel` the chain).
- `check_v13.sh` prints one `VERIFIED` line per model per width. Paste them the first time.
