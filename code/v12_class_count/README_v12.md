# v12: class-count sweep. How to run it

Pattern per class count: **smoke (2 jobs) -> gate -> grid (78 jobs) -> gate.**
Do K=3 first, then 4, then 5. You can start the next K's smoke as soon as the
previous K's grid is submitted.

## 0. Local (PowerShell), from the folder you unzipped GADI_v12.zip into
```powershell
scp -r v12 yh2481@gadi-dm.nci.org.au:/g/data/wa66/yh2481/v12_pkg
```

## 1. Gadi: build the repo
```bash
ssh yh2481@gadi.nci.org.au
cd /g/data/wa66/yh2481/v12_pkg
bash setup_v12.sh /g/data/wa66/yh2481/v10_grid     # or your v11 folder if its code is newer
cd /g/data/wa66/yh2481/qcq_cnn_v12
source /g/data/wa66/yh2481/miniconda3/etc/profile.d/conda.sh
conda activate qcqcnn
export PYTHONPATH=""
```

## 2. Cheap checks on the login node (no training)
```bash
# the preflight must accept 3-, 4-, 5-class subsets
python scripts/preflight_data.py --data_root ./data --dataset mnist --classes 1,4,9       --n_train 1500 --n_val 375 --n_test 600
python scripts/preflight_data.py --data_root ./data --dataset mnist --classes 0,2,3,8     --n_train 2000 --n_val 500 --n_test 800
python scripts/preflight_data.py --data_root ./data --dataset mnist --classes 1,3,5,8,9   --n_train 2500 --n_val 625 --n_test 1000
# train_gadi.py must take --n_train / --n_val (v10 only ever passed --n_test)
python train_gadi.py --help | grep -E -- '--n_train|--n_val|--n_test'
```
If `--n_train`/`--n_val` are missing from `--help`, stop and paste the output.

## 3. Check the PBS resource lines match v10's (which ran clean)
```bash
diff <(grep '^#PBS -[Pql]' /g/data/wa66/yh2481/v10_grid/pbs/run10_mnist10_grid.pbs) \
     <(grep '^#PBS -[Pql]' pbs/run12_classes.pbs)
```
If walltime is the only difference, that's fine. If anything else differs
(queue, ncpus, mem, storage), copy v10's line into `pbs/run12_classes.pbs`.

## 4. Smoke, K=3
```bash
cd pbs
./submit_v12_smoke.sh 3
qstat -u $USER
```
When both jobs have left `qstat`:
```bash
./check_v12.sh 3 smoke
```
Paste the output. It must end in `ALL CHECKS PASSED`, and the smoke accuracy
must be well above chance (33%).

## 5. Grid, K=3
```bash
./submit_v12_grid.sh 3
qstat -u $USER | tail -3
```
When everything is done:
```bash
./check_v12.sh 3 grid
```

## 6. Repeat 4-5 for K=4 and K=5
```bash
./submit_v12_smoke.sh 4 ;  ./check_v12.sh 4 smoke ;  ./submit_v12_grid.sh 4 ;  ./check_v12.sh 4 grid
./submit_v12_smoke.sh 5 ;  ./check_v12.sh 5 smoke ;  ./submit_v12_grid.sh 5 ;  ./check_v12.sh 5 grid
```
(Run each one only after the previous step's jobs finish and its check passes.)

## 7. Results + plots (after all grids pass the gate)
```bash
cd /g/data/wa66/yh2481/qcq_cnn_v12
python scripts/collect_v12.py --v10 /g/data/wa66/yh2481/v10_grid/results_v10_mnist10
python scripts/plot_v12.py
tar czf v12_results.tgz v12_*.csv figures_v12 results_v12_k*/ pbs/*.o*
```
Local (PowerShell):
```powershell
scp yh2481@gadi-dm.nci.org.au:/g/data/wa66/yh2481/qcq_cnn_v12/v12_results.tgz .
```

## If something goes wrong
- A job vanishes from `qstat` in seconds: `cat pbs/v12s_k3A.o*` and paste it.
- A `CONT=1` job appears: epoch timing is off. Stop and look (`qdel` the chain).
- Parameter counts: `check_v12.sh` prints one `VERIFIED` line per model. Paste
  these the first time, so the K=3/4/5 padding gets recorded in MNIST_LOG.md.
