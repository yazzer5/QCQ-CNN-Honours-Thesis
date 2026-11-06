# v14: low data (sample efficiency), M1–M5 at max width (w15). How to run it

Read `PREREGISTRATION_v14.md` first. Pattern: **setup → ONE smoke → gate → grid → gate → collect.**

| Stage | Jobs |
|---|---|
| Smoke (one submission) | 51 cells × seed 42 + 2 reproductions at w15 (M3 vs v13, M5 vs v13.5) = **53** |
| Grid | 459 (K=5: 306, K=10: 153) |
| N = 500 | M1, M4 run here (never run at w15); M2, M3 read from v13 w15; M5 from v13.5 w15 |

## 0. Local (PowerShell), from the folder you unzipped GADI_v14.zip into
```powershell
scp -r v14 yh2481@gadi-dm.nci.org.au:/g/data/wa66/yh2481/v14_pkg
```

## 1. Gadi: build, patch, verify
```bash
ssh yh2481@gadi.nci.org.au
source /g/data/wa66/yh2481/miniconda3/etc/profile.d/conda.sh && conda activate qcqcnn && export PYTHONPATH=""
lquota                                   # gdata inodes must be well under 1,018,000
nci_account -P wa66                      # SU left
cd /g/data/wa66/yh2481/v14_pkg && bash setup_v14.sh
```
Must end in `ALL CHECKS PASSED` (5–10 min: section 1 builds all models at w3 and w15, section 2 runs 11 split-only loads, sections 3–4 four tiny CPU trainings).

## 2. Cheap checks
```bash
cd /g/data/wa66/yh2481/qcq_cnn_v14
python train_gadi.py --help | grep -A1 -- '--model'
ls /g/data/wa66/yh2481/qcq_cnn_v13/results_v13_k5_A_w15/result_M3_mnist_seed42_alpha0.0.json \
   /g/data/wa66/yh2481/qcq_cnn_v135/results_v135_k5_A_w15/result_M5_mnist_seed42_alpha0.0.json
ls -d /g/data/wa66/yh2481/qcq_cnn_v13/results_v13_k{5_A,5_B,10_all}_w15 /g/data/wa66/yh2481/qcq_cnn_v135/results_v135_k{5_A,5_B,10_all}_w15
```
If any `ls` fails, fix the `*_ROOT` paths at the top of `pbs/config14.sh`.
To drop K=10 (SU only, **before** the smoke): set `GRID_KS="5"` in `pbs/config14.sh` and log an amendment.

## 3. Smoke (the one smoke test)
```bash
cd /g/data/wa66/yh2481/qcq_cnn_v14/pbs
./submit_v14_smoke.sh
qstat -u $USER | tail -3
```
When `qstat` is empty: `./check_v14.sh smoke`. Must show 8 `MATCH` lines and `ALL CHECKS PASSED`.

## 4. Grid
```bash
./submit_v14_grid.sh          # resumable: re-run it if qsub stops at a queue limit
```
When `qstat` is empty: `./check_v14.sh grid`.

## 5. Results
```bash
cd /g/data/wa66/yh2481/qcq_cnn_v14
python3 scripts/collect_v14.py | tee collect_v14.log
python3 scripts/plot_v14.py
find . -maxdepth 1 -type d -name 'ckpt_*' -empty -delete
tar czf v14_results.tgz v14_*.csv collect_v14.log figures_v14 results_v14_*/ pbs/*.o* patch_v14.log verify_v14*.log
```
```powershell
scp yh2481@gadi-dm.nci.org.au:/g/data/wa66/yh2481/qcq_cnn_v14/v14_results.tgz .
```

## If something goes wrong
- A job leaves `qstat` in seconds: `cat pbs/<jobname>.o*` and paste it.
- `SPLIT-AUDIT FAILED` or `models saw DIFFERENT ... sets`: stop and paste. Nothing about the data can be trusted until it's explained.
- `stopped at the N-epoch cap` in the smoke: pre-registered stop. Paste it; it becomes an amendment.
- `CONT=1`: time budget hit. Stop and paste (`qdel` the chain).
- `qsub` refuses part-way through the grid: wait for jobs to drain and run `./submit_v14_grid.sh` again; it skips finished and queued jobs.
- Held jobs ("H"): run `lquota` (the v13 inode incident). `conda clean --all` freed ~33K files last time.

## Files
| File | Role |
|---|---|
| `patch_v14_lowdata.py` | Wraps `load_dataset` with a split audit. No model/data/training change. Backup `.v135bak`. |
| `scripts/verify_v14.py` | Models bit-exact; splits correct at every N; audit invisible to training; M5 dry run. |
| `pbs/config14.sh` | Pre-registered design, paths, cap and walltime per N. |
| `pbs/names14.sh` | Job names (`v14s_` smoke, `v14r_` reproductions, `v14g` grid). |
| `pbs/run14.pbs` | One job. |
| `pbs/submit_v14_smoke.sh`, `submit_v14_grid.sh` | One smoke; grid refuses without a passed smoke gate. |
| `pbs/check_v14.sh` → `scripts/gate_v14.py` | Integrity gate. |
| `scripts/collect_v14.py`, `plot_v14.py` | Contrasts (Holm, MDE, TOST, slopes, AULC) and the plot set. |
