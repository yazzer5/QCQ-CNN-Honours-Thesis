# v15.6: Fisher ratio and noise geometry of M3 / M4 / M5

Pre-registration: `PREREGISTRATION_v15_6.md` (project). Re-runs the v15.5 jobs for M3, M4, M5 at
training σ = 0 (deployed) and 0.3 (matched), K5 A, K5 B, K10 × seeds 42–51 (180 jobs), with one added
step inside the v15.5 cross-eval: `geom_v156.run` measures spread, shift, Fisher ratio and bleed at
Lpix, L0, L1, L2 (z) and L3 (logits). Training is untouched, so every job must reproduce v15.5 bit-exactly.

| File | What |
|---|---|
| `geom_v156.py` | the geometry pass (private RNGs only) |
| `patch_v156_geom.py` | inserts a 13-line hook in `_v155_xeval`, before `_v155_rng_restore(st)` |
| `verify_v156.py` | V1–V5: byte-identity minus block, RNG audit, maths, real M3/M4/M5 hook test |
| `setup_v156.sh` | builds `qcq_cnn_v156` (code copy, data link, patch, verify) |
| `gate_v156.py` | smoke/grid gate: bit-exact v15.5 reproduction + geometry checks; no numbers printed |
| `analyse_v156.py` | CSVs, primaries (Holm), contrasts, figures; refuses without `.grid_ok` |
| `submit_v156_*.sh` | added after PASTE A (needs the real v15.5 submit layer) |

## 1. Upload and set up

PowerShell (local):
```powershell
scp v156_pkg.zip yh2481@gadi-dm.nci.org.au:/g/data/wa66/yh2481/
```
Gadi:
```bash
source /g/data/wa66/yh2481/miniconda3/etc/profile.d/conda.sh && conda activate qcqcnn && export PYTHONPATH=""
cd /g/data/wa66/yh2481 && unzip -o v156_pkg.zip && cd v156_pkg
bash setup_v156.sh 2>&1 | tee ~/setup_v156.out
```
📋 **PASTE A:** all of `~/setup_v156.out`. It must contain `ALL CHECKS PASSED`. It also prints the v15.5
submit layer, which `submit_v156_smoke.sh` / `submit_v156_grid.sh` (in `pbs/`) are built on.

## 2. Smoke (18 jobs: seed 42, every model × arm × cell)  — after PASTE A
## 3. Gate
```bash
cd /g/data/wa66/yh2481/qcq_cnn_v156 && python3 gate_v156.py smoke $PWD ../qcq_cnn_v155
```
📋 **PASTE B.** Must end `ALL CHECKS PASSED` (bit-exact reproduction of all 18 v15.5 runs).

## 4. Grid (162 more jobs; refuses without `.smoke_ok`) → gate grid → 📋 PASTE C
## 5. Analyse
```bash
python3 analyse_v156.py $PWD
```
📋 **PASTE D:** the printed primaries. Then copy `v156_out/` locally:
```powershell
scp -r yh2481@gadi-dm.nci.org.au:/g/data/wa66/yh2481/qcq_cnn_v156/v156_out .
```

## Optional inode headroom (nothing queued)
```bash
conda clean --all -y && lquota
```
