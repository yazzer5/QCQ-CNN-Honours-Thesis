# v16b-ext: M4-trim and BFR-ZZ controls at K10 N50 (amendment A3): run guide

23 jobs: 3 smoke (M4-trim s42, BFR-ZZ s42, plus a gated reproduction of BFR s42) and 18 grid. The pre-registration, `PREREGISTRATION_v16b.md`, now includes A3. That file is in the project and in this package, and the smoke refuses to run without A3.

At every 📋 **PASTE** checkpoint, paste the output and **wait for my reply before going on.**

Gadi session setup (every new login):
```bash
source /g/data/wa66/yh2481/miniconda3/etc/profile.d/conda.sh && conda activate qcqcnn && export PYTHONPATH=""
```

## 1. Upload
**Local (PowerShell)**, in the folder holding `GADI_v16bx.zip`:
```powershell
scp GADI_v16bx.zip yh2481@gadi-dm.nci.org.au:/g/data/wa66/yh2481/
ssh yh2481@gadi.nci.org.au
```
**Gadi:**
```bash
cd /g/data/wa66/yh2481 && unzip -o GADI_v16bx.zip && ls v16bx_pkg v16bx_pkg/scripts v16bx_pkg/pbs
```

## 2. Setup: build `qcq_cnn_v16bx`, patch, verify
**Gadi:**
```bash
source /g/data/wa66/yh2481/miniconda3/etc/profile.d/conda.sh && conda activate qcqcnn && export PYTHONPATH=""
cd /g/data/wa66/yh2481/v16bx_pkg && bash setup_v16bx.sh
```
This builds a new repo from `qcq_cnn_v16b`'s code; the v16b repo and its results are untouched. It then patches `train_gadi.py` (adding the `--v16bx_head` flag) and runs `scripts/verify_v16bx.py`. Verify checks:
- M1–M5, M2c and BFR are bit-identical to the v16b file.
- BFR-ZZ matches an independent statevector and equals BFR exactly with ζ switched off.
- The parameter table: M4-trim 1643, BFR-ZZ 1799.
- M4-trim equals BFR's classical path, and BFR-ZZ's init equals BFR's.
- Dry runs.

📋 **PASTE #1: the whole output** (it must contain `ALL CHECKS PASSED`).

## 3. Smoke (3 jobs)
**Gadi:**
```bash
cd /g/data/wa66/yh2481/qcq_cnn_v16bx/pbs && ./submit_v16bx_smoke.sh && qstat -u $USER | tail -5
```
📋 **PASTE #2: the submit lines.** Expect `Stored comparators: all 30 present` and `Submitted 3 smoke jobs`.

## 4. Smoke gate (when `qstat -u $USER` is empty)
**Gadi:**
```bash
cd /g/data/wa66/yh2481/qcq_cnn_v16bx/pbs && ./check_v16bx.sh smoke
```
It must show the 4 reproduction lines as `MATCH` and end in `ALL CHECKS PASSED`. It prints no accuracies.

📋 **PASTE #3: the whole gate output.**

## 5. Grid (18 jobs)
**Gadi:**
```bash
cd /g/data/wa66/yh2481/qcq_cnn_v16bx/pbs && ./submit_v16bx_grid.sh && qstat -u $USER | wc -l
```
📋 **PASTE #4: the submit output.**

## 6. Grid gate (when `qstat` is empty), before any accuracy is read
**Gadi:**
```bash
cd /g/data/wa66/yh2481/qcq_cnn_v16bx/pbs && ./check_v16bx.sh grid
```
📋 **PASTE #5: the whole gate output.**

## 7. Results
**Gadi:**
```bash
cd /g/data/wa66/yh2481/qcq_cnn_v16bx
python3 scripts/collect_v16bx.py | tee collect_v16bx.log
python3 scripts/plot_v16bx.py
rm -rf ckpt_* ; find . -name __pycache__ -prune -exec rm -rf {} +
tar czf v16bx_results.tgz v16bx_*.csv collect_v16bx.log figures_v16bx results_v16bx_*/ pbs/v16x*.o* patch_v16bx.log verify_v16bx.log PREREGISTRATION_v16b.md scripts/plot_v16bx.py
ls -la v16bx_results.tgz
```
**Local (PowerShell):**
```powershell
scp yh2481@gadi-dm.nci.org.au:/g/data/wa66/yh2481/qcq_cnn_v16bx/v16bx_results.tgz .
tar -xzf v16bx_results.tgz
```
📋 **PASTE #6: `collect_v16bx.log`, and attach `v16bx_results.tgz`.**

## If something goes wrong
- A job exits within seconds: `cat pbs/<jobname>.o*` and paste it.
- Setup refuses because `qcq_cnn_v16bx` exists: `mv /g/data/wa66/yh2481/qcq_cnn_v16bx /g/data/wa66/yh2481/qcq_cnn_v16bx_old` (only before the smoke is submitted).

## Files
| File | Role |
|---|---|
| `bfr_ext_v16b.py` | M4-trim (`TrimHead`) and BFR-ZZ (`BFRQuantumZZ`, real-arithmetic complex simulator) plus the surgery. Imports `bfr_head_v16b` without changing it. |
| `patch_v16bx.py` | Adds the `V16BX_EXT_HEAD` block (flag `--v16bx_head m4trim|bfrzz`). Backup `train_gadi.py.v16bbak`. |
| `setup_v16bx.sh`, `scripts/verify_v16bx.py` | Build, patch, verify. |
| `pbs/config16bx.sh`, `run16bx.pbs`, `submit_v16bx_smoke.sh`, `submit_v16bx_grid.sh`, `check_v16bx.sh` → `scripts/gate_v16bx.py` | Jobs and the integrity gate. |
| `scripts/collect_v16bx.py`, `scripts/plot_v16bx.py` | Pre-registered A3 analysis (C1–C3, Holm over 3, E1, E2) and the figures (`figures_v16bx/`, 200 dpi). |
| `PREREGISTRATION_v16b.md` | With amendment A3. |
