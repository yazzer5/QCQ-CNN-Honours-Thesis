# v16a: M2c (closed-form QNN head) + z-dumps: run guide

73 jobs: 11 smoke + 62 grid. Pre-registration: `PREREGISTRATION_v16a.md` (already in the project; it is copied into the repo and the smoke refuses without it).

At every 📋 **PASTE** checkpoint, paste the output and **wait for my reply before going on.**

Gadi session setup (every new login):
```bash
source /g/data/wa66/yh2481/miniconda3/etc/profile.d/conda.sh && conda activate qcqcnn && export PYTHONPATH=""
```

## 1. Upload the package

**Local (PowerShell)**, in the folder where you saved `GADI_v16a.zip`:
```powershell
scp GADI_v16a.zip yh2481@gadi-dm.nci.org.au:/g/data/wa66/yh2481/
ssh yh2481@gadi.nci.org.au
```

**Gadi (bash):**
```bash
cd /g/data/wa66/yh2481 && unzip -o GADI_v16a.zip && ls v16a_pkg
```

## 2. Setup: build, patch, verify

**Gadi:**
```bash
source /g/data/wa66/yh2481/miniconda3/etc/profile.d/conda.sh && conda activate qcqcnn && export PYTHONPATH=""
lquota; nci_account -P wa66 | head -20
cd /g/data/wa66/yh2481/v16a_pkg && bash setup_v16a.sh
```
This builds `qcq_cnn_v16a` from `qcq_cnn_v15` (code only, data linked), patches `train_gadi.py` and runs `scripts/verify_v16a.py`. The verify step checks the real circuit against the closed form (the span check), the M2c surgery, parameter counts, bit-identity of M1–M5, and the dry runs. It takes a few minutes on the login node.

📋 **PASTE #1: the whole output.** It must end in `ALL CHECKS PASSED`.

## 3. Cheap checks, then the smoke

**Gadi:**
```bash
ls -d /g/data/wa66/yh2481/qcq_cnn_v13/results_v13_k{5_A,5_B,10_all}_w15 \
      /g/data/wa66/yh2481/qcq_cnn_v14/results_v14_k{5_A,5_B,10_all}_n50
ls /g/data/wa66/yh2481/qcq_cnn_v14/pbs/v14g10X50M2s4[3-5].o* 2>/dev/null | head -3
cd /g/data/wa66/yh2481/qcq_cnn_v16a/pbs && ./submit_v16a_smoke.sh && qstat -u $USER | tail -4
```
The submit script refuses if any of the 120 stored M2/M3 comparator files is missing. If it does, the `*_ROOT` paths at the top of `pbs/config16a.sh` need fixing; paste what it printed.

📋 **PASTE #2: the `ls` output and the submit line** (`Submitted 11 smoke jobs`).

## 4. Smoke gate

When `qstat -u $USER` is empty (about 30–60 min):

**Gadi:**
```bash
cd /g/data/wa66/yh2481/qcq_cnn_v16a/pbs && ./check_v16a.sh smoke
```
It must show the 5 reproductions (`MATCH` on every line) and end in `ALL CHECKS PASSED`. It prints epochs and job times only, not accuracies. The times decide whether the 1 h walltime stands.

📋 **PASTE #3: the whole gate output.** Don't submit the grid until I've replied.

## 5. Grid

**Gadi:**
```bash
cd /g/data/wa66/yh2481/qcq_cnn_v16a/pbs && ./submit_v16a_grid.sh && qstat -u $USER | wc -l && lquota
```
It is resumable: if qsub stops at a queue limit, run it again later.

📋 **PASTE #4: the submit output** (`Submitted 62 grid jobs`, or how many before a queue limit).

Progress, any time:
```bash
qstat -u $USER | grep -c " Q \| R "                                   # jobs left
ls /g/data/wa66/yh2481/qcq_cnn_v16a/pbs/v16g*.o* 2>/dev/null | wc -l  # logs written (62 at the end)
```

## 6. Grid gate: before any accuracy is read

**Gadi:**
```bash
cd /g/data/wa66/yh2481/qcq_cnn_v16a/pbs && ./check_v16a.sh grid
```
📋 **PASTE #5: the whole gate output.** Don't run collect until I've replied.

## 7. Results

**Gadi:**
```bash
cd /g/data/wa66/yh2481/qcq_cnn_v16a
python3 scripts/collect_v16a.py | tee collect_v16a.log
python3 scripts/plot_v16a.py
rm -rf ckpt_* ; find . -name __pycache__ -prune -exec rm -rf {} +
tar czf v16a_results.tgz v16a_*.csv collect_v16a.log figures_v16a results_v16a_*/ pbs/v16*.o* patch_v16a.log verify_v16a.log PREREGISTRATION_v16a.md
ls -la v16a_results.tgz
```
**Local (PowerShell):**
```powershell
scp yh2481@gadi-dm.nci.org.au:/g/data/wa66/yh2481/qcq_cnn_v16a/v16a_results.tgz .
tar -xzf v16a_results.tgz
```
📋 **PASTE #6: `collect_v16a.log`**, and upload `v16a_results.tgz` (it contains the z-dumps v16c needs: `results_v16a_z_k10_all_n{50,500}/v16_z_*.npz`).

## If something goes wrong

- A job leaves `qstat` within seconds: `cat pbs/<jobname>.o*` and paste it.
- The patch refuses: paste its output and
  `grep -n "^def build_model\|^def load_dataset\|^def expected_params\|class QNNBackendTorch" /g/data/wa66/yh2481/qcq_cnn_v15/train_gadi.py /g/data/wa66/yh2481/qcq_cnn_v15/qnn_torch.py`
- Verify fails: paste the whole `verify_v16a.log`. If it names dry-run logs, paste `tail -40` of the one that failed.
- To start over cleanly: `mv /g/data/wa66/yh2481/qcq_cnn_v16a /g/data/wa66/yh2481/qcq_cnn_v16a_old` (only before the smoke is submitted).

## Files

| File | Role |
|---|---|
| `closed_form_zz_v16.py` | The 12 features in closed form (`ZZTrig12`) and an independent numpy circuit reference. **Shared with v16b.** |
| `v16_tools.py` | M2 → M2c surgery, z-layer lookup, test-set prediction with z capture. **Shared with v16b.** |
| `patch_v16a.py` | Inserts the `V16A_HEAD_SAVEZ` block into `train_gadi.py` (flags `--v16_head m2c`, `--save_z`, `--v16_sidecar`). Edits no existing line. Backup `train_gadi.py.v15bak`. |
| `scripts/verify_v16a.py` | Pre-flight; must end in `ALL CHECKS PASSED`. |
| `pbs/config16a.sh` | Pre-registered design and paths. |
| `pbs/run16a.pbs`, `submit_v16a_smoke.sh`, `submit_v16a_grid.sh` | One job; the smoke (11); the grid (62, refuses without the smoke gate). |
| `pbs/check_v16a.sh` → `scripts/gate_v16a.py` | Integrity gate (stdlib only). |
| `scripts/collect_v16a.py`, `scripts/plot_v16a.py` | Pre-registered analysis; standard figure set (`figures_v16a/`). |

**For v16b (Part 2, other chat):** build from `qcq_cnn_v15`, copy `closed_form_zz_v16.py` and `v16_tools.py`, and import `from closed_form_zz_v16 import ZZTrig12, zz_trig_features, zz_phase_diffs` for the twin. The v16a block is self-contained, so the two patches don't conflict.
