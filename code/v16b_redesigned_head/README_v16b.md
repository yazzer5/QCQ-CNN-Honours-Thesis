# v16b: Design 1 (BFR head) vs M4 and M2-w15 (reused): run guide

61 jobs: 7 smoke (6 BFR + 1 M4 reproduction) + 54 grid. Pre-registration: `PREREGISTRATION_v16b.md` (also in the project; the smoke refuses without it in the repo).

At every 📋 **PASTE** checkpoint, paste the output and **wait for my reply before going on.**

Gadi session setup (every new login):
```bash
source /g/data/wa66/yh2481/miniconda3/etc/profile.d/conda.sh && conda activate qcqcnn && export PYTHONPATH=""
```

## 1. Upload the package

**Local (PowerShell)**, in the folder where you saved `GADI_v16b.zip`:
```powershell
scp GADI_v16b.zip yh2481@gadi-dm.nci.org.au:/g/data/wa66/yh2481/
ssh yh2481@gadi.nci.org.au
```

**Gadi (bash):**
```bash
cd /g/data/wa66/yh2481 && unzip -o GADI_v16b.zip && ls v16b_pkg v16b_pkg/scripts v16b_pkg/pbs
```

## 2. Setup: build, patch, verify

**Gadi:**
```bash
source /g/data/wa66/yh2481/miniconda3/etc/profile.d/conda.sh && conda activate qcqcnn && export PYTHONPATH=""
cd /g/data/wa66/yh2481/v16b_pkg && bash setup_v16b.sh
```
This builds `qcq_cnn_v16b` from `qcq_cnn_v16a` (code only, data linked), patches `train_gadi.py` and runs `scripts/verify_v16b.py`. It takes a few minutes on the login node. Verify checks:
- M1–M5 and M2c are bit-identical to the v16a file;
- the BFR circuit matches an independent statevector and Qiskit;
- the structural facts (6 zero Paulis, angle offsets, bounded frequency);
- the surgery: parameter table, RNG and init pairing with M4, and the function class;
- the gradient regimes at g = 0 and g = 1;
- dry runs, including M4 through the patched file equalling the v16a file prediction by prediction.

📋 **PASTE #1: the whole output.** It must end in `ALL CHECKS PASSED` (atexit lines after it are normal).

## 3. Cheap checks, then the smoke

**Gadi:**
```bash
ls -d /g/data/wa66/yh2481/qcq_cnn_v14/results_v14_k{5_A,5_B,10_all}_n{50,500} \
      /g/data/wa66/yh2481/qcq_cnn_v13/results_v13_k{5_A,5_B,10_all}_w15
ls /g/data/wa66/yh2481/qcq_cnn_v14/pbs/v14g5A500M4s4[3-5].o* /g/data/wa66/yh2481/qcq_cnn_v14/pbs/v14s_5A500M4.o* 2>/dev/null | head -4
cd /g/data/wa66/yh2481/qcq_cnn_v16b/pbs && ./submit_v16b_smoke.sh && qstat -u $USER | tail -8
```
The submit script refuses if any of the 120 stored M4 / M2 comparator files is missing. If it does, fix the `*_ROOT` paths at the top of `pbs/config16b.sh` and paste what it printed. The second `ls` only shows whether the v14 logs are there for the split cross-check; it's fine if it prints nothing.

📋 **PASTE #2: the `ls` output and the submit lines** (`Stored comparators: all 120 present`, `Submitted 7 smoke jobs`).

## 4. Smoke gate

When `qstat -u $USER` is empty:

**Gadi:**
```bash
cd /g/data/wa66/yh2481/qcq_cnn_v16b/pbs && ./check_v16b.sh smoke
```
It must show the reproduction lines `MATCH` and end in `ALL CHECKS PASSED`. It prints epochs, job times and the learned g/s only, not accuracies.

📋 **PASTE #3: the whole gate output.** Don't submit the grid until I've replied.

## 5. Grid

**Gadi:**
```bash
cd /g/data/wa66/yh2481/qcq_cnn_v16b/pbs && ./submit_v16b_grid.sh && qstat -u $USER | wc -l && lquota
```
It is resumable: if qsub stops at a queue limit, run it again later.

📋 **PASTE #4: the submit output** (`Submitted 54 grid jobs`, or how many before a queue limit).

Progress, any time:
```bash
qstat -u $USER | grep -c " Q \| R "                                     # jobs left
ls /g/data/wa66/yh2481/qcq_cnn_v16b/pbs/v16bg*.o* 2>/dev/null | wc -l   # logs written (54 at the end)
```

## 6. Grid gate: before any accuracy is read

**Gadi:**
```bash
cd /g/data/wa66/yh2481/qcq_cnn_v16b/pbs && ./check_v16b.sh grid
```
📋 **PASTE #5: the whole gate output.** Don't run collect until I've replied.

## 7. Results

**Gadi:**
```bash
cd /g/data/wa66/yh2481/qcq_cnn_v16b
python3 scripts/collect_v16b.py | tee collect_v16b.log
python3 scripts/plot_v16b.py
rm -rf ckpt_* ; find . -name __pycache__ -prune -exec rm -rf {} +
tar czf v16b_results.tgz v16b_*.csv collect_v16b.log figures_v16b results_v16b_*/ pbs/v16b*.o* patch_v16b.log verify_v16b.log PREREGISTRATION_v16b.md scripts/plot_v16b.py
ls -la v16b_results.tgz
```
**Local (PowerShell):**
```powershell
scp yh2481@gadi-dm.nci.org.au:/g/data/wa66/yh2481/qcq_cnn_v16b/v16b_results.tgz .
tar -xzf v16b_results.tgz
```
📋 **PASTE #6: `collect_v16b.log`**, and upload `v16b_results.tgz`.

## If something goes wrong

- A job leaves `qstat` within seconds: `cat pbs/<jobname>.o*` and paste it.
- Setup refuses because `qcq_cnn_v16b` exists: `mv /g/data/wa66/yh2481/qcq_cnn_v16b /g/data/wa66/yh2481/qcq_cnn_v16b_old` (only before the smoke is submitted), then rerun setup.
- Verify fails: paste the whole `verify_v16b.log`. If it names dry-run logs, paste `tail -40` of the one that failed.

## Files

| File | Role |
|---|---|
| `bfr_head_v16b.py` | BFR head: batched 2-qubit simulator (15 Paulis), `BFRQuantum`, `BFRHead`, M4 → BFR surgery, parameter rule. |
| `patch_v16b.py` | Inserts the `V16B_BFR_HEAD` block into the v16a `train_gadi.py` (flag `--v16b_head bfr`). Edits no existing line. Backup `train_gadi.py.v16abak`. |
| `setup_v16b.sh` | Build `qcq_cnn_v16b` from `qcq_cnn_v16a`, patch, verify. |
| `scripts/verify_v16b.py` | Pre-flight; must end in `ALL CHECKS PASSED`. |
| `pbs/config16b.sh` | Pre-registered design and comparator paths. |
| `pbs/run16b.pbs`, `submit_v16b_smoke.sh`, `submit_v16b_grid.sh` | One job; the smoke (7); the grid (54, refuses without the smoke gate). |
| `pbs/check_v16b.sh` → `scripts/gate_v16b.py` | Integrity gate (stdlib only, prints no accuracy). |
| `scripts/collect_v16b.py`, `scripts/plot_v16b.py` | Pre-registered analysis; standard figure set (`figures_v16b/`, 200 dpi). |
| `PREREGISTRATION_v16b.md` | Protocol, predictions, amendments A1, A2 and defect D1. |
