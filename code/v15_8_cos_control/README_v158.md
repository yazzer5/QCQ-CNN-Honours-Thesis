# v15.8: M6, the fixed cos(πx) front-end (classical control for v15.5)

Pre-registration: `PREREGISTRATION_v15_8.md` (in the Project). Pattern: setup + verify → smoke (9 jobs) → gate → grid (24 jobs) → gate → analysis. Never read an accuracy before the gate prints `ALL CHECKS PASSED`.

| | |
|---|---|
| New model | M6 = M3 with Circuit 4 replaced by cos(π·pixel) on each pixel of the 2×2 patch (= Circuit 4 with all weights 0). 0 front-end params, total = M4's |
| Reused | M3, M4, M5 from `qcq_cnn_v155` (same seeds, same noisy test sets; the gate checks pairing) |
| Cells | K=5 A, K=5 B, K=10 × σ_train ∈ {0 (deployed), 0.3 (matched)} |
| Seeds | 42–46 |
| Smoke (9) | M6 seed 42 × 6 + M3/M4/M5 seed 42 at K=5 A σ=0.3, which must equal v15.5 bit-exactly |
| Grid (24) | M6 seeds 43–46 × 6 |

## 0. Local (PowerShell), in the folder with `v158_pkg.zip`
```powershell
scp v158_pkg.zip yh2481@gadi-dm.nci.org.au:/g/data/wa66/yh2481/
```

## 1. Gadi: build, patch, verify
```bash
source /g/data/wa66/yh2481/miniconda3/etc/profile.d/conda.sh && conda activate qcqcnn && export PYTHONPATH=""
cd /g/data/wa66/yh2481 && unzip -o v158_pkg.zip && cd v158_pkg && bash setup_v158.sh 2>&1 | tee ~/setup_v158.out
```
📋 **PASTE 1:** the whole output. Must end in `ALL CHECKS PASSED` and show `stored v15.5 M4 sigma=0 runs for seeds 42-46: 15`.

## 2. Smoke (9 jobs, ~1 h)
```bash
cd /g/data/wa66/yh2481/qcq_cnn_v158/pbs && ./submit_v158_smoke.sh
qstat -u $USER | grep -c v158s          # when this prints 0:
./check_v158.sh smoke
```
📋 **PASTE 2:** the gate output. Must show three reproduction blocks of `MATCH` and end in `ALL CHECKS PASSED`.

## 3. Grid (24 jobs, ~1 h if they run in parallel)
```bash
cd /g/data/wa66/yh2481/qcq_cnn_v158/pbs && ./submit_v158_grid.sh
qstat -u $USER | grep -c v158g          # when this prints 0:
./check_v158.sh grid
cd .. && python3 scripts/analyse_v158.py | tee analyse_v158.log
zip -qr ~/v158_out.zip v158_out analyse_v158.log && echo done
```
```powershell
scp yh2481@gadi-dm.nci.org.au:~/v158_out.zip .
```
📋 **PASTE 3:** the grid gate output and `analyse_v158.log`, and attach `v158_out.zip`.

## If something goes wrong
- A job leaves `qstat` in seconds: `cat pbs/<jobname>.o*` and paste it.
- Any `FATAL`, a reproduction `DIFFERENT`, `not paired`, or a smoke cap hit: stop and paste. Stop rules.
- `REFUSED: ... inodes`: free inodes or tell me; don't override.
- `qsub` refuses part-way: wait, then re-run the same submit script (it skips done/queued jobs).

## Files
| File | Role |
|---|---|
| `patch_v158_m6.py` | Adds M6: one block above the v15.5 block (wraps `build_model`/`expected_params` by name) + `"M6"` in `--model` choices. Backup `.v155bak` |
| `scripts/verify_v158.py` | Text diff = only the block; M1–M5 bit-identical to v15.5; M6 = cos(πx) = Circuit 4 at w=0; M6 init = M4's; dry runs incl. v15.8 vs v15.5 trainer on M3 |
| `pbs/config158.sh` | Design and paths (`V155_ROOT`) |
| `pbs/run158.pbs`, `submit_v158_*.sh` | One job; smoke; grid (refuses without `.smoke_ok`) |
| `pbs/check_v158.sh` → `scripts/gate_v158.py` | Integrity gate, stdlib only |
| `scripts/analyse_v158.py` | Primary G = M3−M6, H = M6−M4 (Holm, MDE, TOST), secondaries, standard plots → `v158_out/` |
