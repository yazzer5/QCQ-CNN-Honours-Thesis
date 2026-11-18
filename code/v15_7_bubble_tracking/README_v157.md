# v15.7: does M3 form the tight noise bubble earlier in training?
Pre-registration: PREREGISTRATION_v15_7.md (project). 60 jobs: M3/M4/M5 x trained at sigma 0 / 0.3 x K5 A x seeds 42-51.
Training is unchanged: every job must reproduce v15.5 bit-exactly, and its final geometry must equal v15.6's.

## 1. Upload + setup   (PowerShell)
    scp v157_pkg.zip yh2481@gadi-dm.nci.org.au:/g/data/wa66/yh2481/
## Gadi
    source /g/data/wa66/yh2481/miniconda3/etc/profile.d/conda.sh && conda activate qcqcnn && export PYTHONPATH=""
    cd /g/data/wa66/yh2481 && unzip -o v157_pkg.zip && cd v157_pkg && bash setup_v157.sh 2>&1 | tee ~/setup_v157.out
📋 PASTE 1: ~/setup_v157.out (must show ALL CHECKS PASSED)

## 2. Smoke (6 jobs)
    cd /g/data/wa66/yh2481/qcq_cnn_v157 && ./pbs/submit_v157_smoke.sh && qstat -u yh2481
## 3. Smoke gate (when qstat is empty)
    python3 gate_v157.py smoke $PWD ../qcq_cnn_v156 ../qcq_cnn_v155
📋 PASTE 2 (must end ALL CHECKS PASSED)

## 4. Grid (54 jobs)
    ./pbs/submit_v157_grid.sh
## 5. Grid gate + analysis (when qstat is empty)
    python3 gate_v157.py grid $PWD ../qcq_cnn_v156 ../qcq_cnn_v155
    python3 analyse_v157.py $PWD
📋 PASTE 3: both outputs. Then zip the figures and attach them:
    zip -qr ~/v157_out.zip v157_out && echo done
  (PowerShell)  scp yh2481@gadi-dm.nci.org.au:~/v157_out.zip .
