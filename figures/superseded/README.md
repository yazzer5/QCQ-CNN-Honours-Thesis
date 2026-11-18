# Superseded figures

Kept for the record. Do not use.

- `fig1_accuracy_and_gap.png` (panel b) and `fig3_gap_vs_acc_and_epochs.png` compute the
  generalisation gap as final-epoch train accuracy minus test accuracy. The trainer
  restores the best-validation checkpoint before testing, so those two numbers come from
  different models. The claim built on them was retracted; the corrected gap@best metric
  is used from v8 on. See `logs/FMNIST_LOG.md` §6.1.
