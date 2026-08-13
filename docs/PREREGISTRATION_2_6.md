# Pre-registration — Fashion-MNIST 2,6, arm (a)

**Registered:** 2026-08-10, before any 2,6 grid job was submitted.
**Experiment:** FMNIST pair 2,6 (Pullover vs Shirt), arm (a) iso-parametric.
4 models × 5 seeds {42–46} = 20 runs. Code v8.
**Position in programme:** experiment 2 of 3 in the difficulty gradient.
Rank 1 (0,6, M4 pilot 84.20%) is complete; rank 3 (2,4, pilot 88.40%) follows.

This file exists so the 2,6 and 2,4 grids are **confirmatory rather than
exploratory**. Examiners distinguish the two, and a prediction written after
seeing the data is not a prediction. Do not edit the predictions below once a
job has been submitted — add an outcome section underneath instead.

---

## Predictions

1. **M4, the fully classical control, remains the best model** on 2,6.

2. **The QNN-head vs classical-head gap narrows as the task eases.** The
   grouped contrast (M1,M2) vs (M3,M4) was −1.18 pp at 0,6 (p = 0.011,
   M4 84.20%). Expect a smaller magnitude at 2,6 (~87%) and smaller again at
   2,4 (~88%), trending toward the 0.08 pp total spread seen at MNIST
   saturation.

3. **M1/M2 continue to converge roughly 3–4× earlier than M4.** At 0,6:
   best epoch 11.8 (M1) and 17.4 (M2) against 51.0 (M4).

4. **gap@best stays flat across M1/M2/M3, with only M4 lower.** At 0,6, only
   the M2−M4 contrast reached significance (+1.86 pp, p = 0.023).

## What each outcome would mean

- If (2) holds across all three difficulty levels, that is a **dose-response
  result on task difficulty** — substantially stronger evidence than any
  single significant comparison, because it predicts the direction *and* the
  ordering of three independent grids.
- If (2) fails — the gap does not shrink, or moves the wrong way — the
  ordering is **not difficulty-driven**. That is informative, not a failure:
  it redirects effort toward Aim 3 (noise as a regulariser) and away from
  further class-pair sweeps.
- The three pre-registered attribution contrasts (M2−M1, M2−M3, M3−M4) are
  expected to be **null again**, since each adjacent step at 0,6 was 0.4–0.7 pp
  against an MDE of ~1.8 pp. A null here must be reported with its MDE.

## Power

At ~87% accuracy and n = 1000 test images, per-run SE ≈ 1.06 pp. With 5 seeds
the MDE is roughly **1.6–1.8 pp**. The actual MDE will be computed from the
observed between-seed SD by `collect_results.py` and must be quoted with every
null.

**Confound to disclose:** fixed-seed runs on this stack vary by ~0.8 pp from
GPU non-determinism (`FMNIST_LOG.md` §5.5), comparable to the between-seed SD
of 0.92 pp. Roughly half the observed variance may be machine noise, which
weakens the pairing in the paired t-tests. `torch.use_deterministic_algorithms`
is deliberately **not** enabled for this grid, so that 0,6 / 2,6 / 2,4 share
one protocol and the gradient comparison stays internally consistent.

**Ranking caveat:** the 2,6 vs 2,4 difficulty ordering comes from single-seed
M4 pilots 1.3 pp apart, inside the ~1.16 pp per-run SE. It may not survive five
seeds. Report the measured 5-seed ordering, not the pilot ordering.

---

## Outcome

*(leave blank until all 20 runs are in and the integrity checks pass)*
