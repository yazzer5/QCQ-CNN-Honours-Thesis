# v16 synthesis: one consistent story for the write-up (v12 → v16c)

**Written:** 2026-10-22, after v16a, v16b, v16b-ext and v16c were complete.
**Purpose:** the single source of truth for the write-up. Where this file and an older
results file disagree on *wording or mechanism*, this file wins. Numbers stay in the results files.

> **Later updates:** the open item in §6 (a classical control for v15.5) was run as v15.8
> (`results/V15_8_RESULTS.md`): a fixed cos(πx) front-end with zero parameters is at least as
> robust as the trained circuit. v15.6 and v15.7 (noise geometry and its evolution in
> training) are in `results/V15_6_RESULTS.md` and `results/V15_7_RESULTS.md`.

**Gate record:** v16c gate G-b **passed 2026-10-22** on Gadi (torch 2.5.1+cu121): the 60
Circuit-4 init weights reproduce the local digest `5dba16aa2f1d12ab` bitwise. Part A of
V16C_RESULTS (seed-46 dead channel, conditioning) is final, no longer provisional.

---

## 1. Thesis claim in one paragraph

In a QCQ-CNN on MNIST, the quantum parts act only as **classical feature maps** with particular
inductive biases. The trainable Circuit-4 front-end never beats having no front-end on clean
data or under IBM-calibrated circuit noise, but once trained it is **more robust to corrupted
input images** (v15.5), plausibly because its cos(πx) encoding is flat at black/white pixels.
The ZZ quantum head costs accuracy everywhere (2.3 pp clean, 13–17 pp at low data, amplifies
noise). v16a shows that cost belongs to a classical 12-feature trig map (the circuit adds at
most 0.9 pp). v16c shows its mechanism is a **fixed, high gain** periodic map (~9 rad per unit
input), not an unbounded frequency. A redesigned head with learnable lower gain, gated and
residual (BFR, v16b) recovers 75–95% of the cost, but no quantum head ever beats the classical
head (M4). Seed 46's failures are an exact closed-form dead-channel initialisation (v16c F1).

## 2. Front-end: six conditions

| Condition | Exp | F = M3 − M4 | Reading |
|---|---|---|---|
| Class count K = 3–10 | v12 | TOST ≡ 0 (±1 pp) at K = 3–5; K = 10 +0.33, not shown | neutral |
| Readout width | v13 | unchanged | neutral |
| Train vs freeze | v13.5 | M5 − M4 < 0; M3 − M5 > 0 (mostly seed-46 rescue) | training repairs, adds nothing |
| Low data N = 50–500 | v14 | −1.5 / −1.7 at N = 50; advantage > ~1 pp excluded | neutral to slightly negative |
| IBM circuit noise α ≤ 4 | v15 | not detected; noise cost ≡ 0 when trained under it | NISQ-tolerant, no advantage |
| **Input pixel noise σ** | **v15.5** | **+9.1 pp deployed at σ = 0.3 (K = 5, Holm p = .001); +4.1 matched (K = 10, p = .008)** | **only positive; needs training (M5 not robust)** |
| Brain MRI (binary) | v11 | −3.68 pp (Holm p = .002) | harmful (contrast squashing at 0, π) |

**Mechanism notes.**
- *Why neutral on clean data:* each output is a low-degree trig function of its own pixels
  (v16c F1 closed form); the first conv absorbs it.
- *Why robust to pixel noise (v15.5 L4, exploratory):* ⟨Z⟩ = cos(πx) has zero slope at
  x = 0, 1, where 92% of MNIST pixels sit, so small perturbations are suppressed. Classical in
  nature. **Not a quantum advantage**; confirmed by the matched classical front-end in v15.8.
- *Seed 46 (v16c F1, exact):* channel-b amplitude A_b = √(cos²w4 + sin²w4·cos²(w2/2)) vanishes at
  (w2, w4) = (π, 3π/2); seed 46 lands 0.11 / 0.03 rad away (A_b = 0.063 vs 0.76–0.98). Explains
  worst-M5-everywhere, the training rescue (counterfactual, not observed), and its unique
  fragility to device and pixel noise.

## 3. QNN head: symptoms → cause → fix

| Step | Exp | Result |
|---|---|---|
| Symptom: clean cost | v12–v13 | grows with K at w3 (1.4 → 8.2 pp); ~2.3 pp, flat in K at w15 |
| Symptom: data scarcity | v14 | 13.5 / 17.1 pp at N = 50; it is a generalisation gap |
| Symptom: noise | v15, v15.5 | amplifies circuit noise (7.5–13.8 pp at α = 4) and pixel noise (+18 pp at σ = 0.3); cost vanishes with noise augmentation |
| Cause is classical | v16a | 12-feature closed-form replica (M2c) reproduces cost, collapse and per-image errors; circuit worth ≤ 0.75 / 0.88 pp (95% one-sided) |
| Mechanism | v16c F2 (post hoc) | z stays in |z| ≤ 2; all classes inside one ZZ period; local gain fixed ≈ 2π√2 ≈ 9 rad/unit at every N. **Not** unbounded frequency (H1–H3 failed) |
| Redesign | v16b | BFR (u = s·tanh z, no product term, gated residual): recovers 75–95%; never beats M4; K10 N50 −4.7 pp (Holm p = .024) |
| What drove the recovery | v16b-ext | restoring the product term on bounded inputs costs −1.45 pp (≥ 5 pp excluded) → not the product term. Learned s fell in every run (~4–5× lower gain). About half of the −4.7 pp is the fc_mid trim (M4-trim −2.53), half the branch (−2.12); neither alone detected |

## 4. Wording corrections (apply in the thesis; older results files are not edited)

| Where | Old | Use instead |
|---|---|---|
| V16B §7.1 | "unbounded frequency (and the product term)" | "a fixed, high-gain periodic map (v16c F2)" |
| V16B §11.4.3 | "unboundedness, rather than the product structure, drives M2's failure" | "gain, rather than the product structure" (M2 never uses the unbounded range) |
| V16B §7 / supervisor summary | "the remaining cost comes from an active quantum branch" | "about half is the trimmed classical parameterisation (v16b-ext), the split unresolved at n = 10" |
| Front-end headline (pre-v15.5) | "no front-end advantage in any condition" | "no advantage on clean data or under hardware noise; more robust to corrupted inputs once trained (classical mechanism)" |
| Any results file | "quantum advantage", "equivalent" without TOST | "more robust than no front-end"; "not detected, bounded at X pp" |
| v16a P1 | "equivalent" | "not detected; circuit benefit ≤ 0.75 / 0.88 pp" |

## 5. Failed predictions to report as failed (integrity section)

v13.5 P1, P4 · v14 P1 (equivalence part) · v15.5 P1, P2 (sign wrong) · v16a P1, P2 ·
v16b P2, P5 · v16b-ext E1 (direction only), E2 · v16c C1, C2, H1, H2, H3.
Method lesson (v16c): a median over blank-dominated patches measures one point; stratify.

## 6. Open item (as of 2026-10-22; since run as v15.8)

v15.5's positive result is the thesis's only F > 0 and has no classical control. Minimal design:
- **M6:** M3's architecture with the Circuit-4 front-end replaced by the fixed per-pixel encoding
  cos(πx_i) for the 4 pixels of each 2×2 patch, 0 front-end parameters (the circuit
  with all trainable gates removed).
- Deployed arm: train clean, evaluate at all σ (one job per seed). Matched arm at σ = 0.3.
- K = 5 A, K = 5 B, K = 10 × seeds × {clean-trained, σ = 0.3-trained}.
- Primary: M3 − M6 and M6 − M4 at σ_test = 0.3, deployed and matched; Holm within K.
- Readings: M6 ≈ M3 → the robustness is the fixed cos encoding (classical). M6 ≈ M4 < M3 → it
  needs the trained mixing; still classical in closed form (v16c F1), but specific to the
  trained map.

Outcome: see `results/V15_8_RESULTS.md`.

## 7. Results-chapter map

1. Methods and integrity: batched simulator, defects caught, pre-registration, TOST/Holm/MDE,
   bit-exact reproductions, failed predictions list (§5).
2. Front-end: the six-condition table (§2), MRI, the seed-46 dead channel (lead figure
   `figA_dead_channel_posthoc.png`), v15.5 robustness with its classical control (v15.8).
3. QNN head: v12–v15.5 symptoms → v16a (`backend_cost.png`, `paired_D.png`).
4. Diagnose → redesign → test: v16c F2 (`fig3_f_histograms.png`), v16b (`paired_D2.png` next to
   `paired_D4.png`), v16b-ext decomposition.
5. Discussion: everything here is classically simulable; the quantum parts act through
   inductive bias only; positioning against Long et al. and Bowles et al. 2024.
