#!/usr/bin/env python
"""
verify_dropout_qnn_shift.py — measures the train/eval discrepancy that
Dropout2d induces in the QNN back-end, using the ACTUAL back-end circuit
(ZZFeatureMap(2, reps=1) + RealAmplitudes(2, reps=2), Z(x)Z observable)
simulated exactly in torch.

*** STATUS 2026-07-28: THIS EFFECT IS REAL BUT TOO SMALL TO MATTER. ***
This script was written to test the hypothesis that dropout's train/eval gap
caused M1/M2's observed inversion. **That hypothesis was REFUTED on hardware.**
The A/B test (dropout 0.5 vs 0.0, M2/mnist/seed42) had both arms train
successfully (test 0.97 and 0.95), and the in-training check_train_eval_shift()
diagnostic measured near-identical train/eval margins with 0/32 predictions
flipping at every checkpoint. The shift this script measures (~0.02) is real
but does not reach the magnitude required to flip a trained decision boundary.
Retained as a measurement of a genuine second-order effect, and as a record of
a disproved hypothesis. Do NOT cite it as the cause of the inversion.

Claim under test
----------------
Dropout is expectation-preserving: E[dropout(x)] = x. Therefore:
  * M3/M4's LINEAR head:    E[f(dropout(x))] == f(E[x])      -> no train/eval gap
  * M1/M2's NONLINEAR QNN:  E[QNN(dropout(x))] != QNN(E[x])  -> mean shift

and because fc3 = Linear(1,2) reduces the QNN's single scalar output to a
2-class argmax, a shift in that scalar's mean flips EVERY prediction at once,
rather than degrading accuracy gradually. That is why the observed failure was
a clean inversion (train 0.95 / val 0.032, and 1-0.032 = 0.968) rather than a
slow decay.

Run:  python scripts/verify_dropout_qnn_shift.py
"""
import numpy as np
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
np.random.seed(0)

# ─────────────────────────────────────────────────────────────────────
# Exact 2-qubit simulation of ZZFeatureMap(reps=1) + RealAmplitudes(reps=2)
# ─────────────────────────────────────────────────────────────────────
C = torch.complex128
I2 = torch.eye(2, dtype=C)
H1 = (torch.tensor([[1, 1], [1, -1]], dtype=C) / np.sqrt(2))
CX = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=C)


def kron(A, B):
    return torch.einsum("ij,kl->ikjl", A, B).reshape(4, 4)


def phase(theta):
    """P(theta) = diag(1, e^{i theta})."""
    one = torch.ones((), dtype=C)
    return torch.diag(torch.stack([one, torch.exp(1j * theta.to(C))]))


def ry(t):
    c, s = torch.cos(t / 2), torch.sin(t / 2)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])]).to(C)


def zz_feature_map(x0, x1):
    """Qiskit ZZFeatureMap(2, reps=1): H^2, P(2x_i), then CX-P(2(pi-x0)(pi-x1))-CX."""
    U = kron(H1, H1)
    U = kron(phase(2 * x0), phase(2 * x1)) @ U
    U = CX @ U
    U = kron(I2, phase(2 * (np.pi - x0) * (np.pi - x1))) @ U
    U = CX @ U
    return U


def real_amplitudes(w):
    """RealAmplitudes(2, reps=2): RY layer, CX, RY layer, CX, RY layer. 6 params."""
    U = kron(ry(w[0]), ry(w[1]))
    U = CX @ U
    U = kron(ry(w[2]), ry(w[3])) @ U
    U = CX @ U
    U = kron(ry(w[4]), ry(w[5])) @ U
    return U


ZZ_OBS = torch.tensor([1.0, -1.0, -1.0, 1.0])   # Z(x)Z eigenvalues over |00>,|01>,|10>,|11>


def qnn_forward(feats, w):
    """feats: (B,2) -> (B,) expectation of Z(x)Z. The real EstimatorQNN output."""
    outs = []
    A = real_amplitudes(w)
    for i in range(feats.shape[0]):
        U = A @ zz_feature_map(feats[i, 0], feats[i, 1])
        psi = U[:, 0]                      # apply to |00>
        p = (psi.conj() * psi).real
        outs.append((p * ZZ_OBS).sum())
    return torch.stack(outs)


# ─────────────────────────────────────────────────────────────────────
# Realistic setup: 64-dim backbone features -> Dropout2d(16ch) -> fc2 -> head
# ─────────────────────────────────────────────────────────────────────
B, RATE, TRIALS = 64, 0.5, 400
feat_map = torch.rand(B, 16, 2, 2) * 1.5          # stand-in backbone activations
fc2 = nn.Linear(64, 2).double()
w = torch.rand(6, dtype=torch.float64) * 2 * np.pi
drop = nn.Dropout2d(RATE)


def features(train_mode):
    drop.train(train_mode)
    return fc2(drop(feat_map).flatten(1))


clean = features(False)                                     # eval mode
print("=" * 72)
print(f"Dropout2d(p={RATE}) -> fc2 -> head.  B={B}, {TRIALS} stochastic passes.")
print("=" * 72)

# --- LINEAR head (M3/M4): expectation preserved exactly ---
lin_tr = torch.stack([features(True) for _ in range(TRIALS)]).mean(0)
lin_margin_tr = (lin_tr[:, 0] - lin_tr[:, 1]).mean()
lin_margin_ev = (clean[:, 0] - clean[:, 1]).mean()
print("\nLINEAR head (M3 / M4):")
print(f"  E[f(dropout(x))] margin = {lin_margin_tr:+.5f}")
print(f"  f(E[x])          margin = {lin_margin_ev:+.5f}")
print(f"  |difference|            = {abs(lin_margin_tr - lin_margin_ev):.2e}   "
      f"-> expectation preserved, no train/eval gap")

# --- NONLINEAR QNN head (M1/M2) ---
qnn_tr = torch.stack([qnn_forward(features(True), w) for _ in range(TRIALS)]).mean(0)
qnn_ev = qnn_forward(clean, w)
print("\nNONLINEAR QNN head (M1 / M2)  [ZZFeatureMap + RealAmplitudes, Z(x)Z]:")
print(f"  E[QNN(dropout(x))] mean = {qnn_tr.mean():+.5f}")
print(f"  QNN(E[x])          mean = {qnn_ev.mean():+.5f}")
shift = (qnn_tr.mean() - qnn_ev.mean()).abs()
print(f"  |mean shift|            = {shift:.5f}")

# --- what that shift does to fc3 = Linear(1,2)'s argmax ---
fc3 = nn.Linear(1, 2).double()
with torch.no_grad():
    m_tr = (fc3(qnn_tr.unsqueeze(1))[:, 0] - fc3(qnn_tr.unsqueeze(1))[:, 1])
    m_ev = (fc3(qnn_ev.unsqueeze(1))[:, 0] - fc3(qnn_ev.unsqueeze(1))[:, 1])
flip = (m_tr.sign() != m_ev.sign()).sum().item()
print(f"\n  fc3 margin sign: train-mode mean={m_tr.mean():+.5f}  "
      f"eval-mode mean={m_ev.mean():+.5f}")
print(f"  samples whose argmax flips between modes: {flip}/{B}")

print("\n" + "=" * 72)
print("CONCLUSION")
print("=" * 72)
print(f"The linear head's train/eval gap is ~{abs(lin_margin_tr-lin_margin_ev):.1e} (numerically zero).")
print(f"The QNN head's is {shift:.4f} -- a real shift in the single scalar that")
print("fc3 thresholds, so it moves EVERY sample's decision in the same")
print("direction at once. That is the mechanism behind the observed clean")
print("inversion (train 0.95 / val 0.032) rather than gradual degradation.")
print("\nNOTE: whether the sign actually flips depends on where the learned")
print("decision boundary sits relative to the shift, so a single random draw")
print("here may or may not show flips -- the magnitude of the shift is the")
print("robust finding, not the flip count in this particular seed.")
