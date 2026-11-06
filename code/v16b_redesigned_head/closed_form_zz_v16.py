"""
closed_form_zz_v16.py -- the QNN head's w15 features in closed form (v16a, reused by v16b).

THE CIRCUIT (qnn_torch.QNNBackendTorch, = Qiskit ZZFeatureMap(2, reps=1) + RealAmplitudes(2, reps=2)):
    H q0, H q1 ; P(2 z0) q0 ; P(2 z1) q1 ; CX(0->1) ; P(2 (pi-z0)(pi-z1)) q1 ; CX(0->1)
    then a REAL orthogonal U(theta) (RY layers + CX), then the 15 two-qubit Paulis.
Little-endian basis |b1 b0>, index = b0 + 2 b1.

After the feature map the state is (1/2) sum_b exp(i phi_b) |b>, with zeta = 2 (pi-z0)(pi-z1):
    phi_00 = 0
    phi_01 = 2 z0 + zeta        (index 1: b0=1, b1=0; CX-P-CX gives zeta when b0 XOR b1 = 1)
    phi_10 = 2 z1 + zeta        (index 2)
    phi_11 = 2 z0 + 2 z1        (index 3)
For any observable O, <psi| U^T O U |psi> = Tr(O)/4 + 1/2 sum_{j<k} [Re M_jk cos(d_jk) - Im M_jk sin(d_jk)],
M = U^T O U, d_jk = phi_k - phi_j. So every Pauli expectation, for ANY theta, is a fixed linear
combination of the 12 features cos(d_jk), sin(d_jk) (6 pairs) plus a constant (0 for Paulis).
The six phase differences, written to avoid cancellation:
    d01 = 2 z0 + zeta     d02 = 2 z1 + zeta     d03 = 2 z0 + 2 z1
    d12 = 2 z1 - 2 z0     d13 = 2 z1 - zeta     d23 = 2 z0 - zeta
verify_v16a.py checks this against the REAL module (span residual <= 1e-6) before any job runs.

Public API (keep stable; v16b imports it):
    PAIRS, FEATURE_NAMES, N_FEATURES (=12)
    zz_phase_diffs(z)        -> (..., 6) tensor  [d01, d02, d03, d12, d13, d23]
    zz_trig_features(z)      -> (..., 12) tensor [cos d.., sin d..]
    ZZTrig12()               nn.Module, no parameters: z (B, 2) -> (B, 12)
    reference_paulis_numpy(z, theta, fm_reps=1) -> (B, 15) float64 dict-free reference (numpy),
                             columns in PAULI_NAMES order; used only by the verify script.
"""
import math

import numpy as np
import torch
import torch.nn as nn

PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
N_FEATURES = 12
FEATURE_NAMES = [f"cos_d{j}{k}" for j, k in PAIRS] + [f"sin_d{j}{k}" for j, k in PAIRS]
# Name convention as in the v13 readout patch: first letter acts on q0, second on q1.
PAULI_NAMES = ["ZI", "IZ", "ZZ", "XI", "IX", "XX", "YI", "IY", "YY", "XY", "YX", "XZ", "ZX", "YZ", "ZY"]


def zz_phase_diffs(z):
    """z: (..., 2) tensor (fc2 output). Returns (..., 6): d01, d02, d03, d12, d13, d23."""
    z0, z1 = z[..., 0], z[..., 1]
    zeta = 2.0 * (math.pi - z0) * (math.pi - z1)
    return torch.stack([2.0 * z0 + zeta, 2.0 * z1 + zeta, 2.0 * z0 + 2.0 * z1,
                        2.0 * z1 - 2.0 * z0, 2.0 * z1 - zeta, 2.0 * z0 - zeta], dim=-1)


def zz_trig_features(z):
    d = zz_phase_diffs(z)
    return torch.cat([torch.cos(d), torch.sin(d)], dim=-1)


class ZZTrig12(nn.Module):
    """The 12 closed-form trig features of the ZZFeatureMap state. No parameters."""
    n_features = N_FEATURES

    def forward(self, z):
        return zz_trig_features(z)

    def extra_repr(self):
        return "ZZFeatureMap(reps=1) closed form: cos/sin of 6 phase differences -> 12 features"


# ----------------------------------------------------------------------------------------
# Independent numpy reference of the full circuit (verify only). Qiskit conventions.
# ----------------------------------------------------------------------------------------
_P = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]], complex),
      "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1.0, -1.0]).astype(complex)}


def _op(name):                      # q0 = low bit -> kron(P_q1, P_q0)
    return np.kron(_P[name[1]], _P[name[0]])


_H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
_HH = np.kron(_H, _H)
_CX01 = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], complex)   # ctrl q0, tgt q1


def _ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]])


def reference_states_numpy(z, theta, fm_reps=1):
    """z: (B, 2) float64, theta: (6,). Returns (B, 4) complex statevectors."""
    z = np.asarray(z, float)
    B = z.shape[0]
    psi = np.zeros((B, 4), complex)
    psi[:, 0] = 1.0
    b0 = np.array([0, 1, 0, 1]); b1 = np.array([0, 0, 1, 1])
    for _ in range(fm_reps):
        psi = psi @ _HH.T
        ph = 2 * z[:, :1] * b0 + 2 * z[:, 1:] * b1                       # P(2 z0) q0, P(2 z1) q1
        psi = psi * np.exp(1j * ph)
        psi = psi @ _CX01.T
        zeta = 2 * (np.pi - z[:, :1]) * (np.pi - z[:, 1:])
        psi = psi * np.exp(1j * zeta * b1)                                # P(zeta) on q1
        psi = psi @ _CX01.T
    th = np.asarray(theta, float)
    for layer in range(3):
        U = np.kron(_ry(th[2 * layer + 1]), _ry(th[2 * layer]))
        psi = psi @ U.T
        if layer < 2:
            psi = psi @ _CX01.T
    return psi


def reference_paulis_numpy(z, theta, fm_reps=1):
    psi = reference_states_numpy(z, theta, fm_reps)
    return np.stack([np.einsum("bi,ij,bj->b", psi.conj(), _op(n), psi).real for n in PAULI_NAMES], 1)


def span_fit(features, targets):
    """Least squares targets ~ [features, 1] @ C. Returns (C, max abs residual per target column)."""
    A = np.concatenate([features, np.ones((features.shape[0], 1))], 1)
    C, *_ = np.linalg.lstsq(A, targets, rcond=None)
    return C, np.abs(A @ C - targets).max(0)
