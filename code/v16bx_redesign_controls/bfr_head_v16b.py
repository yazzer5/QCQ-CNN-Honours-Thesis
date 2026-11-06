"""
bfr_head_v16b.py -- v16b Design 1: BFR head (bounded-frequency, residual-gated quantum head).

    logits = fc3(fc_mid(z)) + g * W_q <P>(u),     u_i = s * tanh(z_i),     z = fc2 output (2-D)

* Classical path = M4's own head (fc_mid -> fc3, linear, i.e. W_c z + b of rank <= 2), with fc_mid's
  width trimmed from 15 to w(K) so the total parameter count matches M4 (pre-registration A2):
  K=5: w=5 (1720 vs M4 1719), K=10: w=3 (1799 vs 1799). Any w >= 2 gives the same function class.
* Quantum path: 2 qubits from |00>, L = 2 re-uploading layers, each
      Ry(u0) (x) Ry(u1)  ->  Ry(theta_l0) (x) Ry(theta_l1)  ->  CNOT(control q0, target q1)
  then exact expectations of all 15 non-identity two-qubit Paulis (order = closed_form_zz_v16.PAULI_NAMES,
  first letter acts on q0). No ZZ product term. s learnable (init pi/2), theta init 0,
  g learnable (init 0), W_q (K x 15, no bias) init N(0, 0.01^2).
* Structural facts (verified by verify_v16b.py, stated in the pre-registration):
  - all gates are real, so the state is real and the 6 Paulis with exactly one Y are identically 0;
  - Ry(u) Ry(theta) = Ry(u + theta): the trainable angles act as offsets on the encoded angles;
  - at g = 0 the logits equal the classical path exactly; only g gets gradient on the first step.

install_bfr(model, seed) does the surgery on a freshly built M4 (QCQCNN with ClassicalHead), in place,
inside a forked RNG: every global RNG stream is left exactly where building M4 left it, so a BFR run at
seed s sees M4's data split AND M4's batch order. Backbone and fc2 are M4's, untouched.
"""
import math
import random

import numpy as np
import torch
import torch.nn as nn

from closed_form_zz_v16 import PAULI_NAMES

N_LAYERS = 2
WQ_STD = 1e-2
S_INIT = math.pi / 2
MID_FULL = 15   # M4's fc_mid width at w15

_P1 = {
    "I": torch.tensor([[1, 0], [0, 1]], dtype=torch.complex128),
    "X": torch.tensor([[0, 1], [1, 0]], dtype=torch.complex128),
    "Y": torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex128),
    "Z": torch.tensor([[1, 0], [0, -1]], dtype=torch.complex128),
}


class BFRError(RuntimeError):
    pass


def pauli_stack_real():
    """(15, 4, 4) float64: real parts of the 15 Paulis, basis index = 2*b0 + b1 (q0 = left factor).
    For a REAL state psi, <psi|P|psi> = psi^T Re(P) psi exactly (Im P is antisymmetric or zero)."""
    mats = []
    for lab in PAULI_NAMES:
        P = torch.kron(_P1[lab[0]], _P1[lab[1]])
        if not torch.allclose(P.imag, -P.imag.T):
            raise BFRError("Pauli imaginary part not antisymmetric")
        mats.append(P.real)
    return torch.stack(mats)


CNOT01 = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=torch.float64)


def _ry_pair(a0, a1):
    """Batched Ry(a0) (x) Ry(a1), (B, 4, 4)."""
    c0, s0 = torch.cos(a0 / 2), torch.sin(a0 / 2)
    c1, s1 = torch.cos(a1 / 2), torch.sin(a1 / 2)
    r0 = torch.stack([torch.stack([c0, -s0], -1), torch.stack([s0, c0], -1)], -2)
    r1 = torch.stack([torch.stack([c1, -s1], -1), torch.stack([s1, c1], -1)], -2)
    return torch.einsum("bij,bkl->bikjl", r0, r1).reshape(-1, 4, 4)


def bfr_expectations(u, theta, paulis=None, cnot=None):
    """u (B, 2) encoded angles, theta (L, 2). Returns (B, 15) exact Pauli expectations. Batched, no loop
    over samples. paulis/cnot: cached buffers in the model's dtype/device (built here if None)."""
    B = u.shape[0]
    if paulis is None:
        paulis = pauli_stack_real().to(dtype=u.dtype, device=u.device)
    if cnot is None:
        cnot = CNOT01.to(dtype=u.dtype, device=u.device)
    psi = u.new_zeros(B, 4)
    psi[:, 0] = 1.0
    for l in range(theta.shape[0]):
        psi = torch.einsum("bij,bj->bi", _ry_pair(u[:, 0], u[:, 1]), psi)
        t = theta[l].unsqueeze(0).expand(B, 2)
        psi = torch.einsum("bij,bj->bi", _ry_pair(t[:, 0], t[:, 1]), psi)
        psi = psi @ cnot.T
    return torch.einsum("bi,pij,bj->bp", psi, paulis, psi)


class BFRQuantum(nn.Module):
    """The gated quantum branch: g * W_q <P>(s * tanh(z)). Parameters: s, theta (L x 2), g, W_q (K x 15)."""

    def __init__(self, K, n_layers=N_LAYERS, generator=None):
        super().__init__()
        self.s = nn.Parameter(torch.tensor(S_INIT))
        self.theta = nn.Parameter(torch.zeros(n_layers, 2))
        self.g = nn.Parameter(torch.tensor(0.0))
        self.W_q = nn.Parameter(torch.randn(K, 15, generator=generator) * WQ_STD)
        self.register_buffer("paulis", pauli_stack_real().float(), persistent=False)
        self.register_buffer("cnot", CNOT01.float(), persistent=False)

    def encode(self, z):
        return self.s * torch.tanh(z)

    def features(self, z):
        return bfr_expectations(self.encode(z), self.theta, self.paulis.to(z.dtype), self.cnot.to(z.dtype))

    def forward(self, z):
        return self.g * (self.features(z) @ self.W_q.T)


class BFRHead(nn.Module):
    """Replaces M4's ClassicalHead. Keeps the attribute names fc2 / fc_mid / fc3 so z (fc2's output)
    is found the same way as for every other model (v16_tools.z_layer)."""

    def __init__(self, fc2, fc_mid, fc3, quantum, head_variant="linear"):
        super().__init__()
        if head_variant != "linear":
            raise BFRError("BFR is defined with the linear classical head only")
        self.fc2, self.fc_mid, self.fc3, self.bfr = fc2, fc_mid, fc3, quantum

    def classical(self, z):
        return self.fc3(self.fc_mid(z))

    def forward(self, x):
        z = self.fc2(x)
        return self.classical(z) + self.bfr(z)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def head_params(K, w):
    """Parameters after fc2 in BFR: fc_mid(2->w) + fc3(w->K) + quantum (s + 2L theta + g + 15K)."""
    return (2 * w + w) + (w * K + K) + (1 + 2 * N_LAYERS + 1 + 15 * K)


def m4_params(K):
    return 1464 + 130 + (2 * MID_FULL + MID_FULL) + (MID_FULL * K + K)


def bfr_params(K, w):
    return 1464 + 130 + head_params(K, w)


def choose_width(K):
    """fc_mid width w in [2, 15] minimising |BFR - M4|; ties go to the larger w."""
    best = min(range(2, MID_FULL + 1), key=lambda w: (abs(bfr_params(K, w) - m4_params(K)), -w))
    return best


def _rng_save():
    st = dict(py=random.getstate(), np=np.random.get_state(), t=torch.get_rng_state())
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _rng_restore(st):
    random.setstate(st["py"]); np.random.set_state(st["np"]); torch.set_rng_state(st["t"])
    if "cuda" in st:
        torch.cuda.set_rng_state_all(st["cuda"])


def install_bfr(model, seed, head_variant="linear"):
    """M4 -> BFR in place. Returns a dict describing the surgery. Raises BFRError if not M4-shaped."""
    st = _rng_save()
    try:
        be = getattr(model, "back_end", None)
        if be is None or type(be).__name__ != "ClassicalHead":
            raise BFRError(f"back_end is {type(be).__name__}, expected M4's ClassicalHead")
        if any(p.numel() for p in model.front_end.parameters()):
            raise BFRError("front_end has parameters: BFR is built on M4 (no quantum front-end)")
        fc2, fm, f3 = be.fc2, be.fc_mid, be.fc3
        if fm is None or f3 is None:
            raise BFRError("binary head (no fc_mid): BFR is defined for the multiclass w15 head")
        if not (fc2.in_features == 64 and fc2.out_features == 2 and fm.in_features == 2
                and fm.out_features == MID_FULL and f3.in_features == MID_FULL):
            raise BFRError(f"head shapes fc2 {tuple(fc2.weight.shape)} fc_mid {tuple(fm.weight.shape)} "
                           f"fc3 {tuple(f3.weight.shape)} are not M4 at w15")
        K = f3.out_features
        before = sum(p.numel() for p in model.parameters())
        w = choose_width(K)
        dev, dt = f3.weight.device, f3.weight.dtype
        new_mid = nn.Linear(2, w).to(device=dev, dtype=dt)
        new_f3 = nn.Linear(w, K).to(device=dev, dtype=dt)
        scale = math.sqrt(MID_FULL / w)   # fc3 default bound 1/sqrt(fan_in): 1/sqrt(15) -> 1/sqrt(w)
        with torch.no_grad():
            new_mid.weight.copy_(fm.weight[:w]); new_mid.bias.copy_(fm.bias[:w])
            new_f3.weight.copy_(f3.weight[:, :w] * scale); new_f3.bias.copy_(f3.bias * scale)
        gen = torch.Generator().manual_seed(int(seed))
        q = BFRQuantum(K, generator=gen).to(device=dev, dtype=dt)
        model.back_end = BFRHead(fc2, new_mid, new_f3, q, head_variant)
        after = sum(p.numel() for p in model.parameters())
        return dict(K=K, width=w, params_m4=before, params_bfr=after, expected_bfr=bfr_params(K, w),
                    delta=after - before, rel=(after - before) / before,
                    ok=(before == m4_params(K) and after == bfr_params(K, w)
                        and abs(after - before) <= 0.01 * before))
    finally:
        _rng_restore(st)


def bfr_module(model):
    be = getattr(model, "back_end", None)
    return be.bfr if isinstance(be, BFRHead) else None


def bfr_state(model):
    q = bfr_module(model)
    if q is None:
        return None
    return dict(g=float(q.g.detach()), s=float(q.s.detach()), theta=q.theta.detach().cpu().flatten().tolist(),
                wq_norm=float(q.W_q.detach().norm()))
