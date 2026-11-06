"""
bfr_ext_v16b.py -- v16b-ext controls (PREREGISTRATION_v16b.md amendment A3). Imports bfr_head_v16b, changes nothing in it.

M4-trim  M4 with fc_mid trimmed exactly as in BFR (15 -> w(K): 3 at K=10, 5 at K=5), no quantum branch.
         = BFR's classical path, tensor for tensor (same slicing, same fc3 rescale). K=10: 1643 params.
BFR-ZZ   BFR with the ZZ product term restored in the encoding. Per re-uploading layer:
             Ry(u0) (x) Ry(u1) -> ZZ phase CX.P(zeta)_q1.CX -> Ry(theta_l0) (x) Ry(theta_l1) -> CNOT(q0 -> q1)
         zeta = 2 (pi - u0)(pi - u1) (ZZFeatureMap convention, as closed_form_zz_v16), u = s tanh z (bounded).
         The phase gate is diag(1, e^{i zeta}, e^{i zeta}, 1) in the basis index 2*b0 + b1 (phase when b0 XOR b1).
         Everything else = BFR: s, theta, g, W_q (same private-generator init), fc_mid trim, readout of the 15 Paulis.
         0 extra parameters (1799 at K=10). The state is complex, so the 6 single-Y Paulis are no longer 0.
         Simulated in real arithmetic (psi = a + i b) so no complex autograd is needed:
             <P> = a^T Re(P) a + b^T Re(P) b + 2 b^T Im(P) a.
"""
import math
import random

import numpy as np
import torch
import torch.nn as nn

import bfr_head_v16b as BH
from closed_form_zz_v16 import PAULI_NAMES


class ExtError(RuntimeError):
    pass


def pauli_stack_imag():
    mats = []
    for lab in PAULI_NAMES:
        mats.append(torch.kron(BH._P1[lab[0]], BH._P1[lab[1]]).imag)
    return torch.stack(mats)


def zz_zeta(u):
    return 2.0 * (math.pi - u[:, 0]) * (math.pi - u[:, 1])


def bfrzz_expectations(u, theta, zeta_scale=1.0, pr=None, pi_=None, cnot=None):
    """u (B, 2), theta (L, 2). zeta_scale = 0 switches the product term off (then == BFR exactly)."""
    B = u.shape[0]
    if pr is None:
        pr = BH.pauli_stack_real().to(dtype=u.dtype, device=u.device)
    if pi_ is None:
        pi_ = pauli_stack_imag().to(dtype=u.dtype, device=u.device)
    if cnot is None:
        cnot = BH.CNOT01.to(dtype=u.dtype, device=u.device)
    a = u.new_zeros(B, 4)
    a[:, 0] = 1.0
    b = u.new_zeros(B, 4)
    zeta = zeta_scale * zz_zeta(u)
    c, s = torch.cos(zeta), torch.sin(zeta)
    mask = u.new_tensor([0.0, 1.0, 1.0, 0.0])          # indices with b0 XOR b1 = 1
    for l in range(theta.shape[0]):
        R = BH._ry_pair(u[:, 0], u[:, 1])
        a = torch.einsum("bij,bj->bi", R, a)
        b = torch.einsum("bij,bj->bi", R, b)
        cm = 1.0 + mask * (c.unsqueeze(1) - 1.0)          # cos on phased entries, 1 elsewhere
        sm = mask * s.unsqueeze(1)                        # sin on phased entries, 0 elsewhere
        a, b = a * cm - b * sm, a * sm + b * cm
        t = theta[l].unsqueeze(0).expand(B, 2)
        R2 = BH._ry_pair(t[:, 0], t[:, 1])
        a = torch.einsum("bij,bj->bi", R2, a)
        b = torch.einsum("bij,bj->bi", R2, b)
        a = a @ cnot.T
        b = b @ cnot.T
    return (torch.einsum("bi,pij,bj->bp", a, pr, a) + torch.einsum("bi,pij,bj->bp", b, pr, b)
            + 2.0 * torch.einsum("bi,pij,bj->bp", b, pi_, a))


class BFRQuantumZZ(BH.BFRQuantum):
    """BFRQuantum with the ZZ product phase restored. Same parameters, same init (same generator draw)."""

    def __init__(self, K, n_layers=BH.N_LAYERS, generator=None):
        super().__init__(K, n_layers=n_layers, generator=generator)
        self.register_buffer("paulis_im", pauli_stack_imag().float(), persistent=False)
        self.zeta_scale = 1.0

    def features(self, z):
        return bfrzz_expectations(self.encode(z), self.theta, self.zeta_scale, self.paulis.to(z.dtype),
                                  self.paulis_im.to(z.dtype), self.cnot.to(z.dtype))


class TrimHead(nn.Module):
    """M4-trim: fc2 -> fc_mid(2 -> w) -> fc3(w -> K), linear. No quantum branch."""

    def __init__(self, fc2, fc_mid, fc3):
        super().__init__()
        self.fc2, self.fc_mid, self.fc3 = fc2, fc_mid, fc3

    def classical(self, z):
        return self.fc3(self.fc_mid(z))

    def forward(self, x):
        return self.classical(self.fc2(x))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


def trim_params(K, w):
    return 1464 + 130 + 3 * w + w * K + K


def _rng_save():
    st = dict(py=random.getstate(), np=np.random.get_state(), t=torch.get_rng_state())
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _rng_restore(st):
    random.setstate(st["py"]); np.random.set_state(st["np"]); torch.set_rng_state(st["t"])
    if "cuda" in st:
        torch.cuda.set_rng_state_all(st["cuda"])


def install(model, seed, kind, head_variant="linear"):
    """kind 'm4trim' or 'bfrzz'. M4 -> variant in place, in a forked RNG. Returns a dict."""
    if kind not in ("m4trim", "bfrzz"):
        raise ExtError(f"unknown kind {kind}")
    if head_variant != "linear":
        raise ExtError("defined with the linear head only")
    st = _rng_save()
    try:
        be = getattr(model, "back_end", None)
        if be is None or type(be).__name__ != "ClassicalHead":
            raise ExtError(f"back_end is {type(be).__name__}, expected M4's ClassicalHead")
        if any(p.numel() for p in model.front_end.parameters()):
            raise ExtError("front_end has parameters: built on M4 only")
        fc2, fm, f3 = be.fc2, be.fc_mid, be.fc3
        if fm is None or f3 is None or fm.out_features != BH.MID_FULL or f3.in_features != BH.MID_FULL:
            raise ExtError("head is not M4 at w15")
        K = f3.out_features
        before = sum(p.numel() for p in model.parameters())
        w = BH.choose_width(K)
        dev, dt = f3.weight.device, f3.weight.dtype
        new_mid = nn.Linear(2, w).to(device=dev, dtype=dt)
        new_f3 = nn.Linear(w, K).to(device=dev, dtype=dt)
        scale = math.sqrt(BH.MID_FULL / w)
        with torch.no_grad():
            new_mid.weight.copy_(fm.weight[:w]); new_mid.bias.copy_(fm.bias[:w])
            new_f3.weight.copy_(f3.weight[:, :w] * scale); new_f3.bias.copy_(f3.bias * scale)
        if kind == "m4trim":
            model.back_end = TrimHead(fc2, new_mid, new_f3)
            expected = trim_params(K, w)
        else:
            q = BFRQuantumZZ(K, generator=torch.Generator().manual_seed(int(seed))).to(device=dev, dtype=dt)
            model.back_end = BH.BFRHead(fc2, new_mid, new_f3, q, head_variant)
            expected = BH.bfr_params(K, w)
        after = sum(p.numel() for p in model.parameters())
        return dict(kind=kind, K=K, width=w, params_m4=before, params=after, expected=expected,
                    ok=(before == BH.m4_params(K) and after == expected))
    finally:
        _rng_restore(st)


def expected_count(kind, K):
    w = BH.choose_width(K)
    return trim_params(K, w) if kind == "m4trim" else BH.bfr_params(K, w)
