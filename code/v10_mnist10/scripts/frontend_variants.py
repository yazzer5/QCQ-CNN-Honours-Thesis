"""
frontend_variants.py — evidence for choosing how to strengthen M2's front-end.

Builds a configurable 4-qubit quanvolutional front-end and measures three
things for each candidate design:

  1. EFFECTIVE PARAMETERS — rank of the Jacobian d(outputs)/d(theta) pooled
     over many random inputs and random parameter settings. This is the real
     number of independent knobs, which can be far below the nominal count if
     the ansatz has redundant directions (Circuit 4's final CRZ is the known
     example: nominally 6 parameters, effectively 5).

  2. PIXEL CONNECTIVITY — for each output channel, how many of the 4 input
     pixels it actually depends on. The current design scores badly here:
     two of four channels are single-pixel functions and nothing crosses the
     top/bottom row boundary.

  3. EXPRESSIBILITY — Sim et al. (2019) style. Sample pairs of random
     parameter settings, compute state fidelities, and compare the resulting
     distribution to the Haar-random one. Lower KL = covers more of Hilbert
     space. Reported as KL divergence, so smaller is better.

Conventions match quanv_torch.py: wire 0 is the MOST significant bit of the
state index, RY encoding is x*pi, measurement is PauliZ on every wire.
"""

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
C = torch.complex128
NQ = 4
DIM = 16


# ── gate plumbing on 4 qubits ──────────────────────────────────────────
def _op(single, wire):
    """Embed a 2x2 gate on `wire` into the 16-dim space (wire 0 = MSB)."""
    ops = [torch.eye(2, dtype=C)] * NQ
    ops[wire] = single
    M = ops[0]
    for o in ops[1:]:
        M = torch.einsum("ij,kl->ikjl", M, o).reshape(M.shape[0] * 2, -1)
    return M


def _ry(t):
    c, s = torch.cos(t / 2), torch.sin(t / 2)
    return torch.stack([torch.stack([c, -s]), torch.stack([s, c])]).to(C)


def _crz_diag(t, ctrl, tgt):
    """CRZ as a diagonal vector over the 16 basis states."""
    d = []
    for idx in range(DIM):
        b = [(idx >> (NQ - 1 - k)) & 1 for k in range(NQ)]
        if b[ctrl] == 1:
            d.append(torch.exp(1j * t.to(C) / 2) if b[tgt] == 1
                     else torch.exp(-1j * t.to(C) / 2))
        else:
            d.append(torch.ones((), dtype=C))
    return torch.stack(d)


def _cz_diag(qa, qb):
    d = []
    for idx in range(DIM):
        b = [(idx >> (NQ - 1 - k)) & 1 for k in range(NQ)]
        d.append(torch.tensor(-1.0 + 0j, dtype=C) if (b[qa] and b[qb])
                 else torch.ones((), dtype=C))
    return torch.stack(d)


def _c4_half1(psi, w, a, b):
    psi = psi @ _op(_ry(w[0]), a).transpose(0, 1)
    psi = psi @ _op(_ry(w[1]), b).transpose(0, 1)
    psi = psi * _crz_diag(w[2], b, a)
    return psi


def _c4_half2(psi, w, a, b):
    psi = psi @ _op(_ry(w[3]), a).transpose(0, 1)
    psi = psi @ _op(_ry(w[4]), b).transpose(0, 1)
    psi = psi * _crz_diag(w[5], a, b)
    return psi


def _c4_block(psi, w, a, b):
    """One Circuit 4 block on wires (a,b). w is a 6-vector."""
    psi = psi @ _op(_ry(w[0]), a).transpose(0, 1)
    psi = psi @ _op(_ry(w[1]), b).transpose(0, 1)
    psi = psi * _crz_diag(w[2], b, a)          # ctrl=b, tgt=a
    psi = psi @ _op(_ry(w[3]), a).transpose(0, 1)
    psi = psi @ _op(_ry(w[4]), b).transpose(0, 1)
    psi = psi * _crz_diag(w[5], a, b)          # ctrl=a, tgt=b
    return psi


def encode(x):
    """x: (B,4) -> (B,16) product state from RY(x*pi)|0>."""
    B = x.shape[0]
    ang = x * np.pi
    psi = torch.ones(B, 1, dtype=C)
    for q in range(NQ):
        c = torch.cos(ang[:, q] / 2).to(C).unsqueeze(1)
        s = torch.sin(ang[:, q] / 2).to(C).unsqueeze(1)
        psi = torch.cat([psi * c, psi * s], dim=1)
    return psi


def run(x, theta, layers=1, cross="none", reupload=False):
    """
    cross : "none"  -> two independent Circuit 4 blocks (CURRENT DESIGN)
            "crz"   -> + trainable CRZ between wires 1 and 2 each layer
            "cz"    -> + fixed CZ between wires 1 and 2 each layer (0 params)
            "ring"  -> + fixed CZ on (1,2) and (3,0) each layer (0 params)
    """
    per = 6 + (1 if cross in ("crz", "crz_mid") else 0)
    psi = encode(x)
    for L in range(layers):
        if reupload and L > 0:
            psi = psi * torch.exp(torch.zeros(1, dtype=C))  # no-op guard
            psi = _reupload(psi, x)
        w = theta[L * per:(L + 1) * per]
        if cross.endswith("_mid"):
            # entangler placed BETWEEN the halves of Circuit 4, so the
            # following RY layer converts its phase into a measurable
            # Z-basis change. Placing it at the END makes it inert -- it is
            # diagonal, and a diagonal gate immediately before a diagonal
            # measurement cannot change the statistics (same reason w[5] is
            # dead in the single-layer design).
            psi = _c4_half1(psi, w, 0, 1)
            psi = _c4_half1(psi, w, 2, 3)
            if cross == "cz_mid":
                psi = psi * _cz_diag(1, 2)
            elif cross == "crz_mid":
                psi = psi * _crz_diag(w[6], 1, 2)
            elif cross == "ring_mid":
                psi = psi * _cz_diag(1, 2) * _cz_diag(3, 0)
            psi = _c4_half2(psi, w, 0, 1)
            psi = _c4_half2(psi, w, 2, 3)
        else:
            psi = _c4_block(psi, w, 0, 1)
            psi = _c4_block(psi, w, 2, 3)
            if cross == "crz":
                psi = psi * _crz_diag(w[6], 1, 2)
            elif cross == "cz":
                psi = psi * _cz_diag(1, 2)
            elif cross == "ring":
                psi = psi * _cz_diag(1, 2) * _cz_diag(3, 0)
    p = (psi.conj() * psi).real
    outs = []
    for q in range(NQ):
        z = torch.tensor([1.0 - 2 * ((i >> (NQ - 1 - q)) & 1) for i in range(DIM)])
        outs.append(p @ z)
    return torch.stack(outs, dim=-1)            # (B,4)


def _reupload(psi, x):
    ang = x * np.pi
    for q in range(NQ):
        psi = psi @ _op(_ry(ang[0, q] if x.shape[0] == 1 else ang[:, q].mean()), q).transpose(0, 1)
    return psi


def n_params(layers, cross):
    return layers * (6 + (1 if cross in ("crz", "crz_mid") else 0))


# ── metric 1: effective parameters (Jacobian rank) ────────────────────
def effective_params(layers, cross, trials=4, n_x=24, seed=0):
    """Rank of d(outputs)/d(theta). Uses torch.func.jacrev for speed."""
    from torch.func import jacrev
    g = torch.Generator().manual_seed(seed)
    P = n_params(layers, cross)
    ranks = []
    for _ in range(trials):
        th = torch.rand(P, generator=g, dtype=torch.float64) * 2 * np.pi
        x = torch.rand(n_x, 4, generator=g, dtype=torch.float64)
        J = jacrev(lambda t: run(x, t, layers, cross).reshape(-1))(th)
        ranks.append(torch.linalg.matrix_rank(J, rtol=1e-8).item())
    return int(round(float(np.mean(ranks))))


# ── metric 2: pixel connectivity ──────────────────────────────────────
def connectivity(layers, cross, seed=0, thresh=1e-9, n=40):
    g = torch.Generator().manual_seed(seed)
    th = torch.rand(n_params(layers, cross), generator=g, dtype=torch.float64) * 2 * np.pi
    dep = np.zeros((4, 4))                       # [channel, pixel]
    rs = np.random.RandomState(seed)
    for _ in range(n):
        p = torch.tensor(rs.rand(1, 4))
        a = run(p, th, layers, cross)[0].detach().numpy()
        for j in range(4):
            q = p.clone(); q[0, j] = torch.tensor(rs.rand())
            b = run(q, th, layers, cross)[0].detach().numpy()
            dep[:, j] = np.maximum(dep[:, j], np.abs(a - b))
    seen = (dep > thresh).sum(axis=1)             # pixels seen per channel
    return seen, dep


# ── metric 3: expressibility (Sim et al. 2019) ────────────────────────
def expressibility(layers, cross, n_pairs=600, bins=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    P = n_params(layers, cross)
    x = torch.full((1, 4), 0.5, dtype=torch.float64)   # fixed input
    fid = []
    for _ in range(n_pairs):
        t1 = torch.rand(P, generator=g, dtype=torch.float64) * 2 * np.pi
        t2 = torch.rand(P, generator=g, dtype=torch.float64) * 2 * np.pi
        s1 = _state(x, t1, layers, cross)
        s2 = _state(x, t2, layers, cross)
        fid.append(abs(torch.vdot(s1, s2).item()) ** 2)
    hist, edges = np.histogram(fid, bins=bins, range=(0, 1), density=False)
    P_emp = hist / hist.sum()
    # Haar for dimension N: p(F) = (N-1)(1-F)^(N-2)
    N = DIM
    ctr = (edges[:-1] + edges[1:]) / 2
    P_haar = (N - 1) * (1 - ctr) ** (N - 2)
    P_haar = P_haar / P_haar.sum()
    m = P_emp > 0
    return float(np.sum(P_emp[m] * np.log(P_emp[m] / P_haar[m])))


def _state(x, th, layers, cross):
    return run_state(x, th, layers, cross)[0]


def run_state(x, theta, layers, cross):
    per = n_params(layers, cross) // layers
    psi = encode(x)
    for L in range(layers):
        w = theta[L * per:(L + 1) * per]
        if cross.endswith("_mid"):
            psi = _c4_half1(psi, w, 0, 1); psi = _c4_half1(psi, w, 2, 3)
            if cross == "cz_mid":
                psi = psi * _cz_diag(1, 2)
            elif cross == "crz_mid":
                psi = psi * _crz_diag(w[6], 1, 2)
            elif cross == "ring_mid":
                psi = psi * _cz_diag(1, 2) * _cz_diag(3, 0)
            psi = _c4_half2(psi, w, 0, 1); psi = _c4_half2(psi, w, 2, 3)
        else:
            psi = _c4_block(psi, w, 0, 1); psi = _c4_block(psi, w, 2, 3)
            if cross == "crz":
                psi = psi * _crz_diag(w[6], 1, 2)
            elif cross == "cz":
                psi = psi * _cz_diag(1, 2)
            elif cross == "ring":
                psi = psi * _cz_diag(1, 2) * _cz_diag(3, 0)
    return psi


if __name__ == "__main__":
    designs = [
        ("current (L=1, no cross)",        1, "none"),
        ("L=2, no cross",                  2, "none"),
        ("L=3, no cross",                  3, "none"),
        ("L=4, no cross",                  4, "none"),
        ("L=1 + fixed CZ(1,2)",            1, "cz"),
        ("L=1 + trainable CRZ(1,2)",       1, "crz"),
        ("L=1 + CZ ring",                  1, "ring"),
        ("L=2 + CZ ring",                  2, "ring"),
        ("L=3 + CZ ring",                  3, "ring"),
        ("L=2 + trainable CRZ(1,2)",       2, "crz"),
        ("L=3 + trainable CRZ(1,2)",       3, "crz"),
    ]
    print(f"{'design':30s} {'nominal':>8s} {'effective':>10s} "
          f"{'pixels/ch':>18s} {'expr KL':>9s}")
    print("-" * 82)
    for name, L, cr in designs:
        nom = n_params(L, cr)
        eff = effective_params(L, cr)
        seen, _ = connectivity(L, cr)
        kl = expressibility(L, cr)
        print(f"{name:30s} {nom:8d} {eff:10d} {str(list(seen)):>18s} {kl:9.4f}")
    print("\npixels/ch = number of the 4 input pixels each output channel depends on.")
    print("[4,4,4,4] is full connectivity. expr KL: lower = covers more of Hilbert space.")
