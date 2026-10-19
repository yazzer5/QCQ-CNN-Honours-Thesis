"""
quanv_noise_v15.py -- IBM-calibrated noise for the quanvolutional FRONT-END (v15).

WHAT IT DOES
------------
Adds device noise, calibrated from ibm_brisbane (calib/ibm_brisbane_calib.json),
to the two quantum front-ends in quanv_torch.py, scaled by one factor alpha:

    M2, M3, M5 : TrainableQuanvFrontEndTorch   (Circuit 4, two 2-qubit blocks)
    M1         : FrozenHendersonFrontEndTorch  (RandomLayers, 4 qubits)

The circuits, parameters and initialisation are NOT touched. Noise is installed
by replacing the front-end instance's forward(); at alpha = 0 the ORIGINAL
forward runs (bit-exact with v13 / v13.5 / v14). The QNN head (M1, M2) is
noiseless in v15 by design (pre-registration section 2).

NOISE MODEL (the same channels Qiskit Aer's NoiseModel.from_backend builds)
----------------------------------------------------------------------------
Each logical gate is charged the native IBM gates it compiles to:
    RY(t), RX(t)  -> 2 x SX  (+ virtual RZs, which are error-free and take 0 time)
    RZ(t)         -> virtual, no error
    CRZ(t)        -> 2 x CX  -> 2 x ECR
    CNOT          -> 1 x ECR
Native gate channel = depolarizing(p) then thermal relaxation(T1, T2, duration)
on each qubit it touches, with p chosen so the channel's average gate infidelity
equals the reported gate error (qiskit_aer.noise.device.models, same formula).
Measurement: per-qubit assignment error applied to <Z>:
    <Z>_measured = (1 - P(1|0) - P(0|1)) <Z> + (P(0|1) - P(1|0))
with P(1|0) = P(0|1) = the qubit's readout_error, as from_backend does for
IBM's BackendV2 devices (so here <Z>_measured = (1 - 2 e_ro) <Z>).
Expectation values are EXACT (no shot noise), as in every run since v5.

ALPHA SCALING
-------------
alpha multiplies every error rate: gate errors, readout errors, and the
relaxation rates 1/T1, 1/T2 (i.e. T1 -> T1/alpha, T2 -> T2/alpha). alpha = 1 is
the calibrated device; alpha = 0 is the noiseless simulator. Physical caps:
gate error <= d/(d+1), readout <= 0.5, T2 <= 2 T1 (Qiskit applies the same).

HOW IT IS COMPUTED
------------------
Pauli-transfer matrices (PTMs). A density matrix is the real vector
r_i = Tr(P_i rho) over Pauli strings; every gate and channel is a real matrix on r.
  * Circuit 4: one 16x16 PTM per forward pass, differentiable in the 6 weights
    (autograd through the gate PTMs). Each patch's encoded 2-qubit state is the
    Kronecker product of two single-qubit Pauli vectors, so the cost per patch
    is one 16-vector dot product per output channel.
  * Henderson: the 256x256 PTM of the fixed RandomLayers circuit is built once
    per alpha from PennyLane's decomposition of the SAME RandomLayers op, and the
    decomposition is asserted to reproduce the module's stored unitary.
Verified (scripts/verify_v15.py) against Qiskit Aer density-matrix simulation
and against the channels NoiseModel.from_backend(FakeBrisbane) attaches to sx and
ecr on the calibrated qubits.

CAVEATS (stated in the pre-registration)
  * Device-median calibration, no qubit mapping, no routing SWAPs, no crosstalk,
    no idle (delay) errors -- the same omissions as an unscheduled Aer
    from_backend simulation. RandomLayers' CNOTs between non-neighbouring qubits
    would need SWAPs on heavy-hex; they are charged one ECR each (optimistic for M1).
  * CX -> ECR conversion's extra single-qubit gates are not charged (< 3% of the
    2-qubit error budget at the median calibration).
"""
import json
import math

import numpy as np
import torch

# ════════════════════════════════════════════════════════════════════════════
# Pauli basis
# ════════════════════════════════════════════════════════════════════════════
_P1 = np.array([[[1, 0], [0, 1]],
                [[0, 1], [1, 0]],
                [[0, -1j], [1j, 0]],
                [[1, 0], [0, -1]]], dtype=complex)                      # I X Y Z


def pauli_basis(n):
    """(4^n, 2^n, 2^n). Index = base-4 digits, qubit 0 (wire 0) most significant."""
    B = _P1
    for _ in range(n - 1):
        B = np.einsum("aij,bkl->abikjl", B, _P1).reshape(B.shape[0] * 4, B.shape[1] * 2, B.shape[2] * 2)
    return B


_PB = {1: pauli_basis(1), 2: pauli_basis(2), 4: pauli_basis(4)}


def ptm_of_unitary_np(U):
    n = int(round(math.log2(U.shape[0])))
    P = _PB[n]
    d = U.shape[0]
    UPU = np.einsum("ab,jbc,dc->jad", U, P, U.conj())                   # U P_j U^dag
    return np.einsum("iab,jba->ij", P, UPU).real / d


def ptm_of_unitary_torch(U, P):
    d = U.shape[0]
    UPU = torch.einsum("ab,jbc,dc->jad", U, P, U.conj())
    return torch.einsum("iab,jba->ij", P, UPU).real / d


# ════════════════════════════════════════════════════════════════════════════
# Channels (numpy float64, constant for a given alpha)
# ════════════════════════════════════════════════════════════════════════════
def thermal_ptm(t, T1, T2):
    """Qiskit thermal_relaxation_error(T1, T2, t, excited_state_population=0) as a PTM."""
    if t <= 0 or (math.isinf(T1) and math.isinf(T2)):
        return np.eye(4)
    T2 = min(T2, 2 * T1)
    e1 = 0.0 if T1 == 0 else math.exp(-t / T1)
    e2 = 0.0 if T2 == 0 else math.exp(-t / T2)
    R = np.diag([1.0, e2, e2, e1])
    R[3, 0] = 1.0 - e1                                                  # relaxes towards |0>
    return R


def avg_gate_fidelity(R):
    d = int(round(math.sqrt(R.shape[0])))
    return (np.trace(R) / d + 1) / (d + 1)                              # (d F_pro + 1)/(d+1), F_pro = Tr R / d^2


def depol_param(err, relax_R, n):
    """qiskit_aer.noise.device.models._device_depolarizing_error, solved for p."""
    dim = 2 ** n
    err = min(err, dim / (dim + 1))
    f_rel = avg_gate_fidelity(relax_R)
    p = dim * (err - (1 - f_rel)) / (dim * f_rel - 1)
    return float(min(max(p, 0.0), 4 ** n / (4 ** n - 1)))


def depol_ptm(p, n):
    return np.diag([1.0] + [1.0 - p] * (4 ** n - 1))


class NoiseParams:
    """All alpha-scaled channels for one device configuration.

    phys : None -> device median for every logical qubit (production);
           (qa, qb) -> the calibrated values of those physical qubits / that edge
           (verification against NoiseModel.from_backend on the same qubits).
    """

    def __init__(self, calib, alpha, phys=None):
        self.alpha = float(alpha)
        a = self.alpha
        if phys is None:
            m = calib["median"]
            qs = [m, m]
            edge = m
        else:
            qa, qb = phys
            qs = [calib["qubits"][str(qa)], calib["qubits"][str(qb)]]
            edge = calib["edges"].get(f"{qa}-{qb}") or calib["edges"][f"{qb}-{qa}"]
        self.q = []
        for qp in qs:
            T1 = qp["t1"] / a if a > 0 else math.inf
            T2 = qp["t2"] / a if a > 0 else math.inf
            rel = thermal_ptm(qp["sx_length"], T1, T2) if a > 0 else np.eye(4)
            p = depol_param(a * qp["sx_error"], rel, 1) if a > 0 else 0.0
            sx = rel @ depol_ptm(p, 1)                                 # depol, then relaxation
            # NoiseModel.from_backend (BackendV2 target) uses the symmetric
            # readout_error for both P(1|0) and P(0|1); so do we.
            pa = pb = min(a * qp["readout_error"], 0.5)
            self.q.append(dict(T1=T1, T2=T2, p_sx=p, sx=sx, ry=sx @ sx, ro=(pa, pb)))
        # two-qubit ECR channel on (logical a, logical b)
        if a > 0:
            rel2 = np.kron(thermal_ptm(edge["ecr_length"], self.q[0]["T1"], self.q[0]["T2"]),
                           thermal_ptm(edge["ecr_length"], self.q[1]["T1"], self.q[1]["T2"]))
            p2 = depol_param(a * edge["ecr_error"], rel2, 2)
        else:
            rel2, p2 = np.eye(16), 0.0
        self.p_ecr = p2
        self.ecr = rel2 @ depol_ptm(p2, 2)
        self.crz = self.ecr @ self.ecr

    def summary(self):
        q = self.q[0]
        return (f"p_depol(sx)={q['p_sx']:.3e} p_depol(ecr)={self.p_ecr:.3e} "
                f"T1={q['T1']*1e6 if q['T1'] < math.inf else math.inf:.1f}us "
                f"T2={q['T2']*1e6 if q['T2'] < math.inf else math.inf:.1f}us "
                f"P(1|0)={q['ro'][0]:.4f} P(0|1)={q['ro'][1]:.4f}")


def load_calib(path):
    with open(path) as f:
        c = json.load(f)
    if "median" not in c or "sha256" not in c:
        raise SystemExit(f"{path}: not a v15 calibration file")
    return c


# ════════════════════════════════════════════════════════════════════════════
# Embedding PTMs on 4 wires (Henderson)
# ════════════════════════════════════════════════════════════════════════════
def embed_ptm(R, wires, n=4):
    """PTM acting on `wires` (list, in R's own qubit order) -> PTM on n qubits."""
    k = len(wires)
    Rt = R.reshape([4] * (2 * k))
    out_idx = list(range(n))
    in_idx = list(range(n, 2 * n))
    # identity on untouched wires
    eye_terms = []
    for w in range(n):
        if w not in wires:
            eye_terms.append((np.eye(4), [out_idx[w], in_idx[w]]))
    ops = [Rt, [out_idx[w] for w in wires] + [in_idx[w] for w in wires]]
    for E, idx in eye_terms:
        ops += [E, idx]
    full = np.einsum(*ops, out_idx + in_idx)
    return full.reshape(4 ** n, 4 ** n)


# ════════════════════════════════════════════════════════════════════════════
# Encoding: RY(pi x)|0>, then the RY's native-gate channel, as a Pauli vector
# ════════════════════════════════════════════════════════════════════════════
def encode_qubit(phi, ry_ch):
    """phi: (...,) angles -> (..., 4) Pauli vector [1, x, y, z] after RY(phi) + channel."""
    one = torch.ones_like(phi)
    r = torch.stack([one, torch.sin(phi), torch.zeros_like(phi), torch.cos(phi)], -1)
    return r @ ry_ch.transpose(0, 1)


def readout(z, ro):
    pa, pb = ro
    return (1.0 - pa - pb) * z + (pb - pa)


# ════════════════════════════════════════════════════════════════════════════
# Circuit 4 (M2 / M3 / M5): differentiable PTM of one block
# ════════════════════════════════════════════════════════════════════════════
def circuit4_block_ptm(w, NP, device):
    """16x16 PTM of the Circuit-4 block with noise, differentiable in w (6,).
    Gate order is quanv_torch.TrainableQuanvFrontEndTorch._block_unitary exactly
    (Hur et al. 2022 Fig. 2(d), re-checked 2026-10-17):
      RY(w0) q_a ; RY(w1) q_b ; CRZ(w2) ctrl q_b tgt q_a ;
      RY(w3) q_a ; RY(w4) q_b ; CRZ(w5) ctrl q_a tgt q_b
    """
    cdt = torch.complex128
    w = w.to(torch.float64)
    P = torch.as_tensor(_PB[2], dtype=cdt, device=device)
    I2 = torch.eye(2, dtype=cdt, device=device)
    one = torch.ones((), dtype=cdt, device=device)

    def ry(t):
        c, s = torch.cos(t / 2), torch.sin(t / 2)
        return torch.stack([torch.stack([c, -s]), torch.stack([s, c])]).to(cdt)

    def kron(A, B):
        return torch.einsum("ij,kl->ikjl", A, B).reshape(4, 4)

    def crz(t, ctrl_is_a):
        m = torch.exp(-1j * t.to(cdt) / 2)
        p = torch.exp(1j * t.to(cdt) / 2)
        d = torch.stack([one, one, m, p]) if ctrl_is_a else torch.stack([one, m, one, p])
        return torch.diag(d)

    T = lambda a: torch.as_tensor(a, dtype=torch.float64, device=device)  # noqa: E731
    ch_a = T(np.kron(NP.q[0]["ry"], np.eye(4)))
    ch_b = T(np.kron(np.eye(4), NP.q[1]["ry"]))
    ch_2 = T(NP.crz)
    steps = [(kron(ry(w[0]), I2), ch_a), (kron(I2, ry(w[1])), ch_b),
             (crz(w[2], ctrl_is_a=False), ch_2),
             (kron(ry(w[3]), I2), ch_a), (kron(I2, ry(w[4])), ch_b),
             (crz(w[5], ctrl_is_a=True), ch_2)]
    S = torch.eye(16, dtype=torch.float64, device=device)
    for U, C in steps:
        S = C @ ptm_of_unitary_torch(U, P) @ S
    return S


def circuit4_expvals(pairs_angle, w, NP):
    """pairs_angle: (M, 2) encoding angles (q_a, q_b) of M blocks.
    Returns (M, 2) measured <Z_a>, <Z_b> (readout error applied)."""
    dev = pairs_angle.device
    phi = pairs_angle.to(torch.float64)
    ra = encode_qubit(phi[:, 0], torch.as_tensor(NP.q[0]["ry"], dtype=torch.float64, device=dev))
    rb = encode_qubit(phi[:, 1], torch.as_tensor(NP.q[1]["ry"], dtype=torch.float64, device=dev))
    S = circuit4_block_ptm(w, NP, dev)
    W = S[[12, 3], :]                                     # rows Z(x)I and I(x)Z
    rin = torch.einsum("mi,mj->mij", ra, rb).reshape(-1, 16)
    z = rin @ W.transpose(0, 1)                           # (M, 2)
    return torch.stack([readout(z[:, 0], NP.q[0]["ro"]), readout(z[:, 1], NP.q[1]["ro"])], -1)


def circuit4_noisy_forward(fe, x, NP):
    """Same patching, ordering and output layout as TrainableQuanvFrontEndTorch.forward."""
    B, _, H, W = x.shape
    oh, ow = H // 2, W // 2
    patches = (x.unfold(2, 2, 2).unfold(3, 2, 2)
                .permute(0, 2, 3, 1, 4, 5)
                .reshape(B * oh * ow, 4))
    ang = patches * np.pi
    pairs = torch.stack([ang[:, 0:2], ang[:, 2:4]], 1).reshape(-1, 2)   # (2P, 2) blocks A, B
    z = circuit4_expvals(pairs, fe.weights, NP)                        # (2P, 2)
    ea = z[:, 0].reshape(-1, 2)
    eb = z[:, 1].reshape(-1, 2)
    out = torch.stack([ea[:, 0], eb[:, 0], ea[:, 1], eb[:, 1]], -1)
    return out.reshape(B, oh, ow, 4).permute(0, 3, 1, 2).to(x.dtype)


# ════════════════════════════════════════════════════════════════════════════
# Henderson / RandomLayers (M1): fixed 256x256 PTM per alpha
# ════════════════════════════════════════════════════════════════════════════
def henderson_gates(seed, n_layers=1):
    """PennyLane decomposition of the SAME RandomLayers op FrozenHendersonFrontEndTorch
    builds: [(name, wires, 16x16 unitary on 4 wires)]."""
    import pennylane as qml
    g = np.random.RandomState(seed)
    rand_params = g.uniform(high=2 * np.pi, size=(n_layers, 4))
    op = qml.RandomLayers(rand_params, wires=list(range(4)))
    ops = op.decomposition()
    while any(o.name not in ("RX", "RY", "RZ", "CNOT") for o in ops):
        new = []
        for o in ops:
            new += [o] if o.name in ("RX", "RY", "RZ", "CNOT") else o.decomposition()
        ops = new
    out = []
    for o in ops:
        U = np.asarray(qml.matrix(o, wire_order=list(range(4))), dtype=complex)
        out.append((o.name, [int(w) for w in o.wires], U))
    return out


def henderson_ptm_rows(gates, NP):
    """(4, 256): rows of the full noisy PTM that give <Z_0..3> (pre-readout)."""
    S = np.eye(256)
    for name, wires, U in gates:
        S = ptm_of_unitary_np(U) @ S
        if name in ("RX", "RY"):
            S = embed_ptm(NP.q[0]["ry"], wires) @ S
        elif name == "CNOT":
            S = embed_ptm(NP.ecr, wires) @ S
        # RZ: virtual, no error
    zrows = [3 * 4 ** (3 - w) for w in range(4)]          # Z on wire w, I elsewhere
    return S[zrows, :]


def henderson_expvals(ang4, rows, NP):
    """ang4: (P, 4) encoding angles -> (P, 4) measured <Z>."""
    dev = ang4.device
    ry = torch.as_tensor(NP.q[0]["ry"], dtype=torch.float64, device=dev)
    r = encode_qubit(ang4.to(torch.float64), ry)                      # (P, 4, 4)
    rin = r[:, 0, :]
    for w in range(1, 4):
        rin = torch.einsum("pi,pj->pij", rin, r[:, w, :]).reshape(rin.shape[0], -1)
    z = rin @ torch.as_tensor(rows, dtype=torch.float64, device=dev).transpose(0, 1)
    return torch.stack([readout(z[:, i], NP.q[0]["ro"]) for i in range(4)], -1)


def henderson_noisy_forward(fe, x, rows, NP):
    B, _, H, W = x.shape
    oh, ow = H // 2, W // 2
    patches = (x.unfold(2, 2, 2).unfold(3, 2, 2)
                .permute(0, 2, 3, 1, 4, 5)
                .reshape(B * oh * ow, 4))
    out = henderson_expvals(patches * np.pi, rows, NP)
    return out.reshape(B, oh, ow, 4).permute(0, 3, 1, 2).to(x.dtype)


# ════════════════════════════════════════════════════════════════════════════
# Installing noise on a built model
# ════════════════════════════════════════════════════════════════════════════
class FrontEndNoise:
    """Owns the alpha state of one front-end instance. set_alpha(0) restores the
    ORIGINAL forward path exactly."""

    def __init__(self, fe, kind, calib, seed=None):
        self.fe, self.kind, self.calib, self.seed = fe, kind, calib, seed
        self.alpha = 0.0
        self.NP = None
        self._orig = fe.forward                      # bound original method
        self._rows = {}
        self._gates = None
        if kind == "henderson":
            self._gates = henderson_gates(seed)
            U = np.eye(16, dtype=complex)
            for _, _, G in self._gates:
                U = G @ U
            stored = fe.U.detach().cpu().numpy().astype(complex)
            self.decomp_err = float(np.abs(U - stored).max())
        else:
            self.decomp_err = 0.0
        fe.forward = self._forward

    def set_alpha(self, alpha):
        self.alpha = float(alpha)
        self.NP = NoiseParams(self.calib, self.alpha) if self.alpha > 0 else None
        if self.kind == "henderson" and self.alpha > 0 and self.alpha not in self._rows:
            self._rows[self.alpha] = henderson_ptm_rows(self._gates, self.NP)

    def _forward(self, x):
        if self.alpha == 0:
            return self._orig(x)
        if self.kind == "circuit4":
            return circuit4_noisy_forward(self.fe, x, self.NP)
        return henderson_noisy_forward(self.fe, x, self._rows[self.alpha], self.NP)

    def ptm_path_at_zero(self, x):
        """The PTM path with every error rate set to 0 (for the alpha=0 self-check)."""
        NP0 = NoiseParams(self.calib, 0.0)
        if self.kind == "circuit4":
            return circuit4_noisy_forward(self.fe, x, NP0)
        return henderson_noisy_forward(self.fe, x, henderson_ptm_rows(self._gates, NP0), NP0)
