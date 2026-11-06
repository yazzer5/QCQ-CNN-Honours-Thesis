"""Verify: (a) the 4-qubit Circuit-4 front-end factorises into two independent
2-qubit blocks, (b) a batched torch implementation reproduces a brute-force
4-qubit reference exactly, (c) torch autograd gradients == parameter-shift
gradients to numerical tolerance."""
import numpy as np, torch

torch.set_default_dtype(torch.float64)

# ---------- brute force 4-qubit reference (PennyLane conventions) ----------
I2 = np.eye(2, dtype=complex)


def RY(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def op_on(u, wire, n=4):
    ops = [I2] * n
    ops[wire] = u
    M = ops[0]
    for o in ops[1:]:
        M = np.kron(M, o)
    return M


def crz_full(t, ctrl, tgt, n=4):
    """diag CRZ: phase e^{-i t/2} on tgt=0, e^{+i t/2} on tgt=1, only when ctrl=1."""
    d = np.ones(2 ** n, dtype=complex)
    for idx in range(2 ** n):
        bits = [(idx >> (n - 1 - k)) & 1 for k in range(n)]
        if bits[ctrl] == 1:
            d[idx] = np.exp(1j * t / 2) if bits[tgt] == 1 else np.exp(-1j * t / 2)
    return np.diag(d)


def ref_expvals(x, w):
    """x: (4,) inputs, w: (6,) weights -> (4,) PauliZ expvals."""
    n = 4
    psi = np.array([1.0 + 0j])
    for i in range(n):
        psi = np.kron(psi, RY(x[i] * np.pi) @ np.array([1, 0], dtype=complex))
    for (a, b) in [(0, 1), (2, 3)]:
        psi = op_on(RY(w[0]), a) @ psi
        psi = op_on(RY(w[1]), b) @ psi
        psi = crz_full(w[2], b, a) @ psi
        psi = op_on(RY(w[3]), a) @ psi
        psi = op_on(RY(w[4]), b) @ psi
        psi = crz_full(w[5], a, b) @ psi
    p = np.abs(psi) ** 2
    out = []
    for wire in range(n):
        z = np.array([1 - 2 * ((idx >> (n - 1 - wire)) & 1) for idx in range(2 ** n)])
        out.append(float((p * z).sum()))
    return np.array(out)


# ---------- batched torch implementation: two independent 2-qubit blocks ----------
def circuit4_unitary_2q(w):
    """Build the 4x4 unitary for one Circuit-4 block, differentiable in w."""
    def ry(t):
        c, s = torch.cos(t / 2), torch.sin(t / 2)
        return torch.stack([torch.stack([c, -s]), torch.stack([s, c])]).to(torch.complex128)

    def kron(A, B):
        return torch.einsum("ij,kl->ikjl", A, B).reshape(4, 4)

    Id = torch.eye(2, dtype=torch.complex128)

    def crz(t, ctrl_first):
        # basis order (q_a q_b); ctrl_first=True -> ctrl=q_a, tgt=q_b
        ph_m = torch.exp(-1j * t.to(torch.complex128) / 2)
        ph_p = torch.exp(1j * t.to(torch.complex128) / 2)
        one = torch.ones((), dtype=torch.complex128)
        if ctrl_first:   # ctrl = a (idx bit 1), tgt = b (idx bit 0)
            d = torch.stack([one, one, ph_m, ph_p])
        else:            # ctrl = b, tgt = a  -> |ab>: b=1 rows are idx 1,3
            d = torch.stack([one, ph_m, one, ph_p])
        return torch.diag(d)

    U = torch.eye(4, dtype=torch.complex128)
    U = kron(ry(w[0]), Id) @ U
    U = kron(Id, ry(w[1])) @ U
    U = crz(w[2], ctrl_first=False) @ U
    U = kron(ry(w[3]), Id) @ U
    U = kron(Id, ry(w[4])) @ U
    U = crz(w[5], ctrl_first=True) @ U
    return U


def torch_expvals(xb, w):
    """xb: (B,4) patch inputs, w: (6,). Returns (B,4) PauliZ expvals.
    Splits into two independent 2-qubit blocks -> (2B,4) statevectors."""
    B = xb.shape[0]
    ang = xb * np.pi
    c, s = torch.cos(ang / 2), torch.sin(ang / 2)          # (B,4)
    # per-wire single-qubit amplitudes: [c, s]
    pairs = torch.stack([ang[:, 0:2], ang[:, 2:4]], dim=1).reshape(2 * B, 2)
    ca, sa = torch.cos(pairs[:, 0] / 2), torch.sin(pairs[:, 0] / 2)
    cb, sb = torch.cos(pairs[:, 1] / 2), torch.sin(pairs[:, 1] / 2)
    psi = torch.stack([ca * cb, ca * sb, sa * cb, sa * sb], dim=-1).to(torch.complex128)  # (2B,4)
    U = circuit4_unitary_2q(w)
    psi = psi @ U.transpose(0, 1)                            # (2B,4)
    p = (psi.conj() * psi).real                              # (2B,4)
    za = torch.tensor([1.0, 1.0, -1.0, -1.0])                # Z on first qubit of block
    zb = torch.tensor([1.0, -1.0, 1.0, -1.0])                # Z on second qubit
    ea, eb = p @ za, p @ zb                                  # (2B,)
    ea, eb = ea.reshape(B, 2), eb.reshape(B, 2)
    return torch.stack([ea[:, 0], eb[:, 0], ea[:, 1], eb[:, 1]], dim=-1)  # (B,4)


# ---------- checks ----------
rng = np.random.RandomState(0)
X = rng.rand(64, 4)
W = rng.randn(6)

ref = np.stack([ref_expvals(x, W) for x in X])
got = torch_expvals(torch.tensor(X), torch.tensor(W)).numpy()
print("max |torch - 4q reference| =", np.abs(ref - got).max())

# gradient check: autograd vs parameter-shift on the summed output
wt = torch.tensor(W, requires_grad=True)
loss = torch_expvals(torch.tensor(X), wt).sum()
loss.backward()
g_auto = wt.grad.numpy().copy()

g_ps = np.zeros(6)
for k in range(6):
    for sgn in (+1, -1):
        Wp = W.copy()
        Wp[k] += sgn * np.pi / 2
        val = np.stack([ref_expvals(x, Wp) for x in X]).sum()
        g_ps[k] += sgn * 0.5 * val
print("max |autograd - parameter-shift| =", np.abs(g_auto - g_ps).max())
print("autograd :", np.round(g_auto, 8))
print("param-shift:", np.round(g_ps, 8))

# CRZ has a 4-eigenvalue generator -> needs the four-term shift rule
c1 = (np.sqrt(2) + 1) / (4 * np.sqrt(2)); c2 = (np.sqrt(2) - 1) / (4 * np.sqrt(2))
def f(Wp): return np.stack([ref_expvals(x, Wp) for x in X]).sum()
for k in (2, 5):
    g = 0.0
    for sh, co in [(np.pi/2, c1), (-np.pi/2, -c1), (3*np.pi/2, -c2), (-3*np.pi/2, c2)]:
        Wp = W.copy(); Wp[k] += sh; g += co * f(Wp)
    print(f"4-term param-shift w[{k}] = {g:.8f}   autograd = {g_auto[k]:.8f}")
