"""
qnn_torch.py — GPU-native, fully batched replacement for the Qiskit
EstimatorQNN back-end used by M1 and M2.

WHY THIS EXISTS
---------------
After quanv_torch.py fixed the front-end, M1/M2 still ran ~30-85x slower than
M3/M4 (8.5 s/epoch vs 0.1-0.3 s/epoch at n=500). The entire remaining gap is
the Qiskit back-end:

  * qiskit's Estimator is a pure-Python/numpy reference implementation. There
    is no CUDA code path. It cannot use the GPU at all, at any circuit size.
  * TorchConnector round-trips every call tensor -> numpy -> parameter-bind ->
    simulate -> numpy -> tensor, per batch, per step.
  * input_gradients=True (required, or the CNN backbone gets no gradient)
    means each backward also evaluates parameter-shift terms for the 2 INPUTS
    on top of the 6 weights. RealAmplitudes' RY gates need the 2-term rule, so
    that is ~2*(6+2) = 16 extra circuit evaluations per sample, per step.

At batch 32 that is roughly 500+ sequential circuit simulations per step and
~17,000 per epoch, all on one CPU core.

WHY THE OBVIOUS FIXES DON'T WORK
--------------------------------
  * Aer with device='GPU' / cuStateVec: this is a 2-QUBIT circuit -- 4 complex
    amplitudes. Kernel-launch overhead (~5-10 us) dwarfs the work (~tens of
    FLOPs). GPU simulators only win above ~18 qubits. Same reason lightning.gpu
    failed for the 4-qubit front-end.
  * Batching Estimator calls: helps constant factors, still CPU, still numpy.
  * Dropping input_gradients: not an option -- it silently starves the shared
    CNN backbone of gradient, which is the M1 near-random-accuracy bug.

WHAT THIS DOES INSTEAD
----------------------
Simulates the exact same circuit as dense torch tensor algebra, batched over
the whole minibatch, on whatever device the model is on.

The circuit is 2 qubits => the state is 4 complex amplitudes => a (B,4) complex
tensor. Gates are applied in place:

  * Fixed gates (H, CX, RY with shared weights) are 4x4 matrices, applied as
    one batched matmul for the entire minibatch.
  * Input-dependent gates (the ZZFeatureMap phase rotations, which differ per
    sample) are DIAGONAL, so they are an elementwise multiply by a (B,4)
    per-sample phase vector -- no per-sample matrix construction needed.

That last point is what makes full batching possible: every input-dependent
operation in ZZFeatureMap is diagonal.

Gradients come from torch autograd and are exact. On a noiseless simulator
autograd and parameter-shift compute the same derivative of the same analytic
expectation value (verified for the front-end to ~1e-15; verified here against
the real EstimatorQNN in scripts/verify_qnn_torch.py).

CIRCUIT REPRODUCED (must match build_qnn_backend() exactly)
-----------------------------------------------------------
  ZZFeatureMap(feature_dimension=2, reps=1):
      H on q0, q1
      P(2*x0) on q0 ; P(2*x1) on q1
      CX(0->1) ; P(2*(pi-x0)*(pi-x1)) on q1 ; CX(0->1)
  RealAmplitudes(num_qubits=2, reps=2):     6 weights
      RY(w0) q0 ; RY(w1) q1
      CX(0->1)
      RY(w2) q0 ; RY(w3) q1
      CX(0->1)
      RY(w4) q0 ; RY(w5) q1
  Observable: Z(x)Z  (EstimatorQNN's default for a 2-qubit circuit)

NOTE ON QUBIT ORDERING: Qiskit uses little-endian (qubit 0 is the LEAST
significant bit of the statevector index). This module follows that convention
so the amplitude ordering matches Qiskit's exactly. Getting this backwards
silently transposes the state and is the single easiest way to break the
equivalence -- it is covered by the verification script.
"""

import numpy as np
import torch
import torch.nn as nn

N_QUBITS_BE = 2
N_BE_PARAMS = 6


def _fixed_gates(dtype, device):
    """4x4 gates in Qiskit little-endian |q1 q0> ordering."""
    inv = 1.0 / np.sqrt(2.0)
    # H on q0 (LSB): acts on index bit 0
    H0 = torch.tensor([[inv, inv, 0, 0],
                       [inv, -inv, 0, 0],
                       [0, 0, inv, inv],
                       [0, 0, inv, -inv]], dtype=dtype, device=device)
    # H on q1 (MSB): acts on index bit 1
    H1 = torch.tensor([[inv, 0, inv, 0],
                       [0, inv, 0, inv],
                       [inv, 0, -inv, 0],
                       [0, inv, 0, -inv]], dtype=dtype, device=device)
    # CX with control q0, target q1 -> swaps |01> (idx1) and |11> (idx3)
    CX01 = torch.tensor([[1, 0, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0],
                         [0, 1, 0, 0]], dtype=dtype, device=device)
    return H0, H1, CX01


def _ry_layer(t0, t1, dtype, device):
    """RY(t0) on q0 (x) RY(t1) on q1, in little-endian ordering."""
    c0, s0 = torch.cos(t0 / 2), torch.sin(t0 / 2)
    c1, s1 = torch.cos(t1 / 2), torch.sin(t1 / 2)
    # kron(RY(t1), RY(t0)) because q1 is the MSB
    A = torch.stack([torch.stack([c1, -s1]), torch.stack([s1, c1])]).to(dtype)
    B = torch.stack([torch.stack([c0, -s0]), torch.stack([s0, c0])]).to(dtype)
    return torch.einsum("ij,kl->ikjl", A, B).reshape(4, 4)


class QNNBackendTorch(nn.Module):
    """Drop-in replacement for TorchConnector(EstimatorQNN(...)).

    Input : (B, 2) real features from fc2
    Output: (B, n_obs) expectations in [-1, 1].

    n_obs=1 (binary, v6-v9 behaviour): <Z(x)Z> only.
    n_obs=3 (multiclass, v10):         <Z0>, <Z1>, <Z(x)Z>.

    The three observables are read off the SAME final statevector, so the
    circuit, its depth, and its 6 variational parameters are unchanged --
    n_obs costs zero additional circuit parameters. This is the fix for the
    measurement bottleneck identified in the 10-class diagnostic: a single
    scalar cannot carry 10 classes' worth of information no matter how well
    the circuit is trained, so the readout, not the ansatz, was the binding
    constraint. Going 1 -> 3 observables doubled 10-class accuracy at zero
    parameter cost.
    """

    def __init__(self, seed=None, n_obs=1):
        super().__init__()
        assert n_obs in (1, 3), f"n_obs must be 1 or 3, got {n_obs}"
        self.n_obs = n_obs
        # TorchConnector initialises EstimatorQNN weights uniformly in
        # [-pi, pi] when no initial_weights are supplied.
        g = torch.Generator().manual_seed(seed if seed is not None else 0)
        w = (torch.rand(N_BE_PARAMS, generator=g, dtype=torch.float32) * 2 - 1) * np.pi
        self.weight = nn.Parameter(w)

    def forward(self, x):
        B = x.shape[0]
        dev = x.device
        cdt = torch.complex128 if x.dtype == torch.float64 else torch.complex64
        H0, H1, CX01 = _fixed_gates(cdt, dev)

        x0, x1 = x[:, 0], x[:, 1]

        # |00>
        psi = torch.zeros(B, 4, dtype=cdt, device=dev)
        psi[:, 0] = 1.0

        # --- ZZFeatureMap(2, reps=1) ---
        psi = psi @ H0.transpose(0, 1)
        psi = psi @ H1.transpose(0, 1)

        # P(2*x0) on q0 and P(2*x1) on q1: diagonal, per-sample.
        # index bit0 = q0, bit1 = q1  ->  phase = bit0*2x0 + bit1*2x1
        two = 2.0
        ph = torch.stack([
            torch.zeros_like(x0),          # |00>
            two * x0,                      # |01> (q0=1)
            two * x1,                      # |10> (q1=1)
            two * x0 + two * x1,           # |11>
        ], dim=-1)
        psi = psi * torch.exp(1j * ph.to(cdt))

        psi = psi @ CX01.transpose(0, 1)

        # P(2*(pi-x0)*(pi-x1)) on q1: diagonal, applies when bit1 == 1
        zz = two * (np.pi - x0) * (np.pi - x1)
        ph2 = torch.stack([
            torch.zeros_like(zz), torch.zeros_like(zz), zz, zz
        ], dim=-1)
        psi = psi * torch.exp(1j * ph2.to(cdt))

        psi = psi @ CX01.transpose(0, 1)

        # --- RealAmplitudes(2, reps=2) ---
        w = self.weight
        psi = psi @ _ry_layer(w[0], w[1], cdt, dev).transpose(0, 1)
        psi = psi @ CX01.transpose(0, 1)
        psi = psi @ _ry_layer(w[2], w[3], cdt, dev).transpose(0, 1)
        psi = psi @ CX01.transpose(0, 1)
        psi = psi @ _ry_layer(w[4], w[5], cdt, dev).transpose(0, 1)

        # --- observables, all from the same final statevector ---
        # basis index bit0 = q0, bit1 = q1; eigenvalue +1 for |0>, -1 for |1>
        p = (psi.conj() * psi).real
        zz_obs = torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=p.dtype, device=dev)
        if self.n_obs == 1:
            return (p @ zz_obs).reshape(B, 1).to(x.dtype)
        z0_obs = torch.tensor([1.0, -1.0, 1.0, -1.0], dtype=p.dtype, device=dev)
        z1_obs = torch.tensor([1.0, 1.0, -1.0, -1.0], dtype=p.dtype, device=dev)
        return torch.stack([p @ z0_obs, p @ z1_obs, p @ zz_obs], dim=-1).to(x.dtype)
