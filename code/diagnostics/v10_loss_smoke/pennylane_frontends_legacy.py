"""
pennylane_frontends_legacy.py — the ORIGINAL PennyLane/TorchLayer front-ends.

These are kept for exactly one purpose: proving that `quanv_torch.py` computes
the same thing. They are NOT used in production runs and must not be.

Why they were retired (see train_gadi.py's GPU ROOT CAUSE docstring):
  * 4-qubit circuits cannot occupy a V100 -- launch overhead dominates.
  * TorchLayer re-serialises a batched input into B*196 sequential tape
    executions, because the lightning devices don't support parameter
    broadcasting. The v4 unfold() "batching fix" therefore did nothing.

Reached via `--backend pennylane` in train_gadi.py, or directly by
scripts/equivalence_check.py.
"""

import numpy as np
import torch
import torch.nn as nn

import pennylane as qml
from pennylane import numpy as qnp
from pennylane.qnn import TorchLayer

N_QUBITS_FE = 4
N_C4_PARAMS = 6

_QUANTUM_DEVICE_NAME = None


def get_quantum_device(n_wires):
    """NOTE: lightning.gpu is deliberately NOT tried here. It is slower than
    lightning.qubit below ~18 qubits -- the GPU sits idle between kernel
    launches. This path exists for numerical reference, so use the fastest
    CPU simulator available."""
    global _QUANTUM_DEVICE_NAME
    dev = qml.device("lightning.qubit", wires=n_wires)
    if _QUANTUM_DEVICE_NAME is None:
        _QUANTUM_DEVICE_NAME = "lightning.qubit"
        print(f"[DEVICE][legacy] reference circuits on lightning.qubit "
              f"({n_wires} wires)", flush=True)
    return dev


def circuit4_block(weights, wire_a, wire_b):
    """Hur et al. (2022) Circuit 4. 6 params. CRZ entanglers (NOT CNOT/CZ).
    The two middle single-qubit gates are RY, NOT RZ (verified 2026-07-18
    against Hur et al. Fig. 2(d))."""
    qml.RY(weights[0], wires=wire_a)
    qml.RY(weights[1], wires=wire_b)
    qml.CRZ(weights[2], wires=[wire_b, wire_a])   # ctrl=wire_b, tgt=wire_a
    qml.RY(weights[3], wires=wire_a)
    qml.RY(weights[4], wires=wire_b)
    qml.CRZ(weights[5], wires=[wire_a, wire_b])   # ctrl=wire_a, tgt=wire_b


def build_trainable_quanv_qnode(dev):
    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
    def qnode(inputs, weights):
        for i in range(N_QUBITS_FE):
            qml.RY(inputs[..., i] * np.pi, wires=i)
        circuit4_block(weights, 0, 1)
        circuit4_block(weights, 2, 3)
        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS_FE)]
    return qnode


def build_frozen_henderson_qnode(dev, rand_params):
    @qml.qnode(dev, interface="torch", diff_method=None)
    def qnode(inputs):
        for i in range(N_QUBITS_FE):
            qml.RY(inputs[..., i] * np.pi, wires=i)
        qml.RandomLayers(rand_params, wires=list(range(N_QUBITS_FE)))
        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS_FE)]
    return qnode


def _patches(x):
    B, _, H, W = x.shape
    oh, ow = H // 2, W // 2
    p = (x.unfold(2, 2, 2).unfold(3, 2, 2)
          .permute(0, 2, 3, 1, 4, 5)
          .reshape(B * oh * ow, 4))
    return p, B, oh, ow


class TrainableQuanvFrontEndPennyLane(nn.Module):
    """M2/M3 reference front-end. Slow by construction."""

    def __init__(self):
        super().__init__()
        dev = get_quantum_device(N_QUBITS_FE)
        qnode = build_trainable_quanv_qnode(dev)
        self.qlayer = TorchLayer(qnode, {"weights": (N_C4_PARAMS,)})

    def forward(self, x):
        orig = x.device
        x = x.cpu()
        patches, B, oh, ow = _patches(x)
        res = self.qlayer(patches)
        out = res.reshape(B, oh, ow, N_QUBITS_FE).permute(0, 3, 1, 2)
        return out.to(orig)

    def n_trainable_params(self):
        return N_C4_PARAMS


class FrozenHendersonFrontEndPennyLane(nn.Module):
    """M1 reference front-end. 0 trainable params."""

    def __init__(self, seed, n_layers=1):
        super().__init__()
        dev = get_quantum_device(N_QUBITS_FE)
        g = np.random.RandomState(seed)
        rand_params = qnp.array(
            g.uniform(high=2 * np.pi, size=(n_layers, N_QUBITS_FE)),
            requires_grad=False,
        )
        self.qnode = build_frozen_henderson_qnode(dev, rand_params)

    def forward(self, x):
        orig = x.device
        x = x.cpu()
        patches, B, oh, ow = _patches(x)
        with torch.no_grad():
            raw = self.qnode(patches)
            res = torch.stack([torch.as_tensor(r) for r in raw], dim=-1)
        out = res.reshape(B, oh, ow, N_QUBITS_FE).permute(0, 3, 1, 2).to(x.dtype)
        return out.to(orig)

    def n_trainable_params(self):
        return 0
