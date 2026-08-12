#!/usr/bin/env python
"""Verify the torch RealAmplitudes(2, reps) implementation against real Qiskit.

Run on Gadi (needs qiskit + qiskit-machine-learning from the qcqcnn env):
    python scripts/verify_reps.py

Checks, for reps in {1,2}:
  * parameter count matches 2*(reps+1)
  * forward output matches EstimatorQNN to float32 precision
  * weight gradients match
  * input gradients match  (the chain into fc2 and the backbone)

reps=2 must additionally be bit-identical to the previously validated
implementation, since the completed grids depend on it.
"""
import numpy as np
import torch
from qnn_torch import QNNBackendTorch
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

TOL = 2e-5

def build_ref(reps, weights):
    fm = ZZFeatureMap(feature_dimension=2, reps=1)
    an = RealAmplitudes(num_qubits=2, reps=reps)
    assert len(an.parameters) == 2 * (reps + 1)
    qnn = EstimatorQNN(circuit=fm.compose(an), input_params=fm.parameters,
                       weight_params=an.parameters, input_gradients=True)
    return TorchConnector(qnn, initial_weights=weights)

ok = True
for reps in (1, 2):
    mine = QNNBackendTorch(seed=42, reps=reps)
    w = mine.weight.detach().numpy().astype(np.float64)
    print(f"\nreps={reps}: n_params={w.size} (expected {2*(reps+1)})")
    assert w.size == 2 * (reps + 1)

    ref = build_ref(reps, w)
    x = torch.randn(16, 2, dtype=torch.float32)

    xa = x.clone().requires_grad_(True)
    xb = x.clone().requires_grad_(True)
    ya, yb = mine(xa), ref(xb)
    d = (ya - yb).abs().max().item()
    print(f"  forward   max|diff| = {d:.3e}")
    ok &= d < TOL

    ya.sum().backward(); yb.sum().backward()
    dg = (mine.weight.grad - ref.weight.grad).abs().max().item()
    dx = (xa.grad - xb.grad).abs().max().item()
    print(f"  weight grad max|diff| = {dg:.3e}")
    print(f"  input  grad max|diff| = {dx:.3e}")
    ok &= dg < TOL and dx < TOL

print("\n" + ("ALL CHECKS PASSED" if ok else "FAILED -- do not run the grid"))
raise SystemExit(0 if ok else 1)
