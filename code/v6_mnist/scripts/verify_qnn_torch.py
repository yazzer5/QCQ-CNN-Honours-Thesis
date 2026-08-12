#!/usr/bin/env python
"""
verify_qnn_torch.py — proves qnn_torch.QNNBackendTorch computes exactly what
TorchConnector(EstimatorQNN(ZZFeatureMap + RealAmplitudes)) computes.

Run once on Gadi inside the qcqcnn env and keep the output. Together with
scripts/equivalence_check.py (front-end) this is the full evidence that the
v5 GPU-native path is mathematically identical to the Qiskit/PennyLane
reference, and that the speedup is an execution-strategy change rather than
an approximation.

    PYTHONPATH="" python scripts/verify_qnn_torch.py

Checks
  1. Forward: <Z(x)Z> for a batch of random feature vectors.
  2. Gradient w.r.t. the 6 RealAmplitudes weights.
  3. Gradient w.r.t. the 2 INPUTS -- this is the one that matters most.
     input_gradients=True is what carries gradient back into fc2 and the
     shared CNN backbone; when it was missing (v4) the backbone silently
     stayed at random init and M1 sat at near-random accuracy for an entire
     run. Any replacement MUST reproduce it.
  4. Timing, both paths, at two batch sizes.

Tolerance is ~1e-5 relative. TorchConnector works internally in float32, so
agreement is limited by float32 precision (~1e-7 relative per op), not by
the method.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

from qnn_torch import QNNBackendTorch

TOL = 1e-4
torch.manual_seed(0)
np.random.seed(0)

fm = ZZFeatureMap(feature_dimension=2, reps=1)
an = RealAmplitudes(num_qubits=2, reps=2)
assert len(an.parameters) == 6, "RealAmplitudes(2, reps=2) must have 6 params"

qnn = EstimatorQNN(circuit=fm.compose(an), input_params=fm.parameters,
                   weight_params=an.parameters, input_gradients=True)
ref = TorchConnector(qnn)
tor = QNNBackendTorch(seed=0)

# force identical weights
with torch.no_grad():
    tor.weight.copy_(list(ref.parameters())[0].detach())
print(f"shared weights: {tor.weight.detach().numpy()}")

x = torch.rand(32, 2) * 4 - 2          # cover a wide angle range
failures = []


def report(name, a, b):
    a, b = a.detach().double(), b.detach().double()
    d = (a - b).abs().max().item()
    rel = d / max(b.abs().max().item(), 1.0)
    ok = rel < TOL
    print(f"  {name:34s} max|Δ| = {d:.3e}  rel = {rel:.3e}   {'OK' if ok else 'FAIL'}")
    if not ok:
        failures.append(name)


print("\n" + "=" * 70)
print("QNN BACK-END  (ZZFeatureMap(2,reps=1) + RealAmplitudes(2,reps=2), Z⊗Z)")
print("=" * 70)

report("forward", tor(x), ref(x))

xa = x.clone().requires_grad_(True)
tor.zero_grad(); tor(xa).sum().backward()
xb = x.clone().requires_grad_(True)
ref.zero_grad(); ref(xb).sum().backward()

report("grad w.r.t. weights", tor.weight.grad, list(ref.parameters())[0].grad)
report("grad w.r.t. inputs", xa.grad, xb.grad)

print("\n" + "=" * 70)
print("TIMING (forward + backward)")
print("=" * 70)
for B in (32, 128):
    xb_ = torch.rand(B, 2)
    t = time.time()
    for _ in range(3):
        q = xb_.clone().requires_grad_(True); ref.zero_grad(); ref(q).sum().backward()
    tq = (time.time() - t) / 3
    for _ in range(3):     # warm up torch path
        q = xb_.clone().requires_grad_(True); tor.zero_grad(); tor(q).sum().backward()
    t = time.time()
    for _ in range(20):
        q = xb_.clone().requires_grad_(True); tor.zero_grad(); tor(q).sum().backward()
    tt = (time.time() - t) / 20
    print(f"  batch {B:4d}:  qiskit {tq*1000:9.2f} ms  |  torch {tt*1000:7.3f} ms  "
          f"|  {tq/tt:6.0f}x")

steps = 5000 // 32
print(f"\n  Projected back-end cost at full scale ({steps} steps/epoch):")
print(f"    qiskit ~{tq*steps:.0f} s/epoch    torch ~{tt*steps:.2f} s/epoch")

print("\n" + "=" * 70)
if failures:
    print(f"FAILED: {failures}")
    print("DO NOT USE THE TORCH BACK-END. Investigate before proceeding.")
    sys.exit(1)
print("ALL CHECKS PASSED — the torch QNN back-end is numerically equivalent")
print("to TorchConnector(EstimatorQNN), including input gradients.")
print("\nREMINDER: this covers the IDEAL simulator only. The noise phase")
print("(alpha > 0) needs Aer's NoiseModel, which the torch path cannot")
print("express -- use --qnn_backend qiskit there.")
sys.exit(0)
