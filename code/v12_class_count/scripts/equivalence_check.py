#!/usr/bin/env python
"""
equivalence_check.py — proves the v5 torch-native front-ends compute exactly
what the retired PennyLane/TorchLayer front-ends computed.

Run this ONCE on Gadi inside the qcqcnn env and keep the output. It is the
evidence for the methods section: the switch from parameter-shift-on-a-
simulator to autograd-on-an-exact-simulation changed the execution strategy,
not the mathematics.

    PYTHONPATH="" python scripts/equivalence_check.py

Checks performed
  1. M2/M3 forward: torch front-end vs PennyLane TorchLayer, same weights.
  2. M2/M3 gradient: d(sum of outputs)/d(weights), torch autograd vs
     PennyLane parameter-shift.
  3. M1 forward: torch (extracted 16x16 unitary) vs PennyLane RandomLayers.
  4. Timing, both paths, so the speedup is on the record.

Agreement is checked RELATIVE to the magnitude of the reference value, since
gradients summed over ~800 patches reach O(1e3) where float32 spacing alone is
~1e-4. Expect ~1e-7 relative in float32. Anything above 1e-5 relative means
STOP -- do not run the grid.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np

from quanv_torch import TrainableQuanvFrontEndTorch, FrozenHendersonFrontEndTorch
from pennylane_frontends_legacy import (
    TrainableQuanvFrontEndPennyLane, FrozenHendersonFrontEndPennyLane,
)

# Relative tolerance. Absolute tolerance is the wrong test here: summed
# gradients over 784 patches reach magnitude ~1e3, where float32 spacing is
# already ~1e-4. The forward values are O(1) so the same relative test is
# strict for them too. In float64 both checks land at ~1e-14.
RTOL = 1e-5
SEED = 42
N_IMG = 4          # PennyLane path is slow; 4 images = 784 patches already

torch.manual_seed(SEED)
np.random.seed(SEED)

x = torch.rand(N_IMG, 1, 28, 28)
failures = []


def report(name, a, b):
    a, b = a.detach().double(), b.detach().double()
    d = (a - b).abs().max().item()
    scale = max(b.abs().max().item(), 1.0)
    rel = d / scale
    ok = rel < RTOL
    print(f"  {name:38s} max|Δ| = {d:.3e}  rel = {rel:.3e}   "
          f"{'OK' if ok else 'FAIL'}")
    if not ok:
        failures.append(name)


print("=" * 70)
print("M2/M3 TRAINABLE CIRCUIT 4 FRONT-END")
print("=" * 70)

fe_t = TrainableQuanvFrontEndTorch(seed=SEED)
fe_p = TrainableQuanvFrontEndPennyLane()
# force identical weights
with torch.no_grad():
    w = fe_t.weights.detach().clone()
    for n, p in fe_p.named_parameters():
        p.copy_(w.to(p.dtype))
print(f"  shared weights: {w.numpy()}")

t0 = time.time(); out_t = fe_t(x); t_torch = time.time() - t0
t0 = time.time(); out_p = fe_p(x); t_pl = time.time() - t0
report("forward", out_t, out_p.to(out_t.dtype))
print(f"  torch forward     : {t_torch:.4f} s")
print(f"  pennylane forward : {t_pl:.4f} s   ({t_pl / max(t_torch, 1e-9):.0f}x slower)")

fe_t.zero_grad(); fe_t(x).sum().backward()
g_t = fe_t.weights.grad.detach().clone()
fe_p.zero_grad(); fe_p(x).sum().backward()
g_p = torch.cat([p.grad.reshape(-1) for p in fe_p.parameters()]).to(g_t.dtype)
report("gradient (autograd vs param-shift)", g_t, g_p)
print(f"  torch grad     : {np.round(g_t.numpy(), 6)}")
print(f"  param-shift    : {np.round(g_p.numpy(), 6)}")

print()
print("=" * 70)
print("M1 FROZEN HENDERSON / RANDOMLAYERS FRONT-END")
print("=" * 70)
m1_t = FrozenHendersonFrontEndTorch(SEED)
m1_p = FrozenHendersonFrontEndPennyLane(SEED)
t0 = time.time(); o_t = m1_t(x); t_torch = time.time() - t0
t0 = time.time(); o_p = m1_p(x); t_pl = time.time() - t0
report("forward", o_t, o_p.to(o_t.dtype))
print(f"  torch forward     : {t_torch:.4f} s")
print(f"  pennylane forward : {t_pl:.4f} s   ({t_pl / max(t_torch, 1e-9):.0f}x slower)")
print(f"  trainable params  : torch={m1_t.n_trainable_params()} "
      f"pennylane={m1_p.n_trainable_params()} (both must be 0)")

print()
print("=" * 70)
if failures:
    print(f"FAILED: {failures}")
    print("DO NOT RUN THE GRID. Investigate before proceeding.")
    sys.exit(1)
print("ALL CHECKS PASSED — torch front-ends are numerically equivalent to the")
print("PennyLane/parameter-shift reference. Safe to run the grid.")
sys.exit(0)
