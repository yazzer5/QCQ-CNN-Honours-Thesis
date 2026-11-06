#!/usr/bin/env python3
"""Offline verification of the v10 multiclass changes.

Runs WITHOUT the MNIST data and WITHOUT a GPU. It checks the things that can
be checked without training: parameter counts against the iso-parametric
budget, forward/backward shape and gradient flow for all four models, the
three-observable readout's mathematical properties, and that the binary path
is untouched.

It does NOT check that 10-class training converges, that accuracy improves,
or anything about the data pipeline. Those need a real smoke run on Gadi.

    python scripts/verify_v10_multiclass.py
"""
import sys, os, math
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_gadi as T
from qnn_torch import QNNBackendTorch

FAIL = []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond: FAIL.append(name)

def configure(n_classes, n_obs, head="linear"):
    T.N_CLASSES, T.N_OBS, T.HEAD_VARIANT = n_classes, n_obs, head

print("=" * 70)
print("1. BINARY PATH UNCHANGED  (regression guard for v6-v9 results)")
print("=" * 70)
configure(2, 1)
for m, exp in [("M1", 1604), ("M2", 1610), ("M3", 1600), ("M4", 1594)]:
    got = T.expected_params(m, 2, 1)
    check(f"{m} budget still {exp}", got == exp, f"got {got}")
    mdl = T.build_model(m, seed=42)
    tot, _ = mdl.total_params()
    check(f"{m} built model has {exp} params", tot == exp, f"got {tot}")

print()
print("=" * 70)
print("2. TEN-CLASS ISO-PARAMETRIC BUDGET")
print("=" * 70)
configure(10, 3)
tot = {}
for m in ["M1", "M2", "M3", "M4"]:
    mdl = T.build_model(m, seed=42)
    tot[m], (fe, bb, be) = mdl.total_params()
    print(f"        {m}: front={fe:>3} backbone={bb} back_end={be:>3} total={tot[m]}")
lo, hi = min(tot.values()), max(tot.values())
spread = (hi - lo) / lo * 100
check("all four within 1% of each other", spread < 1.0, f"spread {spread:.2f}% ({lo}-{hi})")
check("M2-M1 == 6 (front-end params only)", tot["M2"] - tot["M1"] == 6)
check("M3-M4 == 6 (front-end params only)", tot["M3"] - tot["M4"] == 6)

print()
print("=" * 70)
print("3. FORWARD / BACKWARD, ALL FOUR MODELS AT 10 CLASSES")
print("=" * 70)
x = torch.rand(8, 1, 28, 28)
y = torch.randint(0, 10, (8,))
for m in ["M1", "M2", "M3", "M4"]:
    mdl = T.build_model(m, seed=42)
    out = mdl(x)
    check(f"{m} output shape (8,10)", tuple(out.shape) == (8, 10), str(tuple(out.shape)))
    loss = torch.nn.functional.binary_cross_entropy(
        torch.sigmoid(out), torch.nn.functional.one_hot(y, 10).float())
    mdl.zero_grad(); loss.backward()
    dead = [n for n, p in mdl.named_parameters() if p.requires_grad and p.grad is None]
    check(f"{m} every trainable tensor got a gradient", not dead, f"dead: {dead}")
    if m in ("M2", "M3"):
        g = mdl.front_end.weights.grad
        check(f"{m} front-end gradient is nonzero", g is not None and g.abs().sum() > 0)

print()
print("=" * 70)
print("4. THREE-OBSERVABLE READOUT")
print("=" * 70)
q1 = QNNBackendTorch(seed=42, n_obs=1)
q3 = QNNBackendTorch(seed=42, n_obs=3)
q3.weight.data = q1.weight.data.clone()
xin = torch.randn(64, 2)
o1, o3 = q1(xin), q3(xin)
check("n_obs=1 shape (B,1)", tuple(o1.shape) == (64, 1))
check("n_obs=3 shape (B,3)", tuple(o3.shape) == (64, 3))
check("<ZZ> identical between n_obs=1 and n_obs=3",
      torch.allclose(o1[:, 0], o3[:, 2], atol=1e-6),
      f"max diff {(o1[:,0]-o3[:,2]).abs().max():.2e}")
check("all expectations within [-1,1]", bool((o3.abs() <= 1.0 + 1e-6).all()),
      f"max |val| {o3.abs().max():.6f}")
# <Z0><Z1> != <Z0 Z1> in general -> the third observable carries information
# the first two do not; if it were separable the readout would be rank 2.
prod = o3[:, 0] * o3[:, 1]
check("<Z0><Z1> != <ZZ> (third observable is not redundant)",
      (prod - o3[:, 2]).abs().max() > 1e-3,
      f"max |<Z0><Z1> - <ZZ>| = {(prod-o3[:,2]).abs().max():.4f}")
check("n_obs=3 costs zero extra circuit parameters",
      q1.weight.numel() == q3.weight.numel() == 6)
check("observables span >1 dimension (rank of output)",
      torch.linalg.matrix_rank(o3 - o3.mean(0)).item() == 3,
      f"rank {torch.linalg.matrix_rank(o3 - o3.mean(0)).item()}")

print()
print("=" * 70)
print("5. HEAD VARIANT 'tanh' ALSO BUILDS AND MATCHES BUDGET")
print("=" * 70)
configure(10, 3, "tanh")
for m in ["M3", "M4"]:
    mdl = T.build_model(m, seed=42)
    t, _ = mdl.total_params()
    check(f"{m} tanh variant param count unchanged", t == tot[m], f"{t} vs {tot[m]}")
    check(f"{m} tanh variant forward OK", tuple(mdl(x).shape) == (8, 10))

print()
print("=" * 70)
print(f"{'ALL CHECKS PASSED' if not FAIL else 'FAILURES: ' + ', '.join(FAIL)}")
print("=" * 70)
print("\nNOT verified here (needs a real smoke run on Gadi):")
print("  - that 10-class training converges at all")
print("  - that M2 > M1 reproduces the +6.4 pp diagnostic result")
print("  - data loading, class balance, and epoch timing at 10 classes")
sys.exit(1 if FAIL else 0)
