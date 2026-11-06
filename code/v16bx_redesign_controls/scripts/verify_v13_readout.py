#!/usr/bin/env python3
"""
v13 offline verification. Run from the repo root AFTER patch_v13_readout.py,
on the login node (CPU, ~1 min, no training):

    python scripts/verify_v13_readout.py              # full: QNN + train_gadi models
    python scripts/verify_v13_readout.py --qnn_only   # QNN checks only

Must end in ALL CHECKS PASSED before any v13 job is submitted.

Section 1 proves the n_obs=1 and n_obs=3 outputs are BIT-IDENTICAL to the
pre-patch code (qnn_torch.py.v12bak), so v10/v12 results stay valid and can
be pooled with v13 as the width-3 arm.
Sections 2-3 check the new readouts are physically correct and that the
structural claims in PREREGISTRATION_v13.md hold for the real circuit.
Section 4 builds every v13 model and checks the parameter budget.
"""
import argparse
import importlib.util
import inspect
import os
import sys
import textwrap

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
ap = argparse.ArgumentParser()
ap.add_argument("--qnn_only", action="store_true")
args = ap.parse_args()

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


def header(s):
    print("\n" + "=" * 72 + "\n" + s + "\n" + "=" * 72)


import qnn_torch as Q
QNN = Q.QNNBackendTorch
bak = os.path.join(ROOT, "qnn_torch.py.v12bak")
if not os.path.exists(bak):
    sys.exit("qnn_torch.py.v12bak not found: run patch_v13_readout.py first (setup_v13.sh does).")
from importlib.machinery import SourceFileLoader
_ld = SourceFileLoader("qnn_torch_v12bak", bak)
spec = importlib.util.spec_from_loader("qnn_torch_v12bak", _ld)
QB = importlib.util.module_from_spec(spec)
_ld.exec_module(QB)

torch.manual_seed(0)
X32 = torch.randn(512, 2) * 1.5
X64 = X32.double()


def pair(cls_new, cls_old, n_obs, seed):
    a, b = cls_new(seed=seed, n_obs=n_obs), cls_old(seed=seed, n_obs=n_obs)
    b.load_state_dict(a.state_dict())
    return a, b


# ---------------------------------------------------------------------------
header("1. n_obs = 1 and 3 UNCHANGED vs pre-patch code (bit-exact)")
for n_obs in (1, 3):
    for seed in (42, 47):
        a, b = pair(QNN, QB.QNNBackendTorch, n_obs, seed)
        for X, tag in ((X32, "float32"), (X64, "float64")):
            oa, ob = a.to(X.dtype)(X), b.to(X.dtype)(X)
            check(f"n_obs={n_obs} seed={seed} {tag}: torch.equal(new, old)", torch.equal(oa, ob),
                  f"max diff {(oa - ob).abs().max():.1e}")
        a32, b32 = pair(QNN, QB.QNNBackendTorch, n_obs, seed)
        Xg = X32.clone().requires_grad_(True); Xh = X32.clone().requires_grad_(True)
        a32(Xg).sum().backward(); b32(Xh).sum().backward()
        check(f"n_obs={n_obs} seed={seed}: gradients bit-identical",
              torch.equal(a32.weight.grad, b32.weight.grad) and torch.equal(Xg.grad, Xh.grad))
if torch.cuda.is_available():
    a, b = pair(QNN, QB.QNNBackendTorch, 3, 42)
    Xc = X32.cuda()
    check("n_obs=3 on CUDA: torch.equal(new, old)", torch.equal(a.cuda()(Xc), b.cuda()(Xc)))
else:
    print("  (no CUDA on this node: GPU path is covered by the reproduction smoke job)")

# ---------------------------------------------------------------------------
header("2. NEW READOUTS: shapes, ranges, nesting, physics")
outs = {}
for n_obs in (3, 6, 15):
    q = QNN(seed=42, n_obs=n_obs).double()
    if n_obs != 3:
        q.load_state_dict(QNN(seed=42, n_obs=3).double().state_dict())
    outs[n_obs] = q(X64)
    check(f"n_obs={n_obs}: shape (B,{n_obs})", tuple(outs[n_obs].shape) == (512, n_obs))
    check(f"n_obs={n_obs}: all expectations in [-1,1]", bool((outs[n_obs].abs() <= 1 + 1e-9).all()))
    check(f"n_obs={n_obs}: 6 circuit parameters (readout adds none)", q.weight.numel() == 6)
check("w6 first 3 columns == w3 exactly", torch.equal(outs[6][:, :3], outs[3]))
check("w15 first 6 columns == w6 exactly", torch.equal(outs[15][:, :6], outs[6]))
s = outs[15].pow(2).sum(1)
check("sum of all 15 Pauli expectations^2 == 3 (pure 2-qubit state)",
      torch.allclose(s, torch.full_like(s, 3.0), atol=1e-9),
      f"range [{s.min():.12f}, {s.max():.12f}]")
check("names table matches widths", all(len(Q.V13_READOUT_NAMES[w]) == w for w in (1, 3, 6, 15)))
q32 = QNN(seed=42, n_obs=15)
Xg = X32.clone().requires_grad_(True)
o = q32(Xg); o.sum().backward()
check("n_obs=15 float32 forward dtype float32", o.dtype == torch.float32)
check("n_obs=15: gradient reaches circuit weights and inputs",
      q32.weight.grad is not None and q32.weight.grad.abs().sum() > 0 and Xg.grad.abs().sum() > 0)

# ---------------------------------------------------------------------------
header("3. STRUCTURAL CLAIMS IN THE PRE-REGISTRATION (real circuit)")


def rank(M):
    M = M - M.mean(0)
    sv = torch.linalg.svdvals(M)
    return int((sv > 1e-8 * sv[0]).sum())


Xr = torch.randn(4000, 2, dtype=torch.float64) * 1.5
R = {w: rank(QNN(seed=42, n_obs=w).double()(Xr)) for w in (3, 6, 15)}
check("w3 readout has 3 independent features", R[3] == 3, str(R[3]))
check("w6 readout has 6 independent features (all cosine terms)", R[6] == 6, str(R[6]))
check("w15 readout has 12 independent features (6 cosine + 6 sine)", R[15] == 12, str(R[15]))


def span_overlap(w, s1, s2, k):
    A = QNN(seed=s1, n_obs=w).double()(Xr); B = QNN(seed=s2, n_obs=w).double()(Xr)
    ua = torch.linalg.svd(A - A.mean(0), full_matrices=False)[0][:, :k]
    ub = torch.linalg.svd(B - B.mean(0), full_matrices=False)[0][:, :k]
    return torch.linalg.svdvals(ua.T @ ub).min().item()


o3, o6, o15 = span_overlap(3, 1, 7, 3), span_overlap(6, 1, 7, 6), span_overlap(15, 1, 7, 12)
check("w3: circuit weights CHANGE which features are visible", o3 < 0.99, f"overlap {o3:.4f}")
check("w6: feature span independent of circuit weights", o6 > 1 - 1e-6, f"overlap {o6:.6f}")
check("w15: feature span independent of circuit weights", o15 > 1 - 1e-6, f"overlap {o15:.6f}")

# ---------------------------------------------------------------------------
if args.qnn_only:
    print("\n(--qnn_only: skipping train_gadi model checks)")
else:
    header("4. train_gadi MODELS: build, budget, forward/backward at every v13 cell")
    import train_gadi as T
    expected_w3 = {(5, "M2"): 1626, (5, "M3"): 1629, (10, "M2"): 1646, (10, "M3"): 1649}
    tab = []
    for K in (5, 10):
        for w in (3, 6, 15):
            T.N_CLASSES, T.N_OBS, T.HEAD_VARIANT = K, w, "linear"
            for m in ("M1", "M2", "M3", "M4"):
                mdl = T.build_model(m, seed=42)
                tot, (fe, bb, be) = mdl.total_params()
                exp = T.expected_params(m, K, w)
                check(f"K={K} w={w} {m}: built {tot} == expected_params {exp}", tot == exp)
                if w == 3 and (K, m) in expected_w3:
                    check(f"K={K} w=3 {m}: matches v10/v12 VERIFIED count {expected_w3[(K, m)]}",
                          tot == expected_w3[(K, m)])
                x = torch.rand(8, 1, 28, 28); y = torch.randint(0, K, (8,))
                out = mdl(x)
                check(f"K={K} w={w} {m}: output (8,{K})", tuple(out.shape) == (8, K))
                mdl.zero_grad(); torch.nn.functional.cross_entropy(out, y).backward()
                dead = [n for n, p in mdl.named_parameters() if p.requires_grad and p.grad is None]
                check(f"K={K} w={w} {m}: every trainable tensor has a gradient", not dead, str(dead))
                tab.append((K, w, m, fe, bb, be, tot))
    print("\n  Parameter table (record in MNIST_LOG.md):")
    print(f"  {'K':>3} {'w':>3} {'model':>5} {'front':>6} {'backbone':>9} {'back':>5} {'total':>6}")
    for r in tab:
        print(f"  {r[0]:>3} {r[1]:>3} {r[2]:>5} {r[3]:>6} {r[4]:>9} {r[5]:>5} {r[6]:>6}")
    for K in (5, 10):
        for w in (6, 15):
            d = {r[2]: r[6] for r in tab if r[0] == K and r[1] == w}
            print(f"  K={K} w={w}: M3-M2 = {d['M3'] - d['M2']:+d} params "
                  f"({(d['M3'] - d['M2']) / d['M2'] * 100:+.2f}%)  [expected and pre-registered: "
                  f"the classical 2->w map has 3w params vs the circuit's 6]")

header("ALL CHECKS PASSED" if not FAIL else f"{len(FAIL)} FAILURE(S): " + "; ".join(FAIL))
sys.exit(1 if FAIL else 0)
