#!/usr/bin/env python3
"""
v16b offline verification. Run from the v16b repo root AFTER patch_v16b.py, on the login node
(CPU, a few minutes), in the qcqcnn env:

    python3 -u scripts/verify_v16b.py | tee verify_v16b.log

Must end in ALL CHECKS PASSED before any v16b job is submitted.

1. UNCHANGED (no v16b flag), K=5 and K=10, w15: M1-M5, and M2c (v16a head active), initial weights,
   outputs, gradients and expected_params bit-identical to the pre-patch file (train_gadi.py.v16abak).
2. THE BFR CIRCUIT (float64): the batched torch simulator equals an independent per-sample complex
   statevector (gate by gate) and, if importable, Qiskit (RY, CX, Statevector) on all 15 Paulis;
   the 6 single-Y Paulis are identically 0; Ry(u)Ry(theta) = Ry(u + theta); every feature is a
   trigonometric polynomial of degree <= 2 in each u_i (FFT, the "bounded frequency" property);
   |u| <= |s| for any z.
3. BFR SURGERY on the real M4 (K=5, K=10): pre-registered parameter table, within +-1% of M4;
   every RNG state after building BFR == after building M4; backbone and fc2 identical to M4;
   fc_mid / fc3 are M4's init sliced (fc3 rescaled to the default bound); g = 0 => logits == the
   classical path exactly; FUNCTION CLASS: the trimmed classical path reproduces M4's logits exactly;
   gradients at g = 0 (only g among the gated tensors) and at g = 1 (all); a BFR forward/backward
   consumes no RNG; expected_params(M4) = BFR count while active, other models unchanged; fc2 is the
   unique Linear(-> 2).
4. DRY RUNS through train_gadi.py (50/class, 2 epochs, CPU, real data):
   M4 through the patched file == M4 through the pre-patch file (every prediction);
   BFR K=5 A and K=10: rc=0, HEAD line VERIFIED, trainer's parameter check OK, warmup min_epochs=0
   (= M4), gradient probe line + g restored bit-exact, GS line, RUN self-check MATCH, n_params in the
   result JSON = BFR count, the same train/val/test images as M4, deterministic (two runs identical).
"""
import copy
import importlib.util
import json
import math
import os
import random
import re
import subprocess
import sys
import tempfile
from importlib.machinery import SourceFileLoader

for k in ("V15_ALPHA", "V15_CALIB", "V15_EVAL_ALPHAS", "V14_NO_AUDIT", "V14_SPLIT_ONLY", "V14_SPLIT_DUMP"):
    os.environ.pop(k, None)

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
FAIL = []
TMP = tempfile.mkdtemp(prefix="v16bverify_")
TABLE = {5: dict(M4=1719, BFR=1720, w=5, M2=1686), 10: dict(M4=1799, BFR=1799, w=3, M2=1766)}


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""), flush=True)
    if not cond:
        FAIL.append(name)


def header(s):
    print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78, flush=True)


import train_gadi as T  # noqa: E402
import bfr_head_v16b as BH  # noqa: E402
import closed_form_zz_v16 as CF  # noqa: E402
import v16_tools as VT  # noqa: E402

bak = os.path.join(ROOT, "train_gadi.py.v16abak")
if not os.path.exists(bak):
    sys.exit("train_gadi.py.v16abak not found: run patch_v16b.py first (setup_v16b.sh does).")
_ld = SourceFileLoader("train_gadi_v16abak", bak)
TB = importlib.util.module_from_spec(importlib.util.spec_from_loader("train_gadi_v16abak", _ld))
_ld.exec_module(TB)
src = open("train_gadi.py").read()
check("v16b patch marker present, v16a block still present", "V16B_BFR_HEAD" in src and "V16A_HEAD_SAVEZ" in src)
check("v16b inactive in this process (no flag)", T._V16B["head"] is None)
check("v16a inactive in this process (no flag)", not T._v16_active())


def setcell(mod, K, w=15):
    mod.N_CLASSES, mod.N_OBS = K, w
    mod.HEAD_VARIANT = "linear"


def build(mod, m, seed=42):
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    return mod.build_model(m, seed=seed)


def sd_equal(a, b, keys=None):
    sa, sb = a.state_dict(), b.state_dict()
    ka = [k for k in sa if keys is None or any(k.startswith(p) for p in keys)]
    kb = [k for k in sb if keys is None or any(k.startswith(p) for p in keys)]
    return ka == kb and all(torch.equal(sa[k], sb[k]) for k in ka)


def xin(n=2, seed=0):
    return torch.rand(n, 1, 28, 28, generator=torch.Generator().manual_seed(seed))


def fwd_grad(model, x):
    model.zero_grad()
    out = model(x)
    out.float().pow(2).sum().backward()
    return out.detach(), {n: (p.grad.clone() if p.grad is not None else None) for n, p in model.named_parameters()}


def grads_equal(ga, gb):
    return ga.keys() == gb.keys() and all(
        (ga[k] is None and gb[k] is None) or (ga[k] is not None and gb[k] is not None and torch.equal(ga[k], gb[k]))
        for k in ga)


def rng_state():
    return (random.getstate(), np.random.get_state()[1].tobytes(), torch.get_rng_state())


def rng_equal(a, b):
    return a[0] == b[0] and a[1] == b[1] and torch.equal(a[2], b[2])


# ---------------------------------------------------------------------------
header("1. UNCHANGED vs pre-patch train_gadi.py (no v16b flag), w15, bit-exact")
for K in (5, 10):
    setcell(T, K); setcell(TB, K)
    for m in ("M1", "M2", "M3", "M4", "M5"):
        a, b = build(T, m), build(TB, m)
        x = xin()
        oa, ga = fwd_grad(a, x)
        ob, gb = fwd_grad(b, x)
        check(f"K={K} {m}: init, output and gradients identical", sd_equal(a, b) and torch.equal(oa, ob) and grads_equal(ga, gb))
        ep, epb = T.expected_params(m, K, 15), TB.expected_params(m, K, 15)
        check(f"K={K} {m}: expected_params unchanged", ep == epb, str(ep))
    # M2c (v16a head) through both files
    for mod in (T, TB):
        mod._V16["head"] = "m2c"; mod._V16["registered"] = True
    a, b = build(T, "M2"), build(TB, "M2")
    oa, ga = fwd_grad(a, xin()); ob, gb = fwd_grad(b, xin())
    check(f"K={K} M2c: init, output and gradients identical", sd_equal(a, b) and torch.equal(oa, ob) and grads_equal(ga, gb),
          f"params {VT.n_params(a)}")
    check(f"K={K} M2c: expected_params unchanged", T.expected_params("M2", K, 15) == TB.expected_params("M2", K, 15))
    for mod in (T, TB):
        mod._V16["head"] = None; mod._V16["info"] = None

# ---------------------------------------------------------------------------
header("2. THE BFR CIRCUIT (float64)")
I2 = np.eye(2); PX = np.array([[0, 1], [1, 0]]); PY = np.array([[0, -1j], [1j, 0]]); PZ = np.diag([1., -1.])
P1 = dict(I=I2, X=PX, Y=PY, Z=PZ)


def ry(a):
    return np.array([[np.cos(a / 2), -np.sin(a / 2)], [np.sin(a / 2), np.cos(a / 2)]], dtype=complex)


CN = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)   # control q0 (left factor)


def ref(u, th, offsets=False):
    psi = np.zeros(4, complex); psi[0] = 1
    for l in range(th.shape[0]):
        if offsets:
            psi = np.kron(ry(u[0] + th[l, 0]), ry(u[1] + th[l, 1])) @ psi
        else:
            psi = np.kron(ry(u[0]), I2) @ psi
            psi = np.kron(I2, ry(u[1])) @ psi
            psi = np.kron(ry(th[l, 0]), I2) @ psi
            psi = np.kron(I2, ry(th[l, 1])) @ psi
        psi = CN @ psi
    return np.array([np.real(psi.conj() @ np.kron(P1[n[0]], P1[n[1]]) @ psi) for n in CF.PAULI_NAMES])


rng = np.random.default_rng(20261021)
U = np.concatenate([rng.uniform(-np.pi, np.pi, (300, 2)), rng.uniform(-6, 6, (100, 2)), [[0, 0], [np.pi / 2, -np.pi / 2]]])
draws = [("theta = 0 (init)", np.zeros((2, 2)))] + [(f"random theta {i}", rng.uniform(-np.pi, np.pi, (2, 2))) for i in range(5)]
worst = worst_off = 0.0
zero_cols = None
for lab, th in draws:
    got = BH.bfr_expectations(torch.tensor(U), torch.tensor(th)).numpy()
    want = np.stack([ref(u, th) for u in U])
    worst = max(worst, float(np.abs(got - want).max()))
    worst_off = max(worst_off, float(np.abs(np.stack([ref(u, th, True) for u in U[:50]]) - want[:50]).max()))
    zc = [i for i in range(15) if np.abs(want[:, i]).max() < 1e-12]
    zero_cols = zc if zero_cols is None else [c for c in zero_cols if c in zc]
check("batched torch simulator == independent per-sample complex statevector (15 Paulis, 6 theta draws)",
      worst < 1e-12, f"max|d|={worst:.1e}")
single_y = [i for i, n in enumerate(CF.PAULI_NAMES) if n.count("Y") == 1]
check("exactly the 6 single-Y Paulis are identically 0 (real state)", zero_cols == single_y,
      ",".join(CF.PAULI_NAMES[i] for i in zero_cols))
check("trainable Ry == offset on the encoded angle: Ry(u)Ry(theta) = Ry(u+theta)", worst_off < 1e-12, f"max|d|={worst_off:.1e}")
try:
    import warnings
    warnings.filterwarnings("ignore")
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector, SparsePauliOp
    mx = 0.0
    th = draws[1][1]
    for u in U[:40]:
        qc = QuantumCircuit(2)
        for l in range(2):
            qc.ry(u[0], 0); qc.ry(u[1], 1); qc.ry(th[l, 0], 0); qc.ry(th[l, 1], 1); qc.cx(0, 1)
        sv = Statevector(qc)
        q = np.array([sv.expectation_value(SparsePauliOp(n[::-1])).real for n in CF.PAULI_NAMES])  # qiskit: rightmost = qubit 0
        mx = max(mx, float(np.abs(q - BH.bfr_expectations(torch.tensor(u[None]), torch.tensor(th)).numpy()[0]).max()))
    check("simulator == Qiskit QuantumCircuit(RY, CX) Statevector expectations", mx < 1e-9, f"max|d|={mx:.1e}")
except ImportError as e:
    print(f"  (Qiskit cross-check skipped: {e}; the numpy reference stands)")
# bounded frequency: Fourier spectrum in u0 (u1 fixed) and in u1 (u0 fixed) has no content above 2
grid = np.linspace(-np.pi, np.pi, 64, endpoint=False)
hi = 0.0
for th in (draws[0][1], draws[2][1]):
    for fix in (0.3, -1.7):
        for axis in (0, 1):
            uu = np.stack([grid, np.full_like(grid, fix)], 1) if axis == 0 else np.stack([np.full_like(grid, fix), grid], 1)
            f = BH.bfr_expectations(torch.tensor(uu), torch.tensor(th)).numpy()
            spec = np.abs(np.fft.rfft(f, axis=0)) / len(grid)
            hi = max(hi, float(spec[3:].max()))
check("BOUNDED FREQUENCY: every feature is a trig polynomial of degree <= 2 in each u_i (FFT content above 2)",
      hi < 1e-12, f"max |coef| above frequency 2 = {hi:.1e}")
q = BH.BFRQuantum(5)
zz = torch.randn(20000, 2) * 50
check("|u| = |s tanh z| <= |s| = pi/2 at init, for any z", float((q.encode(zz)).abs().max()) <= abs(float(q.s)) + 1e-7)

# ---------------------------------------------------------------------------
header("3. BFR SURGERY on the real M4")
x = xin(4, seed=9)
print("  Parameter table (pre-registered):")
print("    K    M4     BFR    delta     fc_mid width   M2-w15")
for K in (5, 10):
    setcell(T, K); setcell(TB, K)
    T._V16B["head"] = None
    m4 = build(T, "M4")
    st_m4 = rng_state()
    T._V16B["head"] = "bfr"; T._V16B["registered"] = True       # no exit hook in this process
    bfr = build(T, "M4")
    st_bfr = rng_state()
    info = T._V16B["info"]
    tb = TABLE[K]
    m2 = TB.expected_params("M2", K, 15)
    print(f"    {K:<4} {info['params_m4']:<6} {info['params_bfr']:<6} {info['delta']:+d} ({100 * info['rel']:+.2f}%)  "
          f"{info['width']:<14} {m2}")
    check(f"K={K}: counts M4={info['params_m4']} BFR={info['params_bfr']} w={info['width']} M2-w15={m2} == table {tb}",
          (info["params_m4"], info["params_bfr"], info["width"], m2) == (tb["M4"], tb["BFR"], tb["w"], tb["M2"]))
    check(f"K={K}: |BFR - M4| <= 1% of M4", abs(info["params_bfr"] - info["params_m4"]) <= 0.01 * info["params_m4"])
    check(f"K={K}: actual nn.Module parameter count == table", VT.n_params(bfr) == tb["BFR"] and VT.n_params(m4) == tb["M4"])
    check(f"K={K}: every RNG state after building BFR == after building M4 (same batch order)", rng_equal(st_m4, st_bfr))
    check(f"K={K}: front-end, backbone and fc2 identical to M4 at the same seed",
          sd_equal(m4, bfr, keys=("front_end", "backbone", "back_end.fc2")))
    w = info["width"]
    sc = math.sqrt(15 / w)
    ok = (torch.equal(bfr.back_end.fc_mid.weight, m4.back_end.fc_mid.weight[:w])
          and torch.equal(bfr.back_end.fc_mid.bias, m4.back_end.fc_mid.bias[:w])
          and torch.equal(bfr.back_end.fc3.weight, m4.back_end.fc3.weight[:, :w] * sc)
          and torch.equal(bfr.back_end.fc3.bias, m4.back_end.fc3.bias * sc))
    check(f"K={K}: fc_mid = M4's first {w} rows; fc3 = M4's first {w} columns x sqrt(15/{w}) (default bound for fan-in {w})", ok)
    qm = BH.bfr_module(bfr)
    check(f"K={K}: init s=pi/2, g=0, theta=0, W_q std ~0.01",
          float(qm.s) == float(torch.tensor(math.pi / 2)) and float(qm.g) == 0.0 and not qm.theta.any()
          and 0.005 < float(qm.W_q.std()) < 0.02, f"W_q std {float(qm.W_q.std()):.4f}")
    bfr.eval()
    with torch.no_grad():
        z = bfr.back_end.fc2(bfr.backbone(bfr.front_end(x)))
        lo, cl = bfr(x), bfr.back_end.classical(z)
    check(f"K={K}: g = 0 => logits == classical path EXACTLY", torch.equal(lo, cl))
    # function class: the trimmed classical path can be M4's head exactly
    eq = copy.deepcopy(bfr)
    W3, Wm = m4.back_end.fc3.weight.double(), m4.back_end.fc_mid.weight.double()
    A = (W3 @ Wm).float(); c = (m4.back_end.fc3.bias.double() + W3 @ m4.back_end.fc_mid.bias.double()).float()
    with torch.no_grad():
        eq.back_end.fc_mid.weight.zero_(); eq.back_end.fc_mid.bias.zero_()
        eq.back_end.fc_mid.weight[0, 0] = 1.0; eq.back_end.fc_mid.weight[1, 1] = 1.0
        eq.back_end.fc3.weight.zero_(); eq.back_end.fc3.weight[:, :2] = A; eq.back_end.fc3.bias.copy_(c)
        m4.eval(); eq.eval()
        xr = xin(64, seed=11)
        d = (m4(xr) - eq(xr)).abs().max().item()
    check(f"K={K}: FUNCTION CLASS: trimmed classical path (w={w}) reproduces M4's logits", d < 1e-4, f"max|d logits|={d:.1e}")
    # gradients
    bfr.train()
    st0 = rng_state()
    _, g0 = fwd_grad(bfr, x)
    check(f"K={K}: a BFR forward + backward in train mode consumes no RNG", rng_equal(st0, rng_state()))
    gated = ("back_end.bfr.s", "back_end.bfr.theta", "back_end.bfr.W_q")
    check(f"K={K}: at g=0 the gated tensors (s, theta, W_q) get exactly zero gradient, g and everything else nonzero",
          all(g0[k] is not None and not g0[k].any() for k in gated)
          and all(v is not None and v.any() for k, v in g0.items() if k not in gated))
    with torch.no_grad():
        qm.g.fill_(1.0)
    _, g1 = fwd_grad(bfr, x)
    with torch.no_grad():
        qm.g.zero_()
    check(f"K={K}: at g=1 every parameter gets gradient (the chain is connected)", all(v is not None and v.any() for v in g1.values()))
    check(f"K={K}: expected_params(M4) while active = {T.expected_params('M4', K, 15)} (BFR); other models unchanged",
          T.expected_params("M4", K, 15) == tb["BFR"]
          and all(T.expected_params(mm, K, 15) == TB.expected_params(mm, K, 15) for mm in ("M1", "M2", "M3", "M5")))
    zn, _ = VT.z_layer(bfr)
    check(f"K={K}: fc2 is the unique Linear(-> 2) (z layer)", zn == "back_end.fc2", zn)
    T._V16B["head"] = None; T._V16B["info"] = None; T._V16B["model"] = None

# ---------------------------------------------------------------------------
header("4. DRY RUNS (50/class, 2 epochs, CPU, real data)")


def dry(tag, script="train_gadi.py", classes="1,3,5,8,9", flags=()):
    K = len(classes.split(","))
    out = os.path.join(TMP, f"out_{tag}")
    env = dict(os.environ)
    for k in ("V15_ALPHA", "V15_CALIB", "V15_EVAL_ALPHAS"):
        env.pop(k, None)
    cmd = [sys.executable, script, "--model", "M4", "--dataset", "mnist", "--classes", classes,
           "--seed", "42", "--noise_alpha", "0.0", "--n_obs", "15", "--head_variant", "linear", "--loss", "ce",
           "--deterministic", "--save_predictions", "--n_train", str(50 * K), "--n_val", str(125 * K),
           "--n_test", str(200 * K), "--max_epochs", "2", "--patience", "20", "--output_dir", out,
           "--data_root", "./data", "--checkpoint_dir", os.path.join(TMP, f"ckpt_{tag}")] + list(flags)
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, env=env)
    log = p.stdout
    open(os.path.join(TMP, f"log_{tag}.txt"), "w").write(log)
    js = sorted(f for f in os.listdir(out) if f.startswith("result_")) if os.path.isdir(out) else []
    j = json.load(open(os.path.join(out, js[0]))) if js else {}
    if p.returncode != 0:
        print(log[-3000:])
    return p.returncode, log, j, out


def same_run(a, b):
    return all(a.get(k) == b.get(k) for k in ("test_acc", "best_epoch", "stop_epoch")) and \
        a.get("test_predictions") is not None and a["test_predictions"] == b.get("test_predictions")


def splits(log):
    return re.findall(r"\[V14\] SPLIT (\w+)\s+n=\d+ .* hash=(\w+)", log)


rc0, log0, j0, _ = dry("M4_patched")
rcb, logb, jb, _ = dry("M4_prepatch", script="train_gadi.py.v16abak")
check("M4 dry runs through the patched and the pre-patch file: rc=0", rc0 == 0 and rcb == 0)
check("M4 through the patched file == pre-patch file (test_acc, epochs, every prediction)", same_run(j0, jb),
      f"{j0.get('test_acc')} vs {jb.get('test_acc')}")
check("M4: no [V16B] output without the flag", "[V16B]" not in log0)
for K, classes in ((5, "1,3,5,8,9"), (10, "0,1,2,3,4,5,6,7,8,9")):
    tb = TABLE[K]
    if K == 10:
        rcm, logm, jm, _ = dry("M4_k10", classes=classes)
    else:
        logm = log0
    rc1, log1, j1, o1 = dry(f"BFR_k{K}", classes=classes, flags=["--v16b_head", "bfr"])
    rc2, log2, j2, _ = dry(f"BFR_k{K}_again", classes=classes, flags=["--v16b_head", "bfr"])
    check(f"K={K} BFR dry run (--model M4 --v16b_head bfr): rc=0 (twice)", rc1 == 0 and rc2 == 0)
    check(f"K={K} BFR: HEAD line VERIFIED {tb['M4']} -> {tb['BFR']} (w={tb['w']})", re.search(
        rf"\[V16B\] HEAD bfr installed model=M4 seed=42 K={K} fc_mid_width={tb['w']} params_M4={tb['M4']} "
        rf"params_BFR={tb['BFR']} expected_BFR={tb['BFR']} .*VERIFIED", log1) is not None)
    check(f"K={K} BFR: the trainer's own parameter check passes", re.search(r"\[VERIFIED\] M4 params.*OK", log1) is not None)
    check(f"K={K} BFR: warmup decision = M4's (min_epochs=0)", "[V135] warmup: model=M4 min_epochs=0" in log1)
    check(f"K={K} BFR: gradient probe at g=1 ran, all tensors OK, g restored bit-exact",
          "[V16B] GRADCHECK probe with g=1" in log1 and "all trainable tensors receive gradient. OK" in log1
          and "[V16B] GRADCHECK g restored to 0.0 (bit-exact: True)" in log1)
    check(f"K={K} BFR: GS line (g, s, theta at best and stop)", re.search(r"\[V16B\] GS best\(epoch \d+\): g=\S+ s=\S+ .* \| stop\(epoch \d+\): g=", log1) is not None)
    check(f"K={K} BFR: RUN self-check MATCH", re.search(rf"\[V16B\] RUN model=BFR arch=M4 head=bfr seed=42 params={tb['BFR']} selfcheck=MATCH", log1) is not None)
    check(f"K={K} BFR: n_params in the result JSON = {tb['BFR']}", j1.get("n_params") == tb["BFR"], str(j1.get("n_params")))
    check(f"K={K} BFR: the same train/val/test images as M4", splits(log1) == splits(logm) and len(splits(log1)) == 3)
    check(f"K={K} BFR: deterministic (two runs: identical test_acc, epochs, predictions)", same_run(j1, j2))
    sf = os.path.join(o1, "v16b_run_BFR_seed42.json")
    ok = os.path.exists(sf)
    if ok:
        s = json.load(open(sf))
        ok = s["selfcheck"] == "MATCH" and s["n_state_loads"] == 1 and s["best"] is not None and s["stop"] is not None \
            and len(s["labels"]) == 200 * K
        print(f"      sidecar: best {s['best']} | stop {s['stop']}   (2 epochs; not a result)")
    check(f"K={K} BFR: sidecar v16b_run_BFR_seed42.json complete (best and stop state, one restore)", ok)
    for ln in log1.splitlines():
        if "[V16B]" in ln or "[VERIFIED]" in ln or "warmup" in ln:
            print("      | " + ln[:220])

print("\n" + "=" * 78)
if FAIL:
    print(f"VERIFY FAILED ({len(FAIL)}):")
    for f in FAIL:
        print("  - " + f)
    print(f"Logs of the dry runs: {TMP}")
    sys.exit(1)
print("ALL CHECKS PASSED")
