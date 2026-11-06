#!/usr/bin/env python3
"""
v16b-ext offline verification (amendment A3). Run from the v16bx repo root AFTER patch_v16bx.py:

    python3 -u scripts/verify_v16bx.py | tee verify_v16bx.log        (must end in ALL CHECKS PASSED)

1. UNCHANGED (no v16bx flag), K=5 and 10: M1-M5, M2c (v16a head) and BFR (v16b head) bit-identical to the pre-patch
   file (train_gadi.py.v16bbak): init, outputs, gradients, expected_params.
2. BFR-ZZ CIRCUIT (float64): == independent complex statevector with Qiskit's ZZFeatureMap entangler CX.P(zeta).CX
   (and Qiskit itself if importable); zeta switched off == BFR exactly; the 6 single-Y Paulis are no longer 0;
   the product term restores Fourier content above degree 2 in u (BFR has none).
3. SURGERY on the real M4: counts M4-trim 1639/1643, BFR-ZZ 1720/1799 (K=5/10); RNG after == after M4;
   backbone/fc2 == M4; M4-trim == BFR's classical path tensor for tensor; BFR-ZZ init == BFR init (every tensor);
   g=0 -> logits == classical path; gradient regimes at g=0 / g=1; expected_params while active; others unchanged.
4. DRY RUNS (K=10, 50/class, 2 epochs, real data): M4 and BFR through the patched file == the pre-patch file
   (every prediction); M4-trim and BFR-ZZ: rc=0, HEAD VERIFIED, warmup 0, BFR-ZZ probe + g restored, RUN MATCH,
   n_params in the result JSON, the same images as M4, deterministic.
"""
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
TMP = tempfile.mkdtemp(prefix="v16bxverify_")
COUNTS = {5: dict(M4=1719, m4trim=1639, bfrzz=1720, w=5), 10: dict(M4=1799, m4trim=1643, bfrzz=1799, w=3)}


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""), flush=True)
    if not cond:
        FAIL.append(name)


def header(s):
    print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78, flush=True)


import train_gadi as T  # noqa: E402
import bfr_head_v16b as BH  # noqa: E402
import bfr_ext_v16b as BX  # noqa: E402
import closed_form_zz_v16 as CF  # noqa: E402
import v16_tools as VT  # noqa: E402

bak = os.path.join(ROOT, "train_gadi.py.v16bbak")
if not os.path.exists(bak):
    sys.exit("train_gadi.py.v16bbak not found: run patch_v16bx.py first (setup_v16bx.sh does).")
_ld = SourceFileLoader("train_gadi_v16bbak", bak)
TB = importlib.util.module_from_spec(importlib.util.spec_from_loader("train_gadi_v16bbak", _ld))
_ld.exec_module(TB)
src = open("train_gadi.py").read()
check("v16bx marker present; v16a and v16b blocks still present",
      all(m in src for m in ("V16BX_EXT_HEAD", "V16B_BFR_HEAD", "V16A_HEAD_SAVEZ")))
check("v16bx, v16b, v16a inactive in this process", T._V16BX["head"] is None and T._V16B["head"] is None and not T._v16_active())


def setcell(mod, K):
    mod.N_CLASSES, mod.N_OBS, mod.HEAD_VARIANT = K, 15, "linear"


def build(mod, m, seed=42):
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    return mod.build_model(m, seed=seed)


def sd_equal(a, b, keys=None, strip=None):
    sa, sb = a.state_dict(), b.state_dict()
    ka = [k for k in sa if (keys is None or any(k.startswith(p) for p in keys)) and not (strip and k.startswith(strip))]
    kb = [k for k in sb if (keys is None or any(k.startswith(p) for p in keys)) and not (strip and k.startswith(strip))]
    return ka == kb and all(torch.equal(sa[k], sb[k]) for k in ka)


def xin(n=2, seed=0):
    return torch.rand(n, 1, 28, 28, generator=torch.Generator().manual_seed(seed))


def fwd_grad(model, x):
    model.zero_grad()
    out = model(x)
    out.float().pow(2).sum().backward()
    return out.detach(), {n: (p.grad.clone() if p.grad is not None else None) for n, p in model.named_parameters()}


def grads_equal(ga, gb):
    return ga.keys() == gb.keys() and all((ga[k] is None and gb[k] is None) or
                                          (ga[k] is not None and gb[k] is not None and torch.equal(ga[k], gb[k])) for k in ga)


def rng_state():
    return (random.getstate(), np.random.get_state()[1].tobytes(), torch.get_rng_state())


def rng_equal(a, b):
    return a[0] == b[0] and a[1] == b[1] and torch.equal(a[2], b[2])


# ---------------------------------------------------------------------------
header("1. UNCHANGED vs pre-patch train_gadi.py (no v16bx flag), w15, bit-exact")
for K in (5, 10):
    setcell(T, K); setcell(TB, K)
    for m in ("M1", "M2", "M3", "M4", "M5"):
        a, b = build(T, m), build(TB, m)
        oa, ga = fwd_grad(a, xin()); ob, gb = fwd_grad(b, xin())
        check(f"K={K} {m}: init, output, gradients identical; expected_params unchanged",
              sd_equal(a, b) and torch.equal(oa, ob) and grads_equal(ga, gb)
              and T.expected_params(m, K, 15) == TB.expected_params(m, K, 15))
    for label, attr, val, mname in (("M2c", "_V16", "m2c", "M2"), ("BFR", "_V16B", "bfr", "M4")):
        for mod in (T, TB):
            getattr(mod, attr)["head"] = val; getattr(mod, attr)["registered"] = True
        a, b = build(T, mname), build(TB, mname)
        oa, ga = fwd_grad(a, xin()); ob, gb = fwd_grad(b, xin())
        check(f"K={K} {label}: init, output, gradients identical; expected_params unchanged",
              sd_equal(a, b) and torch.equal(oa, ob) and grads_equal(ga, gb)
              and T.expected_params(mname, K, 15) == TB.expected_params(mname, K, 15), f"params {VT.n_params(a)}")
        for mod in (T, TB):
            getattr(mod, attr)["head"] = None; getattr(mod, attr)["info"] = None

# ---------------------------------------------------------------------------
header("2. BFR-ZZ CIRCUIT (float64)")
I2 = np.eye(2)
P1 = dict(I=I2, X=np.array([[0, 1], [1, 0]]), Y=np.array([[0, -1j], [1j, 0]]), Z=np.diag([1., -1.]))
CN = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)


def ry(t):
    return np.array([[np.cos(t / 2), -np.sin(t / 2)], [np.sin(t / 2), np.cos(t / 2)]], dtype=complex)


def ref(u, th):
    psi = np.zeros(4, complex); psi[0] = 1
    zeta = 2 * (np.pi - u[0]) * (np.pi - u[1])
    Pq1 = np.kron(I2, np.diag([1, np.exp(1j * zeta)]))
    for l in range(th.shape[0]):
        psi = np.kron(ry(u[0]), I2) @ psi
        psi = np.kron(I2, ry(u[1])) @ psi
        psi = CN @ (Pq1 @ (CN @ psi))                 # ZZFeatureMap entangler: CX(0->1), P(zeta) on q1, CX(0->1)
        psi = np.kron(ry(th[l, 0]), I2) @ psi
        psi = np.kron(I2, ry(th[l, 1])) @ psi
        psi = CN @ psi
    return np.array([np.real(psi.conj() @ np.kron(P1[n[0]], P1[n[1]]) @ psi) for n in CF.PAULI_NAMES])


rng = np.random.default_rng(20261022)
U = np.concatenate([rng.uniform(-np.pi / 2, np.pi / 2, (200, 2)), rng.uniform(-3, 3, (60, 2))])
worst = worst0 = 0.0
ymax = 0.0
for th in [np.zeros((2, 2))] + [rng.uniform(-np.pi, np.pi, (2, 2)) for _ in range(4)]:
    got = BX.bfrzz_expectations(torch.tensor(U), torch.tensor(th)).numpy()
    worst = max(worst, float(np.abs(got - np.stack([ref(u, th) for u in U])).max()))
    off = BX.bfrzz_expectations(torch.tensor(U), torch.tensor(th), 0.0)
    worst0 = max(worst0, float((off - BH.bfr_expectations(torch.tensor(U), torch.tensor(th))).abs().max()))
    ymax = max(ymax, float(np.abs(got[:, [i for i, n in enumerate(CF.PAULI_NAMES) if n.count("Y") == 1]]).max()))
check("BFR-ZZ simulator == independent complex statevector (ZZFeatureMap entangler CX.P(zeta).CX), 5 theta draws",
      worst < 1e-12, f"max|d|={worst:.1e}")
check("product term switched off (zeta=0) == BFR exactly: the product term is the only difference", worst0 == 0.0,
      f"max|d|={worst0:.1e}")
check("the 6 single-Y Paulis are no longer identically 0 (complex state; stated confound)", ymax > 0.1, f"max|<P>|={ymax:.2f}")
try:
    import warnings
    warnings.filterwarnings("ignore")
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector, SparsePauliOp
    th = rng.uniform(-np.pi, np.pi, (2, 2)); mx = 0.0
    for u in U[:30]:
        qc = QuantumCircuit(2)
        z = 2 * (np.pi - u[0]) * (np.pi - u[1])
        for l in range(2):
            qc.ry(u[0], 0); qc.ry(u[1], 1); qc.cx(0, 1); qc.p(z, 1); qc.cx(0, 1)
            qc.ry(th[l, 0], 0); qc.ry(th[l, 1], 1); qc.cx(0, 1)
        sv = Statevector(qc)
        q = np.array([sv.expectation_value(SparsePauliOp(n[::-1])).real for n in CF.PAULI_NAMES])
        mx = max(mx, float(np.abs(q - BX.bfrzz_expectations(torch.tensor(u[None]), torch.tensor(th)).numpy()[0]).max()))
    check("BFR-ZZ simulator == Qiskit QuantumCircuit Statevector", mx < 1e-9, f"max|d|={mx:.1e}")
except Exception as e:  # noqa: BLE001
    print(f"  (Qiskit cross-check skipped: {type(e).__name__}: {str(e)[:90]}; the numpy reference stands)")
grid = np.linspace(-np.pi, np.pi, 256, endpoint=False)
th = rng.uniform(-np.pi, np.pi, (2, 2))
hz, hb = 0.0, 0.0
for fix in (0.3, -1.1):
    uu = np.stack([grid, np.full_like(grid, fix)], 1)
    hz = max(hz, float((np.abs(np.fft.rfft(BX.bfrzz_expectations(torch.tensor(uu), torch.tensor(th)).numpy(), axis=0)) / 256)[3:].max()))
    hb = max(hb, float((np.abs(np.fft.rfft(BH.bfr_expectations(torch.tensor(uu), torch.tensor(th)).numpy(), axis=0)) / 256)[3:].max()))
check("product term restores Fourier content above degree 2 in u (BFR: none)", hz > 1e-2 and hb < 1e-12,
      f"BFR-ZZ max coef above 2 = {hz:.2e}; BFR = {hb:.1e}")
dz = 2 * (np.pi + np.pi / 2)
print(f"  local frequency of zeta in u within the bound |u| <= pi/2: |d zeta/d u0| = 2|pi - u1| <= {dz:.2f} "
      f"(vs 1 per Ry upload)")

# ---------------------------------------------------------------------------
header("3. SURGERY on the real M4")
x = xin(4, seed=9)
print("  Parameter table:   K   M4     M4-trim   BFR-ZZ   (BFR)   fc_mid width")
for K in (5, 10):
    c = COUNTS[K]
    setcell(T, K)
    m4 = build(T, "M4"); st_m4 = rng_state()
    T._V16B["head"] = "bfr"; T._V16B["registered"] = True
    bfr = build(T, "M4"); T._V16B["head"] = None; T._V16B["info"] = None
    built = {}
    for kind in ("m4trim", "bfrzz"):
        T._V16BX["head"] = kind; T._V16BX["registered"] = True
        mdl = build(T, "M4"); st = rng_state()
        info = T._V16BX["info"]
        built[kind] = mdl
        check(f"K={K} {kind}: params {info['params']} == {c[kind]} (w={info['width']}), HEAD ok",
              info["ok"] and info["params"] == c[kind] and info["width"] == c["w"] and VT.n_params(mdl) == c[kind])
        check(f"K={K} {kind}: every RNG state after building == after building M4", rng_equal(st_m4, st))
        check(f"K={K} {kind}: front-end, backbone, fc2 identical to M4", sd_equal(m4, mdl, keys=("front_end", "backbone", "back_end.fc2")))
        check(f"K={K} {kind}: expected_params(M4) while active = {c[kind]}; other models unchanged",
              T.expected_params("M4", K, 15) == c[kind]
              and all(T.expected_params(mm, K, 15) == T._v16bx_ep_orig(mm, K, 15) for mm in ("M1", "M2", "M3", "M5")))
        T._V16BX["head"] = None; T._V16BX["info"] = None; T._V16BX["model"] = None
    print(f"                    {K:<3} {c['M4']:<6} {c['m4trim']:<9} {c['bfrzz']:<8} ({VT.n_params(bfr)})  {c['w']}")
    check(f"K={K}: M4-trim == BFR's classical path, tensor for tensor", sd_equal(built["m4trim"], bfr, strip="back_end.bfr."))
    check(f"K={K}: BFR-ZZ initialisation == BFR initialisation, every tensor (same W_q generator draw)",
          sd_equal(built["bfrzz"], bfr))
    for kind in ("m4trim", "bfrzz"):
        mdl = built[kind]; mdl.eval()
        with torch.no_grad():
            z = mdl.back_end.fc2(mdl.backbone(mdl.front_end(x)))
            same = torch.equal(mdl(x), mdl.back_end.classical(z)) and torch.equal(mdl(x), bfr.eval()(x))
        check(f"K={K} {kind}: at init logits == classical path == BFR at init (g=0)", same)
    zz = built["bfrzz"]; zz.train(); qz = BH.bfr_module(zz)
    st0 = rng_state(); _, g0 = fwd_grad(zz, x)
    gated = ("back_end.bfr.s", "back_end.bfr.theta", "back_end.bfr.W_q")
    check(f"K={K} bfrzz: forward/backward consumes no RNG; at g=0 only g among gated tensors gets gradient",
          rng_equal(st0, rng_state()) and all(not g0[k].any() for k in gated)
          and all(v is not None and v.any() for k, v in g0.items() if k not in gated))
    with torch.no_grad():
        qz.g.fill_(1.0)
    _, g1 = fwd_grad(zz, x)
    with torch.no_grad():
        qz.g.zero_()
    check(f"K={K} bfrzz: at g=1 every parameter gets gradient", all(v is not None and v.any() for v in g1.values()))
    _, gt = fwd_grad(built["m4trim"].train(), x)
    check(f"K={K} m4trim: every parameter gets gradient", all(v is not None and v.any() for v in gt.values()))

# ---------------------------------------------------------------------------
header("4. DRY RUNS (K=10, 50/class, 2 epochs, real data)")
CL10 = "0,1,2,3,4,5,6,7,8,9"


def dry(tag, script="train_gadi.py", flags=()):
    out = os.path.join(TMP, f"out_{tag}")
    env = dict(os.environ)
    cmd = [sys.executable, script, "--model", "M4", "--dataset", "mnist", "--classes", CL10, "--seed", "42",
           "--noise_alpha", "0.0", "--n_obs", "15", "--head_variant", "linear", "--loss", "ce", "--deterministic",
           "--save_predictions", "--n_train", "500", "--n_val", "1250", "--n_test", "2000", "--max_epochs", "2",
           "--patience", "20", "--output_dir", out, "--data_root", "./data", "--checkpoint_dir", os.path.join(TMP, f"ck_{tag}")] + list(flags)
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, env=env)
    js = sorted(f for f in os.listdir(out) if f.startswith("result_")) if os.path.isdir(out) else []
    j = json.load(open(os.path.join(out, js[0]))) if js else {}
    open(os.path.join(TMP, f"log_{tag}.txt"), "w").write(p.stdout)
    if p.returncode != 0:
        print(p.stdout[-3000:])
    return p.returncode, p.stdout, j, out


def same_run(a, b):
    return all(a.get(k) == b.get(k) for k in ("test_acc", "best_epoch", "stop_epoch")) and \
        a.get("test_predictions") is not None and a["test_predictions"] == b.get("test_predictions")


def splits(log):
    return re.findall(r"\[V14\] SPLIT (\w+)\s+n=\d+ .* hash=(\w+)", log)


rm, logm, jm, _ = dry("M4")
rmb, _, jmb, _ = dry("M4_pre", "train_gadi.py.v16bbak")
check("M4 through the patched file == pre-patch file (every prediction)", rm == 0 and rmb == 0 and same_run(jm, jmb))
check("M4: no [V16BX] output without the flag", "[V16BX]" not in logm)
rb, logb, jb, _ = dry("BFR", flags=["--v16b_head", "bfr"])
rbb, _, jbb, _ = dry("BFR_pre", "train_gadi.py.v16bbak", ["--v16b_head", "bfr"])
check("BFR through the patched file == pre-patch file (every prediction); no [V16BX] line",
      rb == 0 and rbb == 0 and same_run(jb, jbb) and "[V16BX]" not in logb)
for kind, label in (("m4trim", "M4trim"), ("bfrzz", "BFRZZ")):
    c = COUNTS[10][kind]
    r1, l1, j1, o1 = dry(kind, flags=["--v16bx_head", kind])
    r2, _, j2, _ = dry(kind + "_again", flags=["--v16bx_head", kind])
    check(f"{label}: rc=0 (twice)", r1 == 0 and r2 == 0)
    check(f"{label}: HEAD line VERIFIED ({c})", re.search(rf"\[V16BX\] HEAD {kind} installed model=M4 seed=42 K=10 fc_mid_width=3 "
                                                         rf"params_M4=1799 params={c} expected={c}.* VERIFIED", l1) is not None)
    check(f"{label}: trainer's parameter check OK; warmup 0 (= M4)",
          re.search(r"\[VERIFIED\] M4 params.*OK", l1) is not None and "[V135] warmup: model=M4 min_epochs=0" in l1)
    if kind == "bfrzz":
        check(f"{label}: gradient probe at g=1, all OK, g restored bit-exact; GS line",
              "[V16BX] GRADCHECK probe with g=1" in l1 and "all trainable tensors receive gradient. OK" in l1
              and "[V16BX] GRADCHECK g restored to 0.0 (bit-exact: True)" in l1 and "[V16BX] GS best(epoch" in l1)
    check(f"{label}: RUN self-check MATCH", re.search(rf"\[V16BX\] RUN model={label} arch=M4 head={kind} seed=42 params={c} selfcheck=MATCH", l1) is not None)
    check(f"{label}: n_params in result JSON = {c}", j1.get("n_params") == c, str(j1.get("n_params")))
    check(f"{label}: same train/val/test images as M4", splits(l1) == splits(logm) and len(splits(l1)) == 3)
    check(f"{label}: deterministic (two runs identical)", same_run(j1, j2))
    sf = os.path.join(o1, f"v16bx_run_{label}_seed42.json")
    ok = os.path.exists(sf) and json.load(open(sf))["selfcheck"] == "MATCH"
    if ok and kind == "bfrzz":
        s = json.load(open(sf)); ok = s["n_state_loads"] == 1 and s["best"] and s["stop"]
    check(f"{label}: sidecar complete", bool(ok))
    for ln in l1.splitlines():
        if "[V16BX]" in ln or "warmup" in ln:
            print("      | " + ln[:220])

print("\n" + "=" * 78)
if FAIL:
    print(f"VERIFY FAILED ({len(FAIL)}):")
    for f in FAIL:
        print("  - " + f)
    print(f"Logs of the dry runs: {TMP}")
    sys.exit(1)
print("ALL CHECKS PASSED")
