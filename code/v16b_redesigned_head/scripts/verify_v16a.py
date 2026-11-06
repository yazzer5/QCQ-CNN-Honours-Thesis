#!/usr/bin/env python3
"""
v16a offline verification. Run from the v16a repo root AFTER patch_v16a.py, on the login
node (CPU, a few minutes), in the qcqcnn env:

    python3 -u scripts/verify_v16a.py | tee verify_v16a.log

Must end in ALL CHECKS PASSED before any v16a job is submitted.

1. M1-M5 UNCHANGED (no v16 flag): K=5 and K=10, w15: initial weights, outputs, gradients and
   expected_params bit-identical to the pre-patch file (train_gadi.py.v15bak).
2. THE CIRCUIT (the real qnn_torch.QNNBackendTorch, float64):
   a. its 15 outputs equal an independent statevector of ZZFeatureMap(2, reps=1) +
      RealAmplitudes(2, reps=2) with Qiskit's phase convention P(2z), P(2(pi-z0)(pi-z1)),
      column by column (<= 1e-6); reps=2 is shown NOT to match; Qiskit itself if importable
   b. SPAN CHECK (pre-registered): every one of the 15 Pauli expectations is a linear
      combination of the 12 closed-form features + a constant, residual <= 1e-6, over random
      z (|z| up to ~12) and 6 random RealAmplitudes parameter draws (+ the seed-42 init)
   c. the converse: each of the 12 features is a linear combination of the 15 Paulis
      (<= 1e-6), so the two heads have the SAME feature span for generic parameters
3. M2c SURGERY on the real M2 (K=5, K=10): exactly one QNN, one Linear(15 -> K) reading it
   directly, fc2 output = QNN input; parameter counts M2 1686/1766 -> M2c 1665/1730;
   front-end, backbone and fc2 identical to M2 at the same seed; every RNG state unchanged;
   gradients reach the Circuit-4 front-end; expected_params(M2) = M2c count while active;
   FUNCTION-CLASS CHECK: M2c with its Linear set to W_fc3 . C (C from 2b) reproduces M2's logits.
4. DRY RUNS through train_gadi.py (K=5 A, 50/class, 2 epochs, CPU, real data):
   M3 and M2 with and without --save_z: identical test_acc, epochs and every prediction
   (the flag is invisible to training), SAVEZ/RUN self-check MATCH, z file (n_test, 2);
   M2 --v16_head m2c: runs, HEAD line VERIFIED, trainer's own parameter check OK,
   warmup 30, RUN MATCH, and the same train/val/test images as M2.
"""
import copy
import importlib.util
import json
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
TMP = tempfile.mkdtemp(prefix="v16averify_")
TABLE_M2 = {5: 1686, 10: 1766}
TABLE_M2C = {5: 1665, 10: 1730}


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""), flush=True)
    if not cond:
        FAIL.append(name)


def header(s):
    print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78, flush=True)


import train_gadi as T  # noqa: E402
import closed_form_zz_v16 as CF  # noqa: E402
import v16_tools as VT  # noqa: E402
import qnn_torch as Q  # noqa: E402

bak = os.path.join(ROOT, "train_gadi.py.v15bak")
if not os.path.exists(bak):
    sys.exit("train_gadi.py.v15bak not found: run patch_v16a.py first (setup_v16a.sh does).")
_ld = SourceFileLoader("train_gadi_v15bak", bak)
TB = importlib.util.module_from_spec(importlib.util.spec_from_loader("train_gadi_v15bak", _ld))
_ld.exec_module(TB)
check("v16a patch marker present", "V16A_HEAD_SAVEZ" in open("train_gadi.py").read())
check("v16a inactive in this process (no flag)", not T._v16_active())
check("v15 noise inactive in this process (no V15_ALPHA)", T._V15["active"] is False)


def setcell(mod, K, w=15):
    mod.N_CLASSES, mod.N_OBS = K, w
    if hasattr(mod, "HEAD_VARIANT"):
        mod.HEAD_VARIANT = "linear"


def build(mod, m, seed=42):
    torch.manual_seed(seed)
    return mod.build_model(m, seed=seed)


def sd_equal(a, b, skip=()):
    sa, sb = a.state_dict(), b.state_dict()
    ka = [k for k in sa if not any(k.startswith(s) for s in skip)]
    kb = [k for k in sb if not any(k.startswith(s) for s in skip)]
    return ka == kb and all(torch.equal(sa[k], sb[k]) for k in ka)


def xin(n=2, seed=0):
    return torch.rand(n, 1, 28, 28, generator=torch.Generator().manual_seed(seed))


def fwd_grad(model, x):
    model.zero_grad()
    out = model(x)
    out.float().pow(2).sum().backward()
    return out.detach(), {n: (p.grad.clone() if p.grad is not None else None) for n, p in model.named_parameters()}


def rng_state():
    return (random.getstate(), np.random.get_state()[1].tobytes(), torch.get_rng_state())


def rng_equal(a, b):
    return a[0] == b[0] and a[1] == b[1] and torch.equal(a[2], b[2])


MODELS = ("M1", "M2", "M3", "M4", "M5")

# ---------------------------------------------------------------------------
header("1. M1-M5 UNCHANGED vs pre-patch train_gadi.py (no v16 flag), w15, bit-exact")
for K in (5, 10):
    setcell(T, K); setcell(TB, K)
    for m in MODELS:
        a, b = build(T, m), build(TB, m)
        x = xin()
        oa, ga = fwd_grad(a, x)
        ob, gb = fwd_grad(b, x)
        same_grad = ga.keys() == gb.keys() and all(
            (ga[k] is None and gb[k] is None) or (ga[k] is not None and gb[k] is not None and torch.equal(ga[k], gb[k]))
            for k in ga)
        check(f"K={K} {m}: init, output and gradients identical", sd_equal(a, b) and torch.equal(oa, ob) and same_grad)
        if hasattr(TB, "expected_params"):
            ep, epb = T.expected_params(m, K, 15), TB.expected_params(m, K, 15)
            check(f"K={K} {m}: expected_params unchanged", ep == epb, str(ep))

# ---------------------------------------------------------------------------
header("2. THE CIRCUIT: real qnn_torch.QNNBackendTorch vs the closed form (float64)")
rng = np.random.default_rng(20261020)
Z = np.concatenate([rng.normal(0, 2.0, (3000, 2)), rng.uniform(-12, 12, (1000, 2)),
                    np.array([[np.pi, np.pi], [0, 0], [np.pi, -np.pi]])])
Zt = torch.tensor(Z, dtype=torch.float64)
F12 = CF.zz_trig_features(Zt).numpy()
check("12 closed-form features are linearly independent (rank 12 after centring)",
      np.linalg.matrix_rank(F12 - F12.mean(0), tol=1e-8) == 12)
setcell(T, 5)
m2_42 = build(T, "M2", 42)
qnn42 = VT.find_qnn(m2_42)[0][1]
draws = [("seed-42 init of M2", qnn42.weight.detach().double().numpy().copy() if hasattr(qnn42, "weight")
          else next(qnn42.parameters()).detach().double().numpy().copy())]
draws += [(f"random draw {i}", rng.uniform(-np.pi, np.pi, 6)) for i in range(6)]
qmod = Q.QNNBackendTorch(seed=3, n_obs=15).double()
wparam = [p for p in qmod.parameters()]
check("QNNBackendTorch has exactly one 6-element parameter", len(wparam) == 1 and wparam[0].numel() == 6,
      str([tuple(p.shape) for p in wparam]))
worst_ref, worst_span, worst_conv, worst_r2 = 0.0, 0.0, 0.0, np.inf
names_seen = None
for lab, th in draws:
    with torch.no_grad():
        wparam[0].copy_(torch.tensor(th, dtype=torch.float64))
        out = qmod(Zt).numpy()
    check(f"[{lab}] module returns 15 columns", out.shape == (len(Z), 15), str(out.shape))
    ref1 = CF.reference_paulis_numpy(Z, th, 1)
    ref2 = CF.reference_paulis_numpy(Z, th, 2)
    # column-by-column bijection with the reps=1 reference
    names = []
    for c in range(out.shape[1]):
        d = np.abs(ref1 - out[:, [c]]).max(0)
        names.append(CF.PAULI_NAMES[int(d.argmin())] if d.min() <= 1e-6 else "?")
        worst_ref = max(worst_ref, float(d.min()))
    names_seen = names
    worst_r2 = min(worst_r2, float(min(np.abs(ref2 - out[:, [c]]).max(0).min() for c in range(15))))
    _, res = CF.span_fit(F12, out)
    worst_span = max(worst_span, float(res.max()))
    _, res2 = CF.span_fit(out, F12)
    worst_conv = max(worst_conv, float(res2.max()))
check("2a. every module column equals one Pauli of the reps=1 reference (bijection)",
      "?" not in names_seen and sorted(names_seen) == sorted(CF.PAULI_NAMES),
      f"max|d|={worst_ref:.2e}  columns={','.join(names_seen)}")
check("2a. ZZFeatureMap reps=2 does NOT reproduce the module (so reps=1 is the convention)", worst_r2 > 1e-3,
      f"best reps=2 column distance {worst_r2:.2e}")
try:
    import warnings
    warnings.filterwarnings("ignore")
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
    from qiskit.quantum_info import Statevector, SparsePauliOp
    fm, an = ZZFeatureMap(2, reps=1), RealAmplitudes(2, reps=2)
    qc = fm.compose(an)
    mx = 0.0
    th = draws[1][1]
    for zz in Z[:40]:
        sv = Statevector(qc.assign_parameters(dict(zip(list(fm.parameters) + list(an.parameters), list(zz) + list(th)))))
        q = np.array([sv.expectation_value(SparsePauliOp(n[::-1])).real for n in CF.PAULI_NAMES])
        mx = max(mx, np.abs(q - CF.reference_paulis_numpy(zz[None], th, 1)[0]).max())
    check("2a. reference == Qiskit ZZFeatureMap(2,reps=1)+RealAmplitudes(2,reps=2) Statevector", mx < 1e-9, f"max|d|={mx:.1e}")
except Exception as e:  # noqa: BLE001
    print(f"  (Qiskit cross-check skipped: {type(e).__name__}: {str(e)[:80]}; the numpy reference stands)")
check("2b. SPAN CHECK: all 15 Paulis = linear combination of the 12 features + constant, residual <= 1e-6",
      worst_span <= 1e-6, f"max residual {worst_span:.2e} over {len(draws)} parameter draws x {len(Z)} z")
check("2c. converse: all 12 features = linear combination of the 15 Paulis, residual <= 1e-6",
      worst_conv <= 1e-6, f"max residual {worst_conv:.2e}")
print("  features: " + ", ".join(CF.FEATURE_NAMES))
print("  phase differences: d01=2z0+zeta d02=2z1+zeta d03=2z0+2z1 d12=2z1-2z0 d13=2z1-zeta d23=2z0-zeta, "
      "zeta=2(pi-z0)(pi-z1)")

# ---------------------------------------------------------------------------
header("3. M2c SURGERY on the real M2")
x = xin(4, seed=9)
for K in (5, 10):
    setcell(T, K)
    T._V16["head"] = None
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    m2 = T.build_model("M2", seed=42)
    st_m2 = rng_state()
    T._V16["head"] = "m2c"; T._V16["registered"] = True      # no exit hook in this process
    torch.manual_seed(0); np.random.seed(0); random.seed(0)
    m2c = T.build_model("M2", seed=42)
    st_m2c = rng_state()
    info = T._V16["info"]
    check(f"K={K}: surgery done (one QNN, one Linear(15->{K}) reading it directly, fc2 -> QNN identity)",
          info is not None and info["K"] == K, f"qnn={info['qnn']} readout={info['readout']} z={info['z_layer']}")
    check(f"K={K}: parameter counts M2={info['params_m2']} (table {TABLE_M2[K]}) -> M2c={info['params_m2c']} "
          f"(table {TABLE_M2C[K]})", info["params_m2"] == TABLE_M2[K] and info["params_m2c"] == TABLE_M2C[K])
    check(f"K={K}: every RNG state after building M2c == after building M2", rng_equal(st_m2, st_m2c))
    skip = (info["qnn"], info["readout"])
    check(f"K={K}: front-end, backbone and fc2 identical to M2 at the same seed", sd_equal(m2, m2c, skip=skip))
    check(f"K={K}: the QNN module is gone, ZZTrig12 is in its place, 0 parameters",
          not VT.find_qnn(m2c) and isinstance(dict(m2c.named_modules())[info["qnn"]], CF.ZZTrig12))
    feat = dict(m2c.named_modules())[info["qnn"]]
    wb = getattr(feat, "weight", None)
    check(f"K={K}: shim for train_model's qnn_weight_drift diagnostic: .weight is a constant 6-element BUFFER, "
          f"not a parameter, not in state_dict",
          isinstance(wb, torch.Tensor) and wb.numel() == 6 and not any(p is wb for p in m2c.parameters())
          and not any(k.endswith(info["qnn"] + ".weight") for k in m2c.state_dict()))
    if hasattr(T, "expected_params"):
        e2c = T.expected_params("M2", K, 15)
        others = all(T.expected_params(mm, K, 15) == TB.expected_params(mm, K, 15) for mm in ("M1", "M3", "M4", "M5"))
        check(f"K={K}: expected_params(M2) while the head is active = {e2c} (M2c); other models unchanged",
              e2c == TABLE_M2C[K] and others)
    out, g = fwd_grad(m2c, x)
    fe_g = [v for k, v in g.items() if k.startswith("front_end")]
    check(f"K={K}: M2c forward (B, {K}) and gradients reach the Circuit-4 front-end",
          out.shape == (4, K) and fe_g and all(v is not None and torch.count_nonzero(v) > 0 for v in fe_g))
    # function-class check: M2c can represent M2 exactly
    qnn = dict(m2.named_modules())[info["qnn"]]
    fc3 = dict(m2.named_modules())[info["readout"]]
    th = next(qnn.parameters()).detach().double().numpy()
    qd = copy.deepcopy(qnn).double()
    with torch.no_grad():
        Cmat, res = CF.span_fit(F12, qd(Zt).numpy())                # (13, 15)
    W = fc3.weight.detach().double().numpy(); b = fc3.bias.detach().double().numpy()
    eq = copy.deepcopy(m2c)
    lin = dict(eq.named_modules())[info["readout"]]
    with torch.no_grad():
        lin.weight.copy_(torch.tensor(W @ Cmat[:12].T, dtype=lin.weight.dtype))
        lin.bias.copy_(torch.tensor(b + W @ Cmat[12], dtype=lin.bias.dtype))
        m2.eval(); eq.eval()
        xr = xin(64, seed=11)
        d = (m2(xr) - eq(xr)).abs().max().item()
    check(f"K={K}: FUNCTION CLASS: M2c with Linear = W_fc3 . C reproduces M2's logits (float32)", d < 2e-4,
          f"max|d logits|={d:.2e}")
    T._V16["head"] = None; T._V16["info"] = None
# z layer, all models
setcell(T, 5)
for m in MODELS:
    mm = build(T, m)
    try:
        zn, _ = VT.z_layer(mm)
        check(f"{m}: unique Linear(-> 2) = z layer", True, zn)
    except VT.V16Error as e:
        check(f"{m}: unique Linear(-> 2) = z layer", False, str(e))
mm = build(T, "M2")
zo, qi, _ = VT.wiring(mm, VT.z_layer(mm)[1], VT.find_qnn(mm)[0][1], xin(3))
check("M2: z (fc2 output) is exactly the QNN head's input", torch.equal(zo, qi))

# ---------------------------------------------------------------------------
header("4. DRY RUNS (K=5 A, 50/class, 2 epochs, CPU, real data)")


def dry(m, tag, flags=()):
    out = os.path.join(TMP, f"out_{tag}")
    env = dict(os.environ)
    for k in ("V15_ALPHA", "V15_CALIB", "V15_EVAL_ALPHAS"):
        env.pop(k, None)
    cmd = [sys.executable, "train_gadi.py", "--model", m, "--dataset", "mnist", "--classes", "1,3,5,8,9",
           "--seed", "42", "--noise_alpha", "0.0", "--n_obs", "15", "--head_variant", "linear", "--loss", "ce",
           "--deterministic", "--save_predictions", "--n_train", "250", "--n_val", "625", "--n_test", "1000",
           "--max_epochs", "2", "--patience", "20", "--output_dir", out, "--data_root", "./data",
           "--checkpoint_dir", os.path.join(TMP, f"ckpt_{tag}")] + list(flags)
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, env=env)
    log = p.stdout
    open(os.path.join(TMP, f"log_{tag}.txt"), "w").write(log)
    js = sorted(f for f in os.listdir(out) if f.startswith("result_")) if os.path.isdir(out) else []
    j = json.load(open(os.path.join(out, js[0]))) if js else {}
    if p.returncode != 0:
        print(log[-3000:])
    return p.returncode, log, j, out


def pk(j):
    return next((k for k in ("predictions", "test_predictions", "preds") if k in j), None)


def same_run(a, b):
    return all(a.get(k) == b.get(k) for k in ("test_acc", "best_epoch", "stop_epoch")) and pk(a) is not None \
        and a[pk(a)] == b.get(pk(a))


def splits(log):
    return re.findall(r"\[V14\] SPLIT (\w+)\s+n=\d+ .* hash=(\w+)", log)


for m in ("M3", "M2"):
    rc0, log0, j0, _ = dry(m, f"{m}_plain")
    rc1, log1, j1, o1 = dry(m, f"{m}_z", ["--save_z"])
    check(f"{m} dry run, plain and --save_z: rc=0", rc0 == 0 and rc1 == 0)
    check(f"{m}: --save_z is invisible to training (test_acc, best/stop epoch, every prediction identical)",
          same_run(j0, j1), f"{j0.get('test_acc')} vs {j1.get('test_acc')}")
    check(f"{m}: no v16 output without a flag", "[V16]" not in log0)
    check(f"{m} --save_z: SAVEZ and RUN self-check MATCH",
          re.search(rf"\[V16\] SAVEZ model={m} seed=42 n=1000 dim=2 .*selfcheck=MATCH", log1) is not None
          and re.search(rf"\[V16\] RUN model={m} .*selfcheck=MATCH", log1) is not None)
    zf = os.path.join(o1, f"v16_z_{m}_seed42.npz")
    ok = os.path.exists(zf)
    if ok:
        zz = np.load(zf)
        ok = zz["z"].shape == (1000, 2) and np.isfinite(zz["z"]).all() and len(zz["labels"]) == 1000
        sc = json.load(open(os.path.join(o1, f"v16_run_{m}_seed42.json")))
        ok = ok and sc["z_sha256"] == VT.sha(zz["z"]) and [int(v) for v in zz["preds"]] == j1[pk(j1)]
        print(f"      z range dim0 [{zz['z'][:, 0].min():.2f}, {zz['z'][:, 0].max():.2f}] "
              f"dim1 [{zz['z'][:, 1].min():.2f}, {zz['z'][:, 1].max():.2f}]   (2 epochs; not a result)")
    check(f"{m} --save_z: z file (1000, 2), finite, sidecar hash matches", ok)
    if m == "M2":
        log_m2 = log0
rcc, logc, jc, oc = dry("M2", "M2c", ["--v16_head", "m2c", "--v16_sidecar"])
check("M2c dry run (--model M2 --v16_head m2c): rc=0", rcc == 0)
check("M2c: HEAD line VERIFIED with 1686 -> 1665", re.search(
    r"\[V16\] HEAD m2c installed model=M2 seed=42 .*wiring=identity params_M2=1686 params_M2c=1665 expected_M2c=1665 VERIFIED",
    logc) is not None)
check("M2c: the trainer's own parameter check passes", re.search(r"VERIFIED.*OK", logc) is not None)
check("M2c: warmup is M2's (30 epochs)", "[V135] warmup: model=M2 min_epochs=30" in logc)
check("M2c: RUN self-check MATCH", re.search(r"\[V16\] RUN model=M2c arch=M2 head=m2c .*selfcheck=MATCH", logc) is not None)
drift = re.findall(r"qnn_weight_drift=(\S+)", logc)
check("M2c: train_model's qnn_weight_drift diagnostic runs and reports 0 (no circuit)",
      ("qnn_weight_drift" not in log_m2) or (bool(drift) and all(float(d) == 0.0 for d in drift)),
      f"{len(drift)} epoch line(s): {sorted(set(drift))}" if drift else "diagnostic not present")
check("M2c: same train/val/test images as M2 at the same seed", splits(logc) == splits(log_m2) and len(splits(logc)) == 3)
for ln in logc.splitlines():
    if "[V16]" in ln or "VERIFIED" in ln:
        print("      | " + ln[:200])

print("\n" + "=" * 78)
if FAIL:
    print(f"VERIFY FAILED ({len(FAIL)}):")
    for f in FAIL:
        print("  - " + f)
    print(f"Logs of the dry runs: {TMP}")
    sys.exit(1)
print("ALL CHECKS PASSED")
