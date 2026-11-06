#!/usr/bin/env python3
"""
v15 offline verification. Run from the v15 repo root AFTER patch_v15_noise.py,
on the login node (CPU, ~5-10 min), in the qcqcnn env:

    python3 -u scripts/verify_v15.py | tee verify_v15.log

Must end in ALL CHECKS PASSED before any v15 job is submitted.

1. MODELS UNCHANGED with V15 inactive: M1-M5 at K=5 and K=10, width 15, build with
   bit-identical initial weights, outputs, gradients and expected_params vs the
   pre-patch file (train_gadi.py.v14bak).
2. NOISE MODEL vs QISKIT AER (scripts/verify_noise_v15.py): native channels equal
   NoiseModel.from_backend(FakeBrisbane)'s; Circuit 4 and a 4-qubit M1-gate-set
   circuit equal Aer density-matrix simulation at several alphas; gap to a full
   transpiled device simulation reported.
3. INSTALL (V15 active, K=5, w15), every model:
   - building with noise installed leaves weights, parameter count and every RNG
     state (python, numpy, torch) identical to the pre-patch build
   - alpha=0: forward bit-identical to the un-installed model
   - alpha=1: output changes for M1/M2/M3/M5, unchanged for M4; gradients reach the
     Circuit-4 weights (M2, M3) and stay off them for M5 (frozen)
   - deepcopy of an installed model gives the same output (best-state copies work)
   - alpha effect size table (mean |delta <Z>| of the front-end output)
4. DRY RUNS (tiny, CPU, real data, K=5 A, 50/class, 2 epochs):
   - M3 with V15 off vs V15_ALPHA=0: identical test_acc, epochs and predictions
     (the patch, the install self-checks and the cross-evaluation are invisible
     to training)
   - M3, M1, M5 at alpha=1 and M4 at alpha=0: cross-evaluation self-check MATCH,
     sidecar written with every evaluation alpha; M5 freeze self-check UNCHANGED OK
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
TMP = tempfile.mkdtemp(prefix="v15verify_")
CALIB = os.path.join(ROOT, "calib", "ibm_brisbane_calib.json")
EVALS = "0 0.5 1 2 4 8 16"


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""), flush=True)
    if not cond:
        FAIL.append(name)


def header(s):
    print("\n" + "=" * 76 + "\n" + s + "\n" + "=" * 76, flush=True)


import train_gadi as T  # noqa: E402
bak = os.path.join(ROOT, "train_gadi.py.v14bak")
if not os.path.exists(bak):
    sys.exit("train_gadi.py.v14bak not found: run patch_v15_noise.py first (setup_v15.sh does).")
_ld = SourceFileLoader("train_gadi_v14bak", bak)
spec = importlib.util.spec_from_loader("train_gadi_v14bak", _ld)
TB = importlib.util.module_from_spec(spec)
_ld.exec_module(TB)
check("v15 patch marker present", "V15_FE_NOISE" in open("train_gadi.py").read())
check("V15 inactive in this process (no V15_ALPHA)", T._V15["active"] is False)
import quanv_noise_v15 as QN  # noqa: E402
cal = QN.load_calib(CALIB)
print(f"  calibration: {cal['backend']}  snapshot {cal['last_update_date']}  sha256 {cal['sha256'][:12]}")


def setcell(mod, K, w):
    mod.N_CLASSES, mod.N_OBS, mod.HEAD_VARIANT = K, w, "linear"


def build(mod, m, seed=42):
    torch.manual_seed(seed)
    return mod.build_model(m, seed=seed)


def sd_equal(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def nparams(m):
    return sum(p.numel() for p in m.parameters())


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
header("1. M1-M5 UNCHANGED vs pre-patch train_gadi.py, V15 inactive (bit-exact, w15)")
for K in (5, 10):
    setcell(T, K, 15); setcell(TB, K, 15)
    for m in MODELS:
        a, b = build(T, m), build(TB, m)
        x = xin()
        oa, ga = fwd_grad(a, x)
        ob, gb = fwd_grad(b, x)
        same_grad = ga.keys() == gb.keys() and all(
            (ga[k] is None and gb[k] is None) or (ga[k] is not None and gb[k] is not None and torch.equal(ga[k], gb[k]))
            for k in ga)
        check(f"K={K} {m}: init, output and gradients identical", sd_equal(a, b) and torch.equal(oa, ob) and same_grad)
        ep, epb = T.expected_params(m, K, 15), TB.expected_params(m, K, 15)
        check(f"K={K} {m}: expected_params unchanged", ep == epb, str(ep))

# ---------------------------------------------------------------------------
header("2. NOISE MODEL vs QISKIT AER")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import verify_noise_v15 as VN  # noqa: E402
gadi_ref = os.path.join(ROOT, "calib", "aer_reference_v15_gadi.npz")
pkg_ref = os.path.join(ROOT, "calib", "aer_reference_v15.npz")
ref = gadi_ref if os.path.exists(gadi_ref) else pkg_ref
print(f"  using {'the Aer reference regenerated on this machine' if ref == gadi_ref else 'the packaged Aer reference'}")
VN.FAIL = []
VN.run(ref, CALIB)
for f in VN.FAIL:
    FAIL.append("noise: " + f)
if ref == gadi_ref:
    VN.FAIL = []
    print("  (packaged reference too, for the record:)")
    VN.run(pkg_ref, CALIB)
    for f in VN.FAIL:
        FAIL.append("noise(packaged): " + f)

# ---------------------------------------------------------------------------
header("3. NOISE INSTALL (V15 active), K=5, w15")
setcell(T, 5, 15); setcell(TB, 5, 15)
os.environ["V15_CALIB"] = CALIB
T._V15["active"] = True
T._V15["registered"] = True            # no exit hook in this process
x = xin(3, seed=5)
effect_rows = []
for m in MODELS:
    os.environ["V15_ALPHA"] = "1"
    torch.manual_seed(42); np.random.seed(0); random.seed(0)
    ref_m = TB.build_model(m, seed=42)
    st_ref = rng_state()
    torch.manual_seed(42); np.random.seed(0); random.seed(0)
    mod = T.build_model(m, seed=42)
    st_new = rng_state()
    nz = T._V15["noise"]
    check(f"{m}: install leaves weights identical", sd_equal(mod, ref_m))
    check(f"{m}: install leaves parameter count identical", nparams(mod) == nparams(ref_m), str(nparams(mod)))
    check(f"{m}: install leaves python/numpy/torch RNG states identical", rng_equal(st_ref, st_new))
    with torch.no_grad():
        o_ref = ref_m(x)
        if nz is not None:
            nz.set_alpha(0.0)
        o0 = mod(x)
        check(f"{m}: alpha=0 output bit-identical to un-installed model", torch.equal(o0, o_ref))
        if nz is not None:
            nz.set_alpha(1.0)
        o1 = mod(x)
    changed = not torch.equal(o1, o_ref)
    check(f"{m}: alpha=1 output {'changes' if m != 'M4' else 'unchanged (no quantum front-end)'}",
          changed if m != "M4" else not changed, f"max|d logits|={(o1 - o_ref).abs().max().item():.3e}")
    if m in ("M2", "M3", "M5"):
        mod.zero_grad()
        mod(x).pow(2).sum().backward()
        g = mod.front_end.weights.grad
        if m == "M5":
            check("M5: front-end stays frozen under noise (no gradient, requires_grad False)",
                  (g is None or torch.count_nonzero(g) == 0) and not mod.front_end.weights.requires_grad)
        else:
            check(f"{m}: gradients reach the Circuit-4 weights under noise",
                  g is not None and torch.isfinite(g).all().item() and torch.count_nonzero(g).item() > 0,
                  f"|grad|={g.norm().item():.3e}" if g is not None else "None")
    with torch.no_grad():
        dc = copy.deepcopy(mod)
        check(f"{m}: deepcopy of the installed model gives the same output", torch.equal(dc(x), mod(x)))
    if nz is not None:
        fe = nz.fe
        with torch.no_grad():
            nz.set_alpha(0.0); z0 = fe(x)
            row = [m]
            for al in (0.5, 1, 2, 4, 8, 16):
                nz.set_alpha(al); row.append((fe(x) - z0).abs().mean().item())
            nz.set_alpha(1.0)
        effect_rows.append(row)
os.environ.pop("V15_ALPHA", None)
T._V15["active"] = False
print("\n  Front-end noise effect, mean |delta <Z>| per output (input in [0,1], record in the log):")
print("    model   a=0.5    a=1      a=2      a=4      a=8      a=16")
for r in effect_rows:
    print(f"    {r[0]:<6} " + " ".join(f"{v:.4f}  " for v in r[1:]))

# ---------------------------------------------------------------------------
header("4. DRY RUNS (K=5 A, 50/class, 2 epochs, CPU)")


def dry(m, alpha, tag, extra_env=None):
    out = os.path.join(TMP, f"out_{tag}")
    env = dict(os.environ)
    for k in ("V15_ALPHA", "V15_CALIB", "V15_EVAL_ALPHAS"):
        env.pop(k, None)
    if alpha is not None:
        env.update(V15_ALPHA=str(alpha), V15_CALIB=CALIB, V15_EVAL_ALPHAS=EVALS)
    env.update(extra_env or {})
    cmd = [sys.executable, "train_gadi.py", "--model", m, "--dataset", "mnist", "--classes", "1,3,5,8,9",
           "--seed", "42", "--noise_alpha", "0.0", "--n_obs", "15", "--head_variant", "linear", "--loss", "ce",
           "--deterministic", "--save_predictions", "--n_train", "250", "--n_val", "625", "--n_test", "1000",
           "--max_epochs", "2", "--patience", "20", "--output_dir", out, "--data_root", "./data",
           "--checkpoint_dir", os.path.join(TMP, f"ckpt_{tag}")]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, env=env)
    log = p.stdout
    open(os.path.join(TMP, f"log_{tag}.txt"), "w").write(log)
    js = [f for f in os.listdir(out) if f.startswith("result_")] if os.path.isdir(out) else []
    j = json.load(open(os.path.join(out, js[0]))) if js else {}
    sc = os.path.join(out, f"v15_xeval_{m}_seed42.json")
    side = json.load(open(sc)) if os.path.exists(sc) else None
    if p.returncode != 0:
        print(log[-3000:])
    return p.returncode, log, j, side


def pk(j):
    return next((k for k in ("predictions", "test_predictions", "preds") if k in j), None)


rc0, log0, j0, s0 = dry("M3", None, "m3_off")
rcA, logA, jA, sA = dry("M3", 0, "m3_a0")
check("M3 dry run, V15 off: rc=0", rc0 == 0)
check("M3 dry run, V15_ALPHA=0: rc=0", rcA == 0)
same = all(j0.get(k) == jA.get(k) for k in ("test_acc", "best_epoch", "stop_epoch")) and pk(j0) and j0[pk(j0)] == jA.get(pk(j0))
check("V15 at alpha=0 is invisible to training: test_acc, best/stop epoch, every prediction identical", bool(same),
      f"test_acc {j0.get('test_acc')} vs {jA.get('test_acc')}")
check("M3 alpha=0: cross-eval self-check MATCH", bool(sA) and sA["selfcheck"] == "MATCH",
      (sA or {}).get("source", "no sidecar"))
for m, al in (("M3", 1), ("M1", 1), ("M5", 1), ("M4", 0)):
    rc, log, j, s = dry(m, al, f"{m.lower()}_a{al}")
    check(f"{m} dry run at alpha={al}: rc=0", rc == 0)
    check(f"{m} alpha={al}: [V15] NOISE installed line, alpha and calibration correct",
          re.search(rf"\[V15\] NOISE installed model={m} kind=\S+ alpha={al:g} calib={cal['sha256'][:12]}", log) is not None)
    if m != "M4":
        check(f"{m} alpha={al}: install self-checks PASS", "selfcheck PTM path at alpha=0 vs exact forward" in log
              and "FAIL" not in "\n".join(l for l in log.splitlines() if "[V15]" in l))
    check(f"{m} alpha={al}: cross-eval self-check MATCH, all {len(EVALS.split())} eval alphas present",
          bool(s) and s["selfcheck"] == "MATCH" and sorted(map(float, s["acc"])) == sorted(map(float, EVALS.split())),
          (s or {}).get("source", "no sidecar"))
    if m == "M5":
        check("M5 alpha=1: freeze banner and exit self-check UNCHANGED OK",
              "[V135] M5 = M3 with front-end frozen" in log and "[V135] M5 front-end after run: UNCHANGED OK" in log)
    if s:
        print("      dry-run acc by eval alpha (2 epochs, not a result): " +
              " ".join(f"{k}={v:.3f}" for k, v in s["acc"].items()))

print("\n" + "=" * 76)
if FAIL:
    print(f"VERIFY FAILED ({len(FAIL)}):")
    for f in FAIL:
        print("  - " + f)
    print(f"Logs of the dry runs: {TMP}")
    sys.exit(1)
print("ALL CHECKS PASSED")
