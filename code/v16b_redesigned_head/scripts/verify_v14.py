#!/usr/bin/env python3
"""
v14 offline verification. Run from the v14 repo root AFTER patch_v14_lowdata.py,
on the login node (CPU, ~5-10 min):

    python3 scripts/verify_v14.py

Must end in ALL CHECKS PASSED before any v14 job is submitted.

1. MODELS UNCHANGED: M1-M5 at K=5 and K=10, widths 3 and 15, build with bit-identical
   initial weights, outputs, gradients and expected_params vs the pre-patch file
   (train_gadi.py.v135bak). The patch touches no model code; this proves it.
2. SPLITS (no training: V14_SPLIT_ONLY=1), through the real load_dataset():
   - sizes: train = N*K, val = 125*K, test = 200*K at N = 50, 100, 250, 500
   - every class present in train at N=50 (min count printed)
   - no image in two splits
   - test set identical across every N and seed (fixed test set)
   - same seed + same N twice -> identical split (determinism)
   - different seeds -> different train sets
   Reported only (not pass/fail): train per-class spread, whether val changes
   with N, whether the N=50 train set is nested inside N=500's.
3. AUDIT IS INVISIBLE TO TRAINING: tiny dry run M3 (w15), K=5 A, N=50/class, audit ON
   vs OFF: test_acc, best/stop epoch and per-image predictions must be identical.
4. M5 DRY RUN at N=50/class: CLI ok, freeze banner, exit self-check UNCHANGED OK,
   warmup min_epochs=30 (same as M3), [V14] SPLIT lines printed.
"""
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
from importlib.machinery import SourceFileLoader

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
FAIL = []
TMP = tempfile.mkdtemp(prefix="v14verify_")


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


def header(s):
    print("\n" + "=" * 72 + "\n" + s + "\n" + "=" * 72)


import train_gadi as T
bak = os.path.join(ROOT, "train_gadi.py.v135bak")
if not os.path.exists(bak):
    sys.exit("train_gadi.py.v135bak not found: run patch_v14_lowdata.py first (setup_v14.sh does).")
_ld = SourceFileLoader("train_gadi_v135bak", bak)
spec = importlib.util.spec_from_loader("train_gadi_v135bak", _ld)
TB = importlib.util.module_from_spec(spec)
_ld.exec_module(TB)
SRC = open("train_gadi.py").read()
check("v14 patch marker present", "V14_SPLIT_AUDIT" in SRC)
# v14 scales --max_epochs with N (cap = 200*500/N). That is only a pure cap if
# nothing else in training depends on max_epochs (e.g. an LR schedule).
sched = [l.strip() for l in SRC.splitlines()
         if re.search(r"lr_scheduler|CosineAnneal|OneCycle|StepLR|LambdaLR|T_max|ReduceLROnPlateau", l)
         and not l.strip().startswith("#")]
check("no LR scheduler in train_gadi.py (so a larger --max_epochs is only a cap)", not sched,
      "found: " + " | ".join(sched[:4]) if sched else "")
uses = [l.strip() for l in SRC.splitlines() if re.search(r"max_epochs", l, re.I) and not l.strip().startswith("#")]
print("    every line using max_epochs (check none but the loop bound / argparse):")
for l in uses[:12]:
    print("      " + l[:110])


def setcell(mod, K, w):
    mod.N_CLASSES, mod.N_OBS, mod.HEAD_VARIANT = K, w, "linear"


def build(mod, m, seed=42):
    torch.manual_seed(seed)
    return mod.build_model(m, seed=seed)


def sd_equal(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


# ---------------------------------------------------------------------------
header("1. M1-M5 UNCHANGED vs pre-patch train_gadi.py (bit-exact, widths 3 and 15)")
torch.manual_seed(0)
XB = torch.rand(8, 1, 28, 28)
PTAB = []
for K, W in ((5, 3), (5, 15), (10, 3), (10, 15)):
    setcell(T, K, W); setcell(TB, K, W)
    for m in ("M1", "M2", "M3", "M4", "M5"):
        a, b = build(T, m), build(TB, m)
        y = torch.randint(0, K, (8,))
        oa, ob = a(XB), b(XB)
        torch.nn.functional.cross_entropy(oa, y).backward()
        torch.nn.functional.cross_entropy(ob, y).backward()
        same_grad = all((x is None and z is None) or (x is not None and z is not None and torch.equal(x, z))
                        for x, z in zip([p.grad for p in a.parameters()], [p.grad for p in b.parameters()]))
        check(f"K={K} w={W} {m}: init, output and gradients identical", sd_equal(a, b) and torch.equal(oa, ob) and same_grad)
        ep = T.expected_params(m, K, W)
        check(f"K={K} w={W} {m}: expected_params unchanged", ep == TB.expected_params(m, K, W), str(ep))
        if W == 15:
            tot = a.total_params()[0]
            check(f"K={K} w=15 {m}: builds at w15 and total_params == expected ({tot})", tot == ep)
            PTAB.append((K, m, tot))
print("\n  w15 parameter table (record in MNIST_LOG.md; v13 table: K=5 1680/1686/1725/1719, K=10 1760/1766/1805/1799):")
for K in (5, 10):
    print(f"    K={K}: " + "  ".join(f"{m}={t}" for k, m, t in PTAB if k == K))

# ---------------------------------------------------------------------------
ENV = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8", CUDA_VISIBLE_DEVICES="", PYTHONPATH="")
CLS = {5: "1,3,5,8,9", 10: "0,1,2,3,4,5,6,7,8,9"}


def run(model, K, N, seed, extra_env=None, max_epochs=1, tag=""):
    out_dir = tempfile.mkdtemp(prefix=f"run_{model}_{K}_{N}_{seed}_{tag}_", dir=TMP)
    cmd = [sys.executable, "train_gadi.py", "--model", model, "--dataset", "mnist",
           "--classes", CLS[K], "--seed", str(seed), "--noise_alpha", "0.0",
           "--n_obs", "15", "--head_variant", "linear", "--loss", "ce", "--deterministic",
           "--save_predictions",
           "--n_train", str(N * K), "--n_val", str(125 * K), "--n_test", str(200 * K),
           "--max_epochs", str(max_epochs), "--patience", "1",
           "--output_dir", out_dir, "--data_root", "./data",
           "--checkpoint_dir", out_dir + "_ckpt", "--time_budget_hours", "0.25"]
    r = subprocess.run(cmd, capture_output=True, text=True, env=dict(ENV, **(extra_env or {})), timeout=3600)
    return r, r.stdout + r.stderr, out_dir


def split(K, N, seed):
    dump = os.path.join(TMP, f"split_k{K}_n{N}_s{seed}_{len(os.listdir(TMP))}.json")
    r, log, _ = run("M4", K, N, seed, {"V14_SPLIT_ONLY": "1", "V14_SPLIT_DUMP": dump}, tag="split")
    if r.returncode != 0 or not os.path.exists(dump):
        print("    last lines:\n      " + "\n      ".join(log.strip().splitlines()[-15:]))
        return None, log
    return json.load(open(dump)), log


header("2. SPLITS through the real load_dataset() (no training)")
S = {}
plan = [(5, N, s) for N in (50, 100, 250, 500) for s in (42, 43)] + [(5, 50, 42), (10, 50, 42), (10, 500, 42)]
for i, (K, N, s) in enumerate(plan):
    d, log = split(K, N, s)
    key = (K, N, s, "again") if (K, N, s) in S else (K, N, s)
    check(f"K={K} N={N} seed={s}: split-only run succeeded{' (repeat)' if key[-1] == 'again' else ''}", d is not None)
    if d is None:
        continue
    S[key] = d
    if key[-1] == "again":
        continue
    n = {k: len(d[k]["labels"]) for k in ("train", "val", "test")}
    check(f"K={K} N={N} seed={s}: sizes train/val/test = {N*K}/{125*K}/{200*K}",
          n == {"train": N * K, "val": 125 * K, "test": 200 * K}, str(n))
    labs = sorted(set(d["test"]["labels"]))
    check(f"K={K} N={N} seed={s}: test has {K} labels", len(labs) == K, str(labs))
    pc = [d["train"]["labels"].count(c) for c in labs]
    check(f"K={K} N={N} seed={s}: all {K} classes in train", len(set(d["train"]["labels"])) == K and min(pc) > 0,
          f"per-class {pc}")
    tr, va, te = (set(d[k]["hashes"]) for k in ("train", "val", "test"))
    check(f"K={K} N={N} seed={s}: no image in two splits", not (tr & va or tr & te or va & te),
          f"t&v={len(tr & va)} t&te={len(tr & te)} v&te={len(va & te)}")
    print(f"    train per-class {pc}  (min {min(pc)}, max {max(pc)})")
    check(f"K={K} N={N} seed={s}: [V14] SPLIT lines printed", log.count("[V14] SPLIT ") >= 4)

hs = lambda d, k: sorted(d[k]["hashes"])
for K in (5, 10):
    tests = {k: hs(v, "test") for k, v in S.items() if k[0] == K}
    check(f"K={K}: test set identical across every N and seed ({len(tests)} splits)",
          len({tuple(v) for v in tests.values()}) == 1)
if (5, 50, 42, "again") in S:
    a, b = S[(5, 50, 42)], S[(5, 50, 42, "again")]
    check("K=5 N=50 seed 42 twice: identical train/val/test (order included)",
          all(a[k] == b[k] for k in ("train", "val", "test")))
for N in (50, 100, 250, 500):
    if (5, N, 42) in S and (5, N, 43) in S:
        check(f"K=5 N={N}: seeds 42 and 43 draw different train sets",
              hs(S[(5, N, 42)], "train") != hs(S[(5, N, 43)], "train"))
print("\n  Reported only (protocol facts for the log, not pass/fail):")
if (5, 50, 42) in S and (5, 500, 42) in S:
    a, b = S[(5, 50, 42)], S[(5, 500, 42)]
    print(f"    val identical at N=50 and N=500 (seed 42): {hs(a, 'val') == hs(b, 'val')}")
    nest = len(set(a["train"]["hashes"]) & set(b["train"]["hashes"]))
    print(f"    N=50 train images also in N=500 train (seed 42): {nest}/{len(a['train']['hashes'])}")
    print(f"    N=50 train images in N=500 VAL (seed 42): "
          f"{len(set(a['train']['hashes']) & set(b['val']['hashes']))}")

# ---------------------------------------------------------------------------
header("3. AUDIT IS INVISIBLE TO TRAINING: M3, K=5 A, N=50, audit ON vs OFF (2 epochs)")


def result(out_dir, model):
    f = [x for x in os.listdir(out_dir) if x.startswith(f"result_{model}_") and x.endswith(".json")]
    return json.load(open(os.path.join(out_dir, f[0]))) if f else None


res = {}
for tag, env in (("on", {}), ("off", {"V14_NO_AUDIT": "1"})):
    r, log, od = run("M3", 5, 50, 42, env, max_epochs=2, tag=tag)
    open(f"verify_v14_dry_M3_{tag}.log", "w").write(log)
    check(f"dry run M3 audit {tag}: exit code 0", r.returncode == 0, f"rc={r.returncode} (verify_v14_dry_M3_{tag}.log)")
    if r.returncode != 0:
        print("    last lines:\n      " + "\n      ".join(log.strip().splitlines()[-15:]))
    res[tag] = result(od, "M3")
    if tag == "on":
        check("dry run M3 audit on: [V14] SPLIT lines printed", log.count("[V14] SPLIT ") >= 4)
        check("dry run M3: warmup min_epochs=30", "[V135] warmup: model=M3 min_epochs=30" in log)
if res.get("on") and res.get("off"):
    a, b = res["on"], res["off"]
    for k in ("test_acc", "best_epoch", "stop_epoch"):
        check(f"audit on == off: {k}", a.get(k) == b.get(k), f"{a.get(k)} vs {b.get(k)}")
    pk = next((k for k in ("predictions", "test_predictions", "preds") if k in a and k in b), None)
    if pk:
        check(f"audit on == off: all {len(a[pk])} per-image predictions", a[pk] == b[pk])
    else:
        print("  (no per-image predictions key in the result JSON; matched on scalars only)")

# ---------------------------------------------------------------------------
header("4. M5 DRY RUN at N=50/class, w15 (K=5 A, 2 epochs) + M1 at w15 (never run before)")
r, log, od = run("M5", 5, 50, 42, {}, max_epochs=2, tag="m5")
open("verify_v14_dry_M5.log", "w").write(log)
check("dry run M5: exit code 0", r.returncode == 0, f"rc={r.returncode} (verify_v14_dry_M5.log)")
check("dry run M5: freeze banner", "[V135] M5 = M3 with front-end frozen" in log)
check("dry run M5: exit self-check UNCHANGED OK", "M5 front-end after run: UNCHANGED OK" in log and "CHANGED -- FAIL" not in log)
check("dry run M5: warmup min_epochs=30 (same as M3)", "[V135] warmup: model=M5 min_epochs=30" in log)
check("dry run M5: [V14] SPLIT lines printed", log.count("[V14] SPLIT ") >= 4)
check("dry run M5: parameter VERIFIED OK", bool(re.search(r"VERIFIED.*OK", log)))
check("dry run M5: result JSON written", result(od, "M5") is not None)
for l in log.splitlines():
    if l.startswith("[V14] SPLIT"):
        print("    " + l)
r, log, od = run("M1", 5, 50, 42, {}, max_epochs=2, tag="m1")
open("verify_v14_dry_M1.log", "w").write(log)
check("dry run M1 at w15: exit code 0", r.returncode == 0, f"rc={r.returncode} (verify_v14_dry_M1.log)")
check("dry run M1 at w15: [MULTICLASS] n_obs=15", bool(re.search(r"\[MULTICLASS\].*n_obs=15 ", log)))
check("dry run M1 at w15: warmup min_epochs=30", "[V135] warmup: model=M1 min_epochs=30" in log)
if r.returncode != 0:
    print("    last lines:\n      " + "\n      ".join(log.strip().splitlines()[-15:]))

header("ALL CHECKS PASSED" if not FAIL else f"{len(FAIL)} FAILURE(S): " + "; ".join(FAIL))
sys.exit(1 if FAIL else 0)
