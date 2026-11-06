#!/usr/bin/env python3
"""
v13.5 offline verification. Run from the repo root AFTER patch_v135_m5.py,
on the login node (CPU, a few minutes, two tiny training dry runs):

    python3 scripts/verify_v135_m5.py
    python3 scripts/verify_v135_m5.py --no_dry_run     # skip section 4

Must end in ALL CHECKS PASSED before any v13.5 job is submitted.

1. M1-M4 UNCHANGED: every model at every cell used by v13.5 builds with
   bit-identical initial weights, outputs and gradients vs the pre-patch file
   (train_gadi.py.v13bak). So existing v12/v13 results stay comparable.
2. M5 STRUCTURE: same architecture and same initial weights as M3 (same seed);
   front-end params frozen, everything else trainable; parameter accounting.
3. M5 TRAINING STEP: after AdamW steps the front-end is bit-identical and the
   rest has moved; front-end receives no gradient.
4. DRY RUN through train_gadi.py main(): M3 and M5, tiny scale, CPU. Checks the
   CLI accepts M5, the run finishes, the exit self-check prints UNCHANGED OK,
   and M5 resolves the SAME min-epochs as M3.
"""
import argparse
import importlib.util
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
ap = argparse.ArgumentParser()
ap.add_argument("--no_dry_run", action="store_true")
args = ap.parse_args()

FAIL = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


def header(s):
    print("\n" + "=" * 72 + "\n" + s + "\n" + "=" * 72)


import train_gadi as T
bak = os.path.join(ROOT, "train_gadi.py.v13bak")
if not os.path.exists(bak):
    sys.exit("train_gadi.py.v13bak not found: run patch_v135_m5.py first (setup_v135.sh does).")
_ld = SourceFileLoader("train_gadi_v13bak", bak)
spec = importlib.util.spec_from_loader("train_gadi_v13bak", _ld)
TB = importlib.util.module_from_spec(spec)
_ld.exec_module(TB)

check("patch marker present", "V135_M5_PATCH" in open(os.path.join(ROOT, "train_gadi.py")).read())

CELLS = [(3, 3), (4, 3), (5, 3), (5, 6), (5, 15), (10, 3), (10, 6), (10, 15)]   # (K, n_obs)


def setcell(mod, K, w):
    mod.N_CLASSES, mod.N_OBS, mod.HEAD_VARIANT = K, w, "linear"


def build(mod, m, seed=42):
    torch.manual_seed(seed)                     # same RNG state whether or not build_model seeds itself
    return mod.build_model(m, seed=seed)


def sd_equal(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


torch.manual_seed(0)
XB = torch.rand(8, 1, 28, 28)

# ---------------------------------------------------------------------------
header("1. M1-M4 UNCHANGED vs pre-patch train_gadi.py (bit-exact)")
for K, w in CELLS:
    setcell(T, K, w); setcell(TB, K, w)
    for m in ("M1", "M2", "M3", "M4"):
        if w != 3 and m in ("M1", "M4"):
            continue                            # not run at w6/w15 in v13
        a, b = build(T, m), build(TB, m)
        same_init = sd_equal(a, b)
        y = torch.randint(0, K, (8,))
        oa, ob = a(XB), b(XB)
        torch.nn.functional.cross_entropy(oa, y).backward()
        torch.nn.functional.cross_entropy(ob, y).backward()
        ga = [p.grad for p in a.parameters()]; gb = [p.grad for p in b.parameters()]
        same_grad = all((x is None and z is None) or (x is not None and z is not None and torch.equal(x, z))
                        for x, z in zip(ga, gb))
        check(f"K={K} w={w} {m}: init, output and gradients identical",
              same_init and torch.equal(oa, ob) and same_grad)
        check(f"K={K} w={w} {m}: expected_params unchanged",
              T.expected_params(m, K, w) == TB.expected_params(m, K, w))

# ---------------------------------------------------------------------------
header("2. M5 STRUCTURE (vs M3, same seed)")
tab = []
for K, w in CELLS:
    setcell(T, K, w)
    m3, m5 = build(T, "M3"), build(T, "M5")
    check(f"K={K} w={w}: M5 initial weights == M3 initial weights (all tensors)", sd_equal(m3, m5))
    fe_names = {f"front_end.{n}" for n, _ in m5.front_end.named_parameters()}
    fe_n = sum(p.numel() for p in m5.front_end.parameters())
    frozen = {n for n, p in m5.named_parameters() if not p.requires_grad}
    check(f"K={K} w={w}: exactly the front-end is frozen ({fe_n} params)",
          frozen == fe_names and fe_n > 0, f"frozen={sorted(frozen)}")
    check(f"K={K} w={w}: M5 front-end is M3's type ({type(m3.front_end).__name__})",
          type(m5.front_end) is type(m3.front_end))
    tot5, parts5 = m5.total_params(); tot3, parts3 = m3.total_params()
    train5 = sum(p.numel() for p in m5.parameters() if p.requires_grad)
    exp5 = T.expected_params("M5", K, w)
    check(f"K={K} w={w}: in-run VERIFIED check will pass (total_params {tot5} == expected {exp5})",
          tot5 == exp5,
          "" if tot5 == exp5 else "total_params() excludes frozen tensors: expected_params wrapper must subtract them. Paste this.")
    tab.append((K, w, tot3, tot5, train5, parts5))
print("\n  Parameter table (record in MNIST_LOG.md):")
print(f"  {'K':>3} {'w':>3} {'M3 total':>9} {'M5 total':>9} {'M5 trainable':>13}  M5 (front, backbone, back)")
for K, w, t3, t5, tr5, p5 in tab:
    print(f"  {K:>3} {w:>3} {t3:>9} {t5:>9} {tr5:>13}  {p5}")

# ---------------------------------------------------------------------------
header("3. M5 TRAINING STEP: front-end never moves")
for K, w in ((5, 3), (10, 15)):
    setcell(T, K, w)
    m5 = build(T, "M5")
    fe0 = {k: v.clone() for k, v in m5.front_end.state_dict().items()}
    rest0 = {n: p.detach().clone() for n, p in m5.named_parameters() if p.requires_grad}
    opt = torch.optim.AdamW(m5.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(3):
        x = torch.rand(16, 1, 28, 28); y = torch.randint(0, K, (16,))
        opt.zero_grad(); torch.nn.functional.cross_entropy(m5(x), y).backward()
        check(f"K={K} w={w}: front-end has no gradient", all(p.grad is None for p in m5.front_end.parameters()))
        opt.step()
    check(f"K={K} w={w}: front-end bit-identical after 3 AdamW steps (incl. weight decay)",
          all(torch.equal(m5.front_end.state_dict()[k], v) for k, v in fe0.items()))
    moved = [n for n, p in m5.named_parameters() if p.requires_grad and not torch.equal(p.detach(), rest0[n])]
    check(f"K={K} w={w}: every trainable tensor moved", len(moved) == len(rest0), f"{len(moved)}/{len(rest0)}")

# ---------------------------------------------------------------------------
if args.no_dry_run:
    print("\n(--no_dry_run: skipping section 4. The smoke jobs cover it at full scale.)")
else:
    header("4. DRY RUN through train_gadi.py (tiny, CPU): M3 vs M5, K=5 subset A, w3")
    env = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8", CUDA_VISIBLE_DEVICES="", PYTHONPATH="")
    mins = {}
    for m in ("M3", "M5"):
        out_dir = tempfile.mkdtemp(prefix=f"v135dry_{m}_")
        cmd = [sys.executable, "train_gadi.py", "--model", m, "--dataset", "mnist",
               "--classes", "1,3,5,8,9", "--seed", "42", "--noise_alpha", "0.0",
               "--n_obs", "3", "--head_variant", "linear", "--loss", "ce", "--deterministic",
               "--n_train", "100", "--n_val", "50", "--n_test", "50",
               "--max_epochs", "2", "--patience", "1",
               "--output_dir", out_dir, "--data_root", "./data",
               "--checkpoint_dir", out_dir + "_ckpt", "--time_budget_hours", "0.25"]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=1800)
        log = r.stdout + r.stderr
        open(f"verify_v135_dry_{m}.log", "w").write(log)
        check(f"dry run {m}: exit code 0", r.returncode == 0, f"rc={r.returncode}  (see verify_v135_dry_{m}.log)")
        if r.returncode != 0:
            print("    last lines:\n      " + "\n      ".join(log.strip().splitlines()[-15:]))
        hits = re.findall(r"min_?epochs?\w*\s*[=:]\s*(\d+)", log, re.I)
        mins[m] = sorted(set(hits))
        lines = [l.strip() for l in log.splitlines() if re.search(r"min_?epoch", l, re.I)]
        print(f"    {m} min-epochs lines: " + (" | ".join(lines[:3]) if lines else "(none printed)"))
        if m == "M5":
            check("dry run M5: freeze banner printed", "[V135] M5 = M3 with front-end frozen" in log)
            check("dry run M5: exit self-check UNCHANGED OK", "M5 front-end after run: UNCHANGED OK" in log
                  and "CHANGED -- FAIL" not in log)
            check("dry run M5: parameter VERIFIED OK", bool(re.search(r"VERIFIED.*OK", log)))
            check("dry run M5: result JSON written as M5",
                  any(f.startswith("result_M5_") for f in os.listdir(out_dir)), str(os.listdir(out_dir)))
    if mins.get("M3") or mins.get("M5"):
        check("M5 resolves the same min-epochs as M3", mins["M3"] == mins["M5"], f"M3={mins['M3']} M5={mins['M5']}")
    else:
        print("  (no min-epochs value printed by train_gadi; the smoke gate compares the PROTOCOL lines instead)")

header("ALL CHECKS PASSED" if not FAIL else f"{len(FAIL)} FAILURE(S): " + "; ".join(FAIL))
sys.exit(1 if FAIL else 0)
