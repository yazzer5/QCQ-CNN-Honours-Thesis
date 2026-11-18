#!/usr/bin/env python3
"""
v15.8 offline verification. Run from the v15.8 repo root AFTER patch_v158_m6.py (setup_v158.sh does):

    python -u scripts/verify_v158.py /g/data/wa66/yh2481/qcq_cnn_v155 | tee verify_v158.log

Must end in ALL CHECKS PASSED before any v15.8 job is submitted.

1. TEXT: train_gadi.py.v155bak == the v15.5 repo's train_gadi.py, byte for byte; the patched
   file minus the V158 block and the one argparse edit == train_gadi.py.v155bak, byte for byte.
2. M1-M5 UNCHANGED vs the v15.5 file (K=5, K=10, w15): initial weights, outputs, gradients,
   global RNG state and expected_params identical, V155 unset and set.
3. M6: parameter count = M4's (pre-registered table); 0 front-end params; front-end equals
   cos(pi * pixel) computed independently AND Circuit 4 with all weights 0; black -> +1,
   white -> -1; the backbone and head start bit-identical to M4's (and M3's) at the same seed,
   and the global RNG ends where M4's build leaves it; every parameter gets a gradient;
   the trainer's warmup rule gives M6 0 epochs (like M4).
4. DRY RUNS (real data, K=5 A, 50/class, 2 epochs, CPU):
   - M3 at sigma=0.3: v15.8 trainer == v15.5 trainer (test_acc, epochs, every prediction,
     every cross-eval prediction). The patch is invisible to M1-M5.
   - M6 at sigma=0.3 twice: identical; cross-eval MATCH; same noisy test-set hashes as M3;
     [V158] self-check PASS; warmup 0. M6 at sigma=0: runs, MATCH.
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

for k in ("V155_SIGMA", "V155_EVAL_SIGMAS", "V155_EVAL_SP", "V14_NO_AUDIT", "V14_SPLIT_ONLY", "V14_SPLIT_DUMP",
          "V15_ALPHA", "V15_CALIB", "V15_EVAL_ALPHAS"):
    os.environ.pop(k, None)

import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else None
sys.path.insert(0, ROOT)
os.chdir(ROOT)
FAIL = []
TMP = tempfile.mkdtemp(prefix="v158verify_")
EVALS = "0 0.1 0.2 0.3 0.5 0.7 1.0"
EVAL_SP = "0.05 0.1 0.2"
PARAMS = {5: dict(M1=1680, M2=1686, M3=1725, M4=1719, M5=1725, M6=1719),
          10: dict(M1=1760, M2=1766, M3=1805, M4=1799, M5=1805, M6=1799)}


def check(name, cond, detail=""):
    print("  %s  %s%s" % ("PASS" if cond else "FAIL", name, ("   " + detail) if detail else ""), flush=True)
    if not cond:
        FAIL.append(name)


def header(s):
    print("\n" + "=" * 76 + "\n" + s + "\n" + "=" * 76, flush=True)


# ---------------------------------------------------------------------------
header("1. TEXT: v15.8 trainer = v15.5 trainer + one block + one argparse edit")
new = open("train_gadi.py").read()
bak_p = os.path.join(ROOT, "train_gadi.py.v155bak")
if not os.path.exists(bak_p):
    sys.exit("train_gadi.py.v155bak not found: run patch_v158_m6.py first (setup_v158.sh does).")
bak = open(bak_p).read()
if SRC:
    check("backup == %s/train_gadi.py byte for byte" % SRC, open(os.path.join(SRC, "train_gadi.py")).read() == bak)
else:
    check("v15.5 repo path given on the command line", False, "usage: verify_v158.py <qcq_cnn_v155>")
blk = re.search(r"# ═+\n# V158_M6_COS .*?# ═══ end V158_M6_COS ═+\n\n", new, re.S)
check("exactly one V158 block", blk is not None and new.count("# V158_M6_COS  (inserted") == 1)
stripped = new.replace(blk.group(0), "", 1) if blk else new
stripped = stripped.replace('choices=list(["M1", "M2", "M3", "M4"]) + ["M5"] + ["M6"])  # V158_M6_COS',
                            'choices=list(["M1", "M2", "M3", "M4"]) + ["M5"])')
check("patched file minus the V158 block and the argparse edit == v15.5 file", stripped == bak,
      "%d vs %d chars" % (len(stripped), len(bak)))
check("V158 block sits above the V155 block (so v15.5's wrap sees M6)",
      bool(blk) and new.index("V158_M6_COS  (inserted") < new.index("V155_INPUT_NOISE  (inserted"))

import train_gadi as T  # noqa: E402
_ld = SourceFileLoader("train_gadi_v155bak", bak_p)
spec = importlib.util.spec_from_loader("train_gadi_v155bak", _ld)
TB = importlib.util.module_from_spec(spec)
_ld.exec_module(TB)
check("V155 inactive in this process", T._v155_sigma() is None)
print("  torch %s, python %s" % (torch.__version__, sys.version.split()[0]))


def setcell(mod, K, w=15):
    mod.N_CLASSES, mod.N_OBS, mod.HEAD_VARIANT = K, w, "linear"


def sd_equal(a, b, prefix=""):
    sa = {k: v for k, v in a.state_dict().items() if k.startswith(prefix)}
    sb = {k: v for k, v in b.state_dict().items() if k.startswith(prefix)}
    return sa.keys() == sb.keys() and len(sa) > 0 and all(torch.equal(sa[k], sb[k]) for k in sa)


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


def seeded_build(mod, m, seed=42):
    T.set_seed(seed); TB.set_seed(seed)
    mdl = mod.build_model(m, seed)
    return mdl, rng_state()


# ---------------------------------------------------------------------------
header("2. M1-M5 UNCHANGED vs the v15.5 trainer (bit-exact, w15), V155 unset and set")
for K in (5, 10):
    setcell(T, K); setcell(TB, K)
    for m in ("M1", "M2", "M3", "M4", "M5"):
        for env in (None, "0.3"):
            if env is None:
                os.environ.pop("V155_SIGMA", None)
            else:
                os.environ["V155_SIGMA"] = env
            a, st_a = seeded_build(T, m)
            b, st_b = seeded_build(TB, m)
            x = xin()
            oa, ga = fwd_grad(a, x)
            ob, gb = fwd_grad(b, x)
            same_grad = ga.keys() == gb.keys() and all(
                (ga[k] is None and gb[k] is None) or (ga[k] is not None and gb[k] is not None and torch.equal(ga[k], gb[k]))
                for k in ga)
            check("K=%d %s (%s): init, output, gradients, RNG identical" % (K, m, "V155 unset" if env is None else "V155_SIGMA=" + env),
                  sd_equal(a, b) and torch.equal(oa, ob) and same_grad and rng_equal(st_a, st_b))
        os.environ.pop("V155_SIGMA", None)
        ep, epb = T.expected_params(m, K, 15), TB.expected_params(m, K, 15)
        check("K=%d %s: expected_params unchanged = %d" % (K, m, PARAMS[K][m]), ep == epb == PARAMS[K][m], str(ep))
T._V155["models"] = []
T._V155["model"] = None

# ---------------------------------------------------------------------------
header("3. M6 (fixed cos(pi x) front-end)")
for K in (5, 10):
    setcell(T, K)
    m6, st6 = seeded_build(T, "M6")
    m4, st4 = seeded_build(T, "M4")
    m3, st3 = seeded_build(T, "M3")
    tot, (fe, bb, be) = m6.total_params()
    check("K=%d M6: total params %d = pre-registered %d (= M4); front-end %d" % (K, tot, PARAMS[K]["M6"], fe),
          tot == PARAMS[K]["M6"] == T.expected_params("M6", K, 15) and fe == 0
          and len(list(m6.front_end.parameters())) == 0 and len(list(m6.front_end.buffers())) == 0)
    check("K=%d M6: backbone and head start bit-identical to M4's at seed 42" % K,
          sd_equal(m6, m4, "backbone.") and sd_equal(m6, m4, "back_end."))
    check("K=%d M6: backbone and head start bit-identical to M3's at seed 42" % K,
          sd_equal(m6, m3, "backbone.") and sd_equal(m6, m3, "back_end."))
    check("K=%d M6: global RNG after build == after M4's build" % K, rng_equal(st6, st4))
    x = xin(4, 1)
    x[0] = 0.0
    x[1] = 1.0
    with torch.no_grad():
        y = m6.front_end(x)
    ind = torch.stack([x[:, 0, 0::2, 0::2], x[:, 0, 0::2, 1::2], x[:, 0, 1::2, 0::2], x[:, 0, 1::2, 1::2]], 1)
    check("K=%d M6 front-end == cos(pi * pixel), pixels taken by strided indexing" % K,
          y.shape == (4, 4, 14, 14) and torch.allclose(y, torch.cos(math.pi * ind), atol=1e-6, rtol=0),
          "max|diff|=%.1e" % (y - torch.cos(math.pi * ind)).abs().max().item())
    c4 = T.TrainableQuanvFrontEndTorch(seed=0)
    with torch.no_grad():
        c4.weights.zero_()
        r = c4(x)
    check("K=%d M6 front-end == Circuit 4 with all 6 weights = 0 (same channel order)" % K,
          torch.allclose(y, r, atol=1e-5, rtol=0), "max|diff|=%.1e" % (y - r).abs().max().item())
    check("K=%d M6: black pixel -> +1, white -> -1" % K, bool((y[0] == 1).all()) and bool((y[1] + 1).abs().max() < 1e-6))
    m6.train(True)
    _, g = fwd_grad(m6, xin(3, 2))
    check("K=%d M6: every parameter receives a nonzero gradient" % K,
          all(v is not None and v.abs().sum() > 0 for v in g.values()), "%d tensors" % len(g))
    check("K=%d M6: no .weights on the front-end (no front-end gradient logging)" % K,
          getattr(m6.front_end, "weights", None) is None)
    nw = "M6" in ("M2", "M3", "M5") or isinstance(m6.back_end, T.QNNBackend)
    check("K=%d M6: trainer's warmup rule gives 0 epochs (like M4)" % K, not nw)
T._V155["models"] = []
T._V155["model"] = None

# ---------------------------------------------------------------------------
header("4. DRY RUNS (K=5 A, 50/class, 2 epochs, CPU, real data)")


def dry(script, m, sigma, tag, seed=42):
    outd = os.path.join(TMP, "out_" + tag)
    env = dict(os.environ)
    for k in ("V155_SIGMA", "V155_EVAL_SIGMAS", "V155_EVAL_SP"):
        env.pop(k, None)
    env.update(V155_SIGMA=str(sigma), V155_EVAL_SIGMAS=EVALS, V155_EVAL_SP=EVAL_SP)
    cmd = [sys.executable, script, "--model", m, "--dataset", "mnist", "--classes", "1,3,5,8,9",
           "--seed", str(seed), "--noise_alpha", "0.0", "--n_obs", "15", "--head_variant", "linear", "--loss", "ce",
           "--deterministic", "--save_predictions", "--n_train", "250", "--n_val", "625", "--n_test", "1000",
           "--max_epochs", "2", "--patience", "20", "--output_dir", outd, "--data_root", "./data",
           "--checkpoint_dir", os.path.join(TMP, "ckpt_" + tag)]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, env=env)
    log = p.stdout
    open(os.path.join(TMP, "log_%s.txt" % tag), "w").write(log)
    js = [f for f in os.listdir(outd) if f.startswith("result_")] if os.path.isdir(outd) else []
    j = json.load(open(os.path.join(outd, js[0]))) if js else {}
    sc = os.path.join(outd, "v155_xeval_%s_seed%d.json" % (m, seed))
    side = json.load(open(sc)) if os.path.exists(sc) else None
    if p.returncode != 0:
        print(log[-3000:])
    return p.returncode, log, j, side


def same_run(j1, j2):
    return all(j1.get(k) == j2.get(k) for k in ("test_acc", "best_epoch", "stop_epoch", "test_predictions")) \
        and j1.get("test_predictions") is not None


rcA, logA, jA, sA = dry("train_gadi.py", "M3", 0.3, "m3_new")
rcB, logB, jB, sB = dry("train_gadi.py.v155bak", "M3", 0.3, "m3_v155")
check("M3 sigma=0.3 on v15.8 and v15.5 trainers: rc=0", rcA == 0 and rcB == 0)
check("M3 sigma=0.3: v15.8 == v15.5 (test_acc, best/stop epoch, every prediction)", same_run(jA, jB),
      "test_acc %s vs %s" % (jA.get("test_acc"), jB.get("test_acc")))
check("M3 sigma=0.3: cross-eval predictions and hashes identical at every level",
      bool(sA) and bool(sB) and sA["predictions"] == sB["predictions"] and sA["test_hash"] == sB["test_hash"]
      and sA["selfcheck"] == sB["selfcheck"] == "MATCH")

r1 = dry("train_gadi.py", "M6", 0.3, "m6_a")
r2 = dry("train_gadi.py", "M6", 0.3, "m6_b")
r0 = dry("train_gadi.py", "M6", 0, "m6_s0")
for (rc, log, j, s), tag in ((r1, "sigma=0.3"), (r0, "sigma=0")):
    check("M6 %s: rc=0" % tag, rc == 0)
    check("M6 %s: [VERIFIED] M6 params 1719 OK and [V158] self-check PASS" % tag,
          re.search(r"\[VERIFIED\] M6 params: front=0 .*total=1719 \(expected 1719\) OK", log) is not None
          and re.search(r"\[V158\] M6 front-end = Circuit 4 at w=0 .* PASS", log) is not None and "[V158] FATAL" not in log)
    check("M6 %s: warmup 0 epochs" % tag, "[V135] warmup: model=M6 min_epochs=0" in log)
    check("M6 %s: gradient check OK, determinism on" % tag,
          "[GRADCHECK] M6: all trainable tensors receive gradient. OK" in log and re.search(r"DETERMINISM.*ENABLED", log, re.I) is not None)
    check("M6 %s: cross-eval MATCH, all levels" % tag,
          bool(s) and s["selfcheck"] == "MATCH" and sorted(map(float, s["acc"])) == sorted(map(float, EVALS.split()))
          and sorted(map(float, s["acc_sp"])) == sorted(map(float, EVAL_SP.split())), (s or {}).get("source", "no sidecar"))
    check("M6 %s: result JSON model=M6, n_params=1719" % tag, j.get("model") == "M6" and j.get("n_params") == 1719)
check("M6 sigma=0.3 run twice: identical", same_run(r1[2], r2[2]) and r1[3]["predictions"] == r2[3]["predictions"])
check("M6 and M3 saw the same noisy test sets at every level (hashes)",
      bool(r1[3]) and bool(sA) and r1[3]["test_hash"] == sA["test_hash"] and r0[3]["test_hash"] == sA["test_hash"])
spl = lambda log: re.findall(r"^\[V14\] SPLIT (train|val|test)\s+n=\d+ .*hash=(\w+)", log, re.M)  # noqa: E731
check("M6 and M3 saw the same clean splits ([V14] hashes)", bool(spl(r1[1])) and spl(r1[1]) == spl(logA))
if r1[3]:
    print("      M6 dry-run acc by eval sigma (2 epochs, not a result): " +
          " ".join("%s=%.3f" % (k, v) for k, v in r1[3]["acc"].items()))

print("\n" + "=" * 76)
if FAIL:
    print("VERIFY FAILED (%d):" % len(FAIL))
    for f in FAIL:
        print("  - " + f)
    print("Dry-run logs: " + TMP)
    sys.exit(1)
print("ALL CHECKS PASSED")
