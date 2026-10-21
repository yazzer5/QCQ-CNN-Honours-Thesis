#!/usr/bin/env python3
"""
v15.5 offline verification. Run from the v15.5 repo root AFTER patch_v155_input_noise.py,
on the login node (CPU, ~5-15 min), in the qcqcnn env:

    python -u scripts/verify_v155.py | tee verify_v155.log

Must end in ALL CHECKS PASSED before any v15.5 job is submitted.

1. MODELS UNCHANGED: M1-M5 at K=5 and K=10, width 15, build with bit-identical initial
   weights, outputs, gradients and expected_params vs the pre-patch file
   (train_gadi.py.v14bak), with V155 unset AND with V155_SIGMA set (the build_model wrap
   only keeps a reference; it must not touch weights or any RNG).
2. NOISE WRAPPER (load_dataset wrap on synthetic _TensorImgDS data, no files needed):
   - V155 unset and V155_SIGMA=0: the very same dataset objects are returned (clean).
   - sigma>0: labels unchanged; images in [0,1]; test and val noise fixed (two calls give
     the same hash) and independent of the seed; test noise differs between sigmas;
     training noise identical for the same (seed, sigma), different for another seed or
     sigma, fresh on every fetch; the global RNGs (python, numpy, torch) end in exactly
     the state the unpatched function leaves them in, also after fetching training images.
   - statistics: on a mid-grey image the pre-clip noise SD equals sigma (2%); on a black
     image the clipped mean equals sigma/sqrt(2*pi) (3%); salt-and-pepper sets a fraction p.
3. DRY RUNS (tiny, CPU, real data, K=5 A, 50/class, 2 epochs):
   - M3 with V155 off vs V155_SIGMA=0: identical test_acc, epochs and every prediction
     (the patch, the self-checks and the cross-evaluation are invisible to training);
   - M3 at sigma=0.3 run twice: identical (training noise deterministic);
   - M3 sigma=0.3 seed 43: same noisy test-set hash as seed 42 (test noise fixed);
   - M1, M4, M5 at sigma=0.3 and M3 at sigma=0.1: cross-evaluation MATCH, every sigma and
     salt-and-pepper level present, test hashes consistent; M5 freeze self-check.
4. Writes noise_examples_v155.npz (8 real test digits at every sigma; for figure D0).
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
sys.path.insert(0, ROOT)
os.chdir(ROOT)
FAIL = []
TMP = tempfile.mkdtemp(prefix="v155verify_")
EVALS = "0 0.1 0.2 0.3 0.5 0.7 1.0"
EVAL_SP = "0.05 0.1 0.2"


def check(name, cond, detail=""):
    print("  %s  %s%s" % ("PASS" if cond else "FAIL", name, ("   " + detail) if detail else ""), flush=True)
    if not cond:
        FAIL.append(name)


def header(s):
    print("\n" + "=" * 76 + "\n" + s + "\n" + "=" * 76, flush=True)


import train_gadi as T  # noqa: E402
bak = os.path.join(ROOT, "train_gadi.py.v14bak")
if not os.path.exists(bak):
    sys.exit("train_gadi.py.v14bak not found: run patch_v155_input_noise.py first (setup_v155.sh does).")
_ld = SourceFileLoader("train_gadi_v14bak", bak)
spec = importlib.util.spec_from_loader("train_gadi_v14bak", _ld)
TB = importlib.util.module_from_spec(spec)
_ld.exec_module(TB)
check("v15.5 patch marker present", "V155_INPUT_NOISE" in open("train_gadi.py").read())
check("v15 circuit-noise patch absent", "V15_FE_NOISE" not in open("train_gadi.py").read())
check("V155 inactive in this process (no V155_SIGMA)", T._v155_sigma() is None)
print("  torch %s, python %s" % (torch.__version__, sys.version.split()[0]))


def setcell(mod, K, w):
    mod.N_CLASSES, mod.N_OBS, mod.HEAD_VARIANT = K, w, "linear"


def build(mod, m, seed=42):
    torch.manual_seed(seed)
    return mod.build_model(m, seed=seed)


def sd_equal(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


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
PARAMS = {5: dict(M1=1680, M2=1686, M3=1725, M4=1719, M5=1725), 10: dict(M1=1760, M2=1766, M3=1805, M4=1799, M5=1805)}

# ---------------------------------------------------------------------------
header("1. M1-M5 UNCHANGED vs pre-patch train_gadi.py (bit-exact, w15), V155 unset and set")
for K in (5, 10):
    setcell(T, K, 15); setcell(TB, K, 15)
    for m in MODELS:
        for env in (None, "0.3"):
            if env is None:
                os.environ.pop("V155_SIGMA", None)
            else:
                os.environ["V155_SIGMA"] = env
            torch.manual_seed(0); np.random.seed(0); random.seed(0)
            a = build(T, m)
            st_a = rng_state()
            torch.manual_seed(0); np.random.seed(0); random.seed(0)
            b = build(TB, m)
            st_b = rng_state()
            x = xin()
            oa, ga = fwd_grad(a, x)
            ob, gb = fwd_grad(b, x)
            same_grad = ga.keys() == gb.keys() and all(
                (ga[k] is None and gb[k] is None) or (ga[k] is not None and gb[k] is not None and torch.equal(ga[k], gb[k]))
                for k in ga)
            tag = "V155 unset" if env is None else "V155_SIGMA=%s" % env
            check("K=%d %s (%s): init, output, gradients and RNG states identical" % (K, m, tag),
                  sd_equal(a, b) and torch.equal(oa, ob) and same_grad and rng_equal(st_a, st_b))
        os.environ.pop("V155_SIGMA", None)
        ep, epb = T.expected_params(m, K, 15), TB.expected_params(m, K, 15)
        check("K=%d %s: expected_params unchanged and = pre-registered table" % (K, m),
              ep == epb == PARAMS[K][m], str(ep))
T._V155["models"] = []
T._V155["model"] = None

# ---------------------------------------------------------------------------
header("2. NOISE WRAPPER on synthetic data (load_dataset wrap; no files)")
rs = np.random.RandomState(7)


def synth(n, k=5):
    imgs = (rs.rand(n, 28, 28) * 255).astype(np.uint8)
    imgs[:, :6, :] = 0                                    # some black background, like MNIST
    return imgs, rs.randint(0, k, n)


TR, VA, TE = synth(250), synth(125), synth(200)


def fake_inner(name, data_root, seed):
    return (T._TensorImgDS(*TR), T._TensorImgDS(*VA), T._TensorImgDS(*TE))


orig_inner = T._v155_load_dataset_orig
T._v155_load_dataset_orig = fake_inner
T._V155["registered"] = True                             # no exit hook in this process


def call(sigma, seed=42):
    if sigma is None:
        os.environ.pop("V155_SIGMA", None)
    else:
        os.environ["V155_SIGMA"] = str(sigma)
    out = T.load_dataset("mnist", "./data", seed)
    os.environ.pop("V155_SIGMA", None)
    return out


def stack(ds, n=None):
    n = len(ds) if n is None else n
    xs, ys = zip(*[ds[i] for i in range(n)])
    return torch.stack(xs), [int(y) for y in ys]


raw = fake_inner("mnist", "./data", 42)
out = call(None)
check("V155 unset: datasets returned as they come (same images)", all(torch.equal(stack(a)[0], stack(b)[0]) for a, b in zip(out, raw)))
out0 = call(0)
check("V155_SIGMA=0: datasets returned untouched (same object types and images)",
      all(type(a) is type(b) and torch.equal(stack(a)[0], stack(b)[0]) for a, b in zip(out0, raw)))

torch.manual_seed(123); np.random.seed(123); random.seed(123)
_ = fake_inner("mnist", "./data", 42)
st_ref = rng_state()
torch.manual_seed(123); np.random.seed(123); random.seed(123)
o3 = call(0.3)
st_new = rng_state()
check("sigma=0.3: global RNG states identical to the unpatched function's", rng_equal(st_ref, st_new))
for i in range(60):
    o3[0][i]
check("sigma=0.3: fetching 60 noisy training images leaves the global RNGs untouched", rng_equal(st_ref, rng_state()))
Xte3, yte3 = stack(o3[2]); Xte_c, yte_c = stack(raw[2])
check("sigma=0.3: test labels unchanged", yte3 == yte_c)
check("sigma=0.3: train/val labels unchanged",
      stack(o3[1])[1] == stack(raw[1])[1] and [int(o3[0][i][1]) for i in range(50)] == [int(raw[0][i][1]) for i in range(50)])
check("sigma=0.3: noisy test images in [0,1], same shape, changed",
      bool(((Xte3 >= 0) & (Xte3 <= 1)).all()) and Xte3.shape == Xte_c.shape and not torch.equal(Xte3, Xte_c))
o3b, o3s43, o1 = call(0.3), call(0.3, seed=43), call(0.1)
check("test noise fixed: two calls give identical noisy test sets", torch.equal(stack(o3b[2])[0], Xte3))
check("test noise independent of the seed (seed 43 == seed 42)", torch.equal(stack(o3s43[2])[0], Xte3))
check("val noise fixed across calls and seeds", torch.equal(stack(o3b[1])[0], stack(o3[1])[0])
      and torch.equal(stack(o3s43[1])[0], stack(o3[1])[0]))
check("test noise differs between sigma=0.1 and 0.3", not torch.equal(stack(o1[2])[0], Xte3))
h1 = T._v155_hash(Xte3, yte3)
check("noisy test hash reproducible", h1 == T._v155_hash(stack(o3b[2])[0], stack(o3b[2])[1]), h1[:16])
a = torch.stack([o3b[0][i][0] for i in range(20)])
fresh = call(0.3)[0]                                     # one new dataset, same (seed, sigma)
b = torch.stack([fresh[i][0] for i in range(20)])
c = torch.stack([o3s43[0][i][0] for i in range(20)])
d = torch.stack([o1[0][i][0] for i in range(20)])
check("training noise identical for the same (seed, sigma)", torch.equal(a, b))
check("training noise differs for another seed and another sigma", not torch.equal(a, c) and not torch.equal(a, d))
again = torch.stack([o3b[0][i][0] for i in range(20)])
check("training noise is fresh on every fetch (second pass over the same images differs)", not torch.equal(a, again))
check("train generator seed = 155000000 + 10000*seed + round(1000*sigma)",
      o3[0].gen_seed == 155000000 + 10000 * 42 + 300 and o3s43[0].gen_seed == 155000000 + 10000 * 43 + 300,
      str(o3[0].gen_seed))
T._v155_load_dataset_orig = orig_inner

grey = torch.full((400, 1, 28, 28), 0.5)
black = torch.zeros(400, 1, 28, 28)
sd = (T._v155_fixed(grey, "gauss", 0.1, 1) - 0.5).std().item()
check("pre-clip noise SD on mid-grey = sigma (0.1) within 2%", abs(sd - 0.1) / 0.1 < 0.02, "%.5f" % sd)
mb = T._v155_fixed(black, "gauss", 0.3, 1).mean().item()
check("clipped mean on black = sigma/sqrt(2 pi) (0.3) within 3%", abs(mb - 0.3 / math.sqrt(2 * math.pi)) / (0.3 / math.sqrt(2 * math.pi)) < 0.03,
      "%.5f vs %.5f" % (mb, 0.3 / math.sqrt(2 * math.pi)))
spg = T._v155_fixed(grey, "sp", 0.1, 1)
frac = (spg != 0.5).float().mean().item()
check("salt-and-pepper p=0.1 sets 10% of pixels (2%), half to 0 and half to 1",
      abs(frac - 0.1) < 0.002 and abs((spg == 0).float().mean().item() - 0.05) < 0.002, "%.4f" % frac)

# ---------------------------------------------------------------------------
header("3. DRY RUNS (K=5 A, 50/class, 2 epochs, CPU, real data)")


def dry(m, sigma, tag, seed=42):
    outd = os.path.join(TMP, "out_" + tag)
    env = dict(os.environ)
    for k in ("V155_SIGMA", "V155_EVAL_SIGMAS", "V155_EVAL_SP"):
        env.pop(k, None)
    if sigma is not None:
        env.update(V155_SIGMA=str(sigma), V155_EVAL_SIGMAS=EVALS, V155_EVAL_SP=EVAL_SP)
    cmd = [sys.executable, "train_gadi.py", "--model", m, "--dataset", "mnist", "--classes", "1,3,5,8,9",
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


def testhash(log):
    m = re.search(r"\[V155\] TESTSET sigma=(\S+) n=(\d+) sha=(\w+)", log)
    return m.group(3) if m else None


rc0, log0, j0, s0 = dry("M3", None, "m3_off")
rcZ, logZ, jZ, sZ = dry("M3", 0, "m3_s0")
check("M3 dry run, V155 off: rc=0", rc0 == 0)
check("M3 dry run, V155_SIGMA=0: rc=0", rcZ == 0)
check("V155 at sigma=0 is invisible to training: test_acc, best/stop epoch, every prediction identical",
      same_run(j0, jZ), "test_acc %s vs %s" % (j0.get("test_acc"), jZ.get("test_acc")))
check("M3 sigma=0: cross-eval self-check MATCH", bool(sZ) and sZ["selfcheck"] == "MATCH", (sZ or {}).get("source", "no sidecar"))
check("M3 sigma=0: global-RNG self-check PASS", "unchanged by noise setup: PASS" in logZ)

runs = {}
for m, sg, tag, seed in (("M3", 0.3, "m3_s03_a", 42), ("M3", 0.3, "m3_s03_b", 42), ("M3", 0.3, "m3_s03_r43", 43),
                         ("M4", 0.3, "m4_s03", 42), ("M5", 0.3, "m5_s03", 42), ("M1", 0.3, "m1_s03", 42),
                         ("M3", 0.1, "m3_s01", 42)):
    rc, log, j, s = dry(m, sg, tag, seed)
    runs[tag] = (rc, log, j, s)
    check("%s sigma=%g seed %d: rc=0" % (m, sg, seed), rc == 0)
    check("%s sigma=%g: [V155] NOISE line with the pre-registered seeds" % (m, sg),
          re.search(r"\[V155\] NOISE sigma=%g model=%s seed=%d .*train_gen_seed=%d val_seed=%d test_seed=%d" % (
              sg, m, seed, 155000000 + 10000 * seed + int(round(1000 * sg)), 15600000 + int(round(1000 * sg)),
              15500000 + int(round(1000 * sg))), log) is not None)
    check("%s sigma=%g: self-checks PASS (images in range, global RNG untouched), no FATAL" % (m, sg),
          "noisy images in [0,1], finite, same shape: PASS" in log and "unchanged by noise setup: PASS" in log
          and "[V155] FATAL" not in log)
    ok_x = bool(s) and s["selfcheck"] == "MATCH" and sorted(map(float, s["acc"])) == sorted(map(float, EVALS.split())) \
        and sorted(map(float, s["acc_sp"])) == sorted(map(float, EVAL_SP.split()))
    check("%s sigma=%g: cross-eval MATCH, all %d sigmas and %d salt-and-pepper levels present" % (
        m, sg, len(EVALS.split()), len(EVAL_SP.split())), ok_x, (s or {}).get("source", "no sidecar"))
    if s:
        check("%s sigma=%g: sidecar hash at its own sigma == the log's TESTSET hash" % (m, sg),
              s["test_hash"].get("g%g" % sg) == testhash(log))
        check("%s sigma=%g: training noise was drawn (train fetches > 0)" % (m, sg), s.get("train_fetches", 0) > 0,
              str(s.get("train_fetches")))
        print("      dry-run acc by eval sigma (2 epochs, not a result): " +
              " ".join("%s=%.3f" % (k, v) for k, v in s["acc"].items()))
    if m == "M5":
        check("M5 sigma=0.3: freeze banner and exit self-check UNCHANGED OK",
              "[V135] M5 = M3 with front-end frozen" in log and "[V135] M5 front-end after run: UNCHANGED OK" in log)

ja, jb = runs["m3_s03_a"][2], runs["m3_s03_b"][2]
check("M3 sigma=0.3 run twice: identical (training noise deterministic)", same_run(ja, jb))
hs = {t: testhash(runs[t][1]) for t in ("m3_s03_a", "m3_s03_r43", "m4_s03", "m5_s03", "m1_s03")}
check("sigma=0.3 noisy test-set hash identical across models and seeds", len(set(hs.values())) == 1 and None not in hs.values(),
      str({k: (v or "")[:12] for k, v in hs.items()}))
check("sigma=0.1 test hash differs from sigma=0.3", testhash(runs["m3_s01"][1]) not in hs.values())
sides = [r[3] for r in runs.values() if r[3]]
same_hashes = all(s["test_hash"] == sides[0]["test_hash"] for s in sides)
check("every sidecar has identical test-set hashes at every eval level", same_hashes and len(sides) == len(runs))
check("sigma=0 run's sidecar test hashes == the sigma>0 runs' (one test set per level)",
      bool(sZ) and sZ["test_hash"] == sides[0]["test_hash"])
spl = lambda log: re.findall(r"^\[V14\] SPLIT (train|val|test)\s+n=\d+ .*hash=(\w+)", log, re.M)  # noqa: E731
check("[V14] SPLIT hashes (clean data) identical with and without noise",
      spl(log0) and spl(log0) == spl(runs["m3_s03_a"][1]) == spl(runs["m4_s03"][1]))

# ---------------------------------------------------------------------------
header("4. Example images for figure D0 (not a check)")
try:
    te_raw = T._open_offline_first(T.datasets.MNIST, "./data", False)
    imgs = te_raw.data.numpy()[:8].astype(np.float32) / 255.0
    X = torch.from_numpy(imgs).unsqueeze(1)
    lv = [float(x) for x in EVALS.split()]
    arr = np.stack([T._v155_fixed(X, "gauss", s_, T.V155_TEST_GAUSS_BASE).numpy()[:, 0] for s_ in lv])
    sp = np.stack([T._v155_fixed(X, "sp", p_, T.V155_TEST_SP_BASE).numpy()[:, 0] for p_ in map(float, EVAL_SP.split())])
    np.savez_compressed(os.path.join(ROOT, "noise_examples_v155.npz"), gauss=arr, sigmas=np.array(lv), sp=sp,
                        ps=np.array([float(x) for x in EVAL_SP.split()]), labels=te_raw.targets.numpy()[:8])
    print("  wrote noise_examples_v155.npz (8 test digits x %d sigmas, + salt-and-pepper)" % len(lv))
except Exception as e:  # noqa: BLE001
    print("  examples skipped (%s: %s) -- not a failure" % (type(e).__name__, e))

print("\n" + "=" * 76)
if FAIL:
    print("VERIFY FAILED (%d):" % len(FAIL))
    for f in FAIL:
        print("  - " + f)
    print("Logs of the dry runs: " + TMP)
    sys.exit(1)
print("ALL CHECKS PASSED")
