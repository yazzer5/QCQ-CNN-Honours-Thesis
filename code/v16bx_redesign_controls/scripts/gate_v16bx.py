#!/usr/bin/env python3
"""
v16b-ext integrity gate (amendment A3). Called by pbs/check_v16bx.sh:   python3 ../scripts/gate_v16bx.py smoke|grid
Standard library only. Prints NO accuracy.

Per log: rc=0; no continuation; loss=ce; n_classes=10; class list; head_variant linear; n_obs=15; trainer M4 check
VERIFIED (1799); determinism; max_epochs 2000; warmup model=M4 min_epochs=0; noise off; no v16a line; no FATAL/traceback;
  M4trim: '[V16BX] HEAD m4trim ... fc_mid_width=3 params_M4=1799 params=1643 expected=1643 VERIFIED', no gated branch;
  BFRZZ : '[V16BX] HEAD bfrzz ... params=1799 ... s0=1.570796 g0=0.0 theta0=[0,0,0,0] ... zeta_scale=1 VERIFIED',
          probe at g=1 + all tensors OK + g restored bit-exact, GS line;
  both  : '[V16BX] RUN ... selfcheck=MATCH', sidecar agrees with the result JSON, n_params in the JSON;
  BFR reproduction: '[V16B] HEAD bfr ... VERIFIED' and no [V16BX] HEAD line.
Reproduction (smoke): BFR K=10 N=50 seed 42 == the stored v16b run EXACTLY (test_acc, best/stop epoch, predictions).
Splits: same images within a seed; one test set; distinct train sets per seed; every run's split == the stored v14 M4 log
  and the stored v16b BFR log of the same seed (where found).
Smoke: a run at its cap stops the gate. Grid: completeness; cap hits reported.
"""
import glob
import json
import os
import re
import subprocess
import sys
from collections import defaultdict

MODE = sys.argv[1] if len(sys.argv) > 1 else ""
if MODE not in ("smoke", "grid"):
    sys.exit("usage: gate_v16bx.py smoke|grid")
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/../pbs")


def sh(cmd):
    return subprocess.run(["bash", "-c", "source ./config16bx.sh; " + cmd], stdout=subprocess.PIPE,
                          universal_newlines=True, check=True).stdout


SEEDS, SMOKE, XM, V14, V16B, VAL, TEST = sh('echo "$GRID_SEEDS"; echo "$SMOKE_SEED"; echo "$X_MODELS"; echo "$V14_ROOT"; '
                                            'echo "$V16B_ROOT"; echo "$PER_CLASS_VAL"; echo "$PER_CLASS_TEST"').split("\n")[:7]
SEEDS, SMOKE, XM, VAL, TEST = [int(s) for s in SEEDS.split()], int(SMOKE), XM.split(), int(VAL), int(TEST)
K, N, CAP = 10, 50, int(sh("max_epochs_for 50").strip())
P = {"M4trim": 1643, "BFRZZ": 1799, "BFR": 1799}
FAIL, WARN = [], []


def bad(m):
    print("  FAIL: " + m); FAIL.append(m)


def jobid(f):
    m = re.search(r"\.o(\d+)", f)
    return int(m.group(1)) if m else 0


def expected(mode):
    seeds = [SMOKE] if mode == "smoke" else [s for s in SEEDS if s != SMOKE]
    out = [("x", M, S) for M in XM for S in seeds]
    if mode == "smoke":
        out.append(("r", "BFR", SMOKE))
    return out


pats = ["v16xs*.o*", "v16xr_*.o*"] if MODE == "smoke" else ["v16xg*.o*"]
files = sorted({f for p in pats for f in glob.glob(p)}, key=jobid)
if not files:
    print("No %s logs found in pbs/. Check qstat." % MODE); sys.exit(1)
RX_HEAD = re.compile(r"^\[V16BX\] ARM=(\w) K=10 TAG=all N=50 CLASSES=\S+ MODEL=(M4trim|BFRZZ|BFR) SEED=(\d+) OUT=(\S+) CONT=(\d+)", re.M)
RX_SPLIT = re.compile(r"^\[V14\] SPLIT (train|val|test)\s+n=(\d+) labels=(\S+) per_class=(\S+) min=(\d+) max=(\d+) hash=(\w+)", re.M)
RX_OVER = re.compile(r"^\[V14\] SPLIT overlap train&val=(\d+) train&test=(\d+) val&test=(\d+)", re.M)
runs, dups = {}, defaultdict(int)
for f in files:
    txt = open(f, errors="replace").read()
    h = RX_HEAD.search(txt)
    if not h:
        bad("%s: no [V16BX] header line (cat it)" % f); continue
    key = (h.group(1), h.group(2), int(h.group(3)))
    dups[key] += 1
    runs[key] = dict(f=f, txt=txt, OUT=h.group(4), CONT=int(h.group(5)))
for k, n in dups.items():
    if n > 1:
        WARN.append("%s: %d logs (resubmitted); the latest is used" % (k, n))

print("Checking %d %s run(s) from %d log file(s)." % (len(runs), MODE, len(files)))
splits, results, sidecars, caps = {}, {}, {}, []
for key, r in sorted(runs.items()):
    ARM, M, S = key
    f, txt = r["f"], r["txt"]
    if not re.search(r"Job end.*rc=0", txt):
        bad("%s: no 'Job end rc=0'" % f); continue
    if r["CONT"] != 0 or "continuation" in txt: bad("%s: continuation" % f)
    for rx, what in ((r"PROTOCOL RESOLVED.*loss=ce", "loss ce"), (r"PROTOCOL RESOLVED.*n_classes=10\b", "n_classes 10"),
                     (r"PROTOCOL RESOLVED.*classes=\[0, 1, 2, 3, 4, 5, 6, 7, 8, 9\]", "class list"),
                     (r"PROTOCOL RESOLVED.*head_variant=linear", "linear head"), (r"\[MULTICLASS\].*n_obs=15\b", "n_obs 15"),
                     (r"\[VERIFIED\] M4 params: .*total=1799 \(expected 1799\) OK", "M4 trainer check"),
                     (r"DETERMINISM.*ENABLED", "determinism"), (r"max_epochs=%d\b" % CAP, "max_epochs %d" % CAP)):
        if not re.search(rx, txt): bad("%s: %s missing" % (f, what))
    if "[V135] warmup: model=M4 min_epochs=0" not in txt: bad("%s: warmup is not M4's (0)" % f)
    if "[V15] NOISE" in txt or re.search(r"\[V16\] (HEAD|RUN|SAVEZ)", txt): bad("%s: noise or a v16a line present" % f)
    if re.search(r"FATAL|GRADCHECK FAILED|PARAM COUNT MISMATCH|Traceback", txt.split("[V16BX] split")[-1]): bad("%s: FATAL / traceback" % f)
    if M == "BFR":
        if not re.search(r"\[V16B\] HEAD bfr installed model=M4 seed=%d K=10 fc_mid_width=3 params_M4=1799 params_BFR=1799 .*VERIFIED" % S, txt):
            bad("%s: BFR HEAD line missing" % f)
        if "[V16BX] HEAD" in txt: bad("%s: v16bx head active in the BFR reproduction" % f)
    else:
        kind = "m4trim" if M == "M4trim" else "bfrzz"
        rx = r"\[V16BX\] HEAD %s installed model=M4 seed=%d K=10 fc_mid_width=3 params_M4=1799 params=%d expected=%d" % (kind, S, P[M], P[M])
        if kind == "bfrzz":
            rx += r" s0=1\.570796 g0=0\.0 theta0=\[0\.0, 0\.0, 0\.0, 0\.0\] .*zeta_scale=1 VERIFIED"
            if not ("[V16BX] GRADCHECK probe with g=1" in txt and "[GRADCHECK] M4: all trainable tensors receive gradient. OK" in txt
                    and "[V16BX] GRADCHECK g restored to 0.0 (bit-exact: True)" in txt and "[V16BX] GS best(epoch" in txt):
                bad("%s: BFR-ZZ probe / restore / GS lines missing" % f)
        else:
            rx += r" VERIFIED"
            if "GRADCHECK probe" in txt: bad("%s: gated probe ran in an M4-trim job" % f)
        if not re.search(rx, txt): bad("%s: HEAD %s VERIFIED line missing or wrong" % (f, kind))
        if "[V16B] HEAD" in txt: bad("%s: v16b BFR head active in an ext job" % f)
        if not re.search(r"\[V16BX\] RUN model=%s arch=M4 head=%s seed=%d params=%d selfcheck=MATCH" % (M, kind, S, P[M]), txt):
            bad("%s: RUN self-check not MATCH" % f)
    sp = {mm.group(1): mm for mm in RX_SPLIT.finditer(txt)}
    ov = RX_OVER.search(txt)
    if len(sp) != 3 or not ov:
        bad("%s: split audit lines missing" % f)
    else:
        for name, n_exp in (("train", N * K), ("val", VAL * K), ("test", TEST * K)):
            if int(sp[name].group(2)) != n_exp: bad("%s: %s n=%s != %d" % (f, name, sp[name].group(2), n_exp))
            if len(sp[name].group(3).split(",")) != K or int(sp[name].group(5)) == 0: bad("%s: %s misses classes" % (f, name))
        if any(int(x) for x in ov.groups()): bad("%s: overlapping splits" % f)
        splits[key] = {n: sp[n].group(7) for n in sp}
    jf = os.path.join("..", r["OUT"], "result_M4_mnist_seed%d_alpha0.0.json" % S)
    if not os.path.exists(jf):
        bad("%s: result JSON missing (%s)" % (f, jf)); continue
    j = json.load(open(jf)); results[key] = j
    if j.get("n_params") != P[M]: bad("%s: n_params %s != %d" % (f, j.get("n_params"), P[M]))
    if len(j.get("test_predictions") or []) != TEST * K: bad("%s: predictions missing" % f)
    if j.get("stop_epoch") is not None and int(j["stop_epoch"]) >= CAP:
        caps.append(key)
        (bad if MODE == "smoke" else WARN.append)("%s: stopped at the %d cap" % (key, CAP))
    if M != "BFR":
        sf = os.path.join("..", r["OUT"], "v16bx_run_%s_seed%d.json" % (M, S))
        if not os.path.exists(sf):
            bad("%s: sidecar missing" % f); continue
        s = json.load(open(sf)); sidecars[key] = s
        if s.get("selfcheck") != "MATCH" or abs(s.get("acc", -1) - float(j["test_acc"])) > 1e-6 or s.get("labels") != j.get("test_labels") \
                or s.get("params") != P[M]:
            bad("%s: sidecar disagrees with the result JSON" % f)
        if M == "BFRZZ" and (s.get("n_state_loads") != 1 or not s.get("best") or not s.get("stop")):
            bad("%s: BFR-ZZ sidecar best/stop state incomplete" % f)

by_seed = defaultdict(list)
for (ARM, M, S), s in splits.items():
    by_seed[S].append(s)
for S, lst in by_seed.items():
    for part in ("train", "val", "test"):
        if len({x[part] for x in lst}) != 1: bad("seed %d: runs saw different %s sets" % (S, part))
if len({s["test"] for s in splits.values()}) > 1: bad("test set differs between runs")
if len({lst[0]["train"] for lst in by_seed.values()}) != len(by_seed): bad("seeds do not draw distinct train sets")
nf = ns = 0
for (ARM, M, S), s in splits.items():
    for pat in ("%s/pbs/v14g10X50M4s%d.o*" % (V14, S), "%s/pbs/v14s_10X50M4.o*" % V14 if S == SMOKE else None,
                "%s/pbs/v16bg10X50s%d.o*" % (V16B, S), "%s/pbs/v16bs10X50.o*" % V16B if S == SMOKE else None):
        if pat is None:
            continue
        cand = sorted(glob.glob(pat), key=jobid)
        if not cand:
            continue
        old = {mm.group(1): mm.group(7) for mm in RX_SPLIT.finditer(open(cand[-1], errors="replace").read())}
        if len(old) != 3:
            continue
        nf += 1
        if old == s:
            ns += 1
        else:
            bad("%s: split differs from stored log %s" % ((ARM, M, S), os.path.basename(cand[-1])))
print("Splits: %d seed group(s), identical images within a seed, one test set. Equal to stored v14 M4 / v16b BFR logs: %d/%d found."
      % (len(by_seed), ns, nf))
if nf == 0:
    WARN.append("no stored v14/v16b logs found for the split cross-check")
missing = [k for k in expected(MODE) if k not in results]
if missing:
    bad("%d expected run(s) missing, e.g. %s" % (len(missing), missing[:3]))

print("\nReproduction (gated: must equal the stored v16b BFR run EXACTLY):")
nr = 0
for key in sorted(k for k in results if k[0] == "r"):
    nr += 1
    ref = "%s/results_v16b_bfr_k10_all_n50/result_M4_mnist_seed%d_alpha0.0.json" % (V16B, key[2])
    if not os.path.exists(ref):
        bad("reproduction reference missing: %s" % ref); continue
    old, new = json.load(open(ref)), results[key]
    for k in ("test_acc", "best_epoch", "stop_epoch", "test_predictions"):
        same = old.get(k) == new.get(k)
        shown = "(value not printed)" if k == "test_acc" else ("%d per-image" % len(new.get(k) or []) if k == "test_predictions"
                                                              else "ref=%-6s new=%-6s" % (old.get(k), new.get(k)))
        print("    BFR K=10 N=50 s%d %-17s %-26s %s" % (key[2], k, shown, "MATCH" if same else "DIFFERENT"))
        if not same:
            bad("reproduction %s: %s DIFFERENT -- stop rule" % (key, k))
if MODE == "smoke" and nr == 0:
    bad("the BFR reproduction (v16xr_BFR) has no result")
print("  %d reproduction run(s) checked." % nr)

if WARN:
    print("\nNotes:")
    for w in WARN:
        print("  - " + w)
flag = ".smoke_ok" if MODE == "smoke" else ".grid_ok"
if FAIL:
    print("\nGATE FAILED (%d problem(s)) -- do not read accuracies. Paste this output." % len(FAIL))
    if os.path.exists(flag):
        os.remove(flag)
    sys.exit(1)
open(flag, "w").write("ok\n")
print("\nALL CHECKS PASSED")
if MODE == "smoke":
    print("Sanity only (not a result): epochs, job times, g/s at the best checkpoint:")
    for k, j in sorted(results.items()):
        t = re.findall(r"=== Job (?:start|end): (.*?)(?: rc=\d+)? ===", runs[k]["txt"])
        b = (sidecars.get(k) or {}).get("best") or {}
        gs = ("g=%+.3f s=%.3f" % (b["g"], b["s"])) if b else ""
        print("  %s %-6s best=%-4s stop=%-4s %-18s %s" % (k[0], k[1], j.get("best_epoch"), j.get("stop_epoch"), gs, " -> ".join(t)))
    print("Then: ./submit_v16bx_grid.sh")
else:
    print("Then: cd .. && python3 scripts/collect_v16bx.py | tee collect_v16bx.log && python3 scripts/plot_v16bx.py")
