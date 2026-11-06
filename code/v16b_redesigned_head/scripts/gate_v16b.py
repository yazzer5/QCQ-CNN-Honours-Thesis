#!/usr/bin/env python3
"""
v16b integrity gate. Called by pbs/check_v16b.sh from inside pbs/:
    python3 ../scripts/gate_v16b.py smoke|grid
Standard library only. Prints NO accuracy.

Every job log (the latest per run if a job was resubmitted):
  'Job end rc=0'; CONT=0 (no continuation); loss=ce; n_classes=K; class list; n_obs=15;
  trainer parameter check VERIFIED OK; determinism ENABLED; max_epochs = 200*500/N;
  warmup 'model=M4 min_epochs=0' (BFR runs as --model M4; pre-registered warmup decision);
  [V14] SPLIT: train n = N*K, val 125*K, test 200*K, every class present, zero overlap;
  no [V15] noise line, no v16a line, no FATAL;
  BFR: '[V16B] HEAD bfr installed model=M4 seed=S K=K fc_mid_width=w params_M4=.. params_BFR=.. VERIFIED'
       (pre-registered table), the gradient probe at g=1 + 'all trainable tensors receive gradient. OK'
       + g restored bit-exact, the GS line, '[V16B] RUN ... selfcheck=MATCH', the sidecar agrees with the
       result JSON (test_acc, labels, params, one state restore, best and stop state), n_params in the
       result JSON = BFR count;
  M4 reproduction: no [V16B] line at all.
Reproduction (smoke): M4, K=5 A, N=500, seed 42 == stored v14 run EXACTLY (test_acc, best/stop epoch,
  every prediction).
Across logs: runs at the same (K, subset, N, seed) saw identical images; one test set per (K, subset);
  different seeds drew different train sets; every run's split hashes equal the stored v14 M4 log of the
  same cell and seed (where that log is found) -- the pairing with the reused comparator.
Smoke: a run at its epoch cap STOPS the gate. Grid: completeness; cap hits reported.
Writes .smoke_ok / .grid_ok only if everything passes.
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
    sys.exit("usage: gate_v16b.py smoke|grid")
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/../pbs")


def sh(cmd):
    return subprocess.run(["bash", "-c", "source ./config16b.sh; " + cmd], stdout=subprocess.PIPE,
                          universal_newlines=True, check=True).stdout


def cfg(*names):
    return sh("; ".join('echo "${%s}"' % n for n in names)).split("\n")[:len(names)]


(SEEDS, SMOKE_SEED, B_KS, B_NS, W, VAL, TEST, V14) = cfg(
    "GRID_SEEDS", "SMOKE_SEED", "B_KS", "B_NS", "W_OBS", "PER_CLASS_VAL", "PER_CLASS_TEST", "V14_ROOT")
SEEDS = [int(s) for s in SEEDS.split()]
SMOKE_SEED, W, VAL, TEST = int(SMOKE_SEED), int(W), int(VAL), int(TEST)
B_KS, B_NS = [int(x) for x in B_KS.split()], [int(x) for x in B_NS.split()]
CAP = {n: int(sh("max_epochs_for %d" % n).strip()) for n in B_NS}
TAGS = {5: ["A", "B"], 10: ["all"]}
CLS = {(5, "A"): "1, 3, 5, 8, 9", (5, "B"): "0, 2, 6, 8, 9", (10, "all"): "0, 1, 2, 3, 4, 5, 6, 7, 8, 9"}
P_M4 = {5: 1719, 10: 1799}
P_BFR = {5: 1720, 10: 1799}
WIDTH = {5: 5, 10: 3}
TL = {"A": "A", "B": "B", "all": "X"}


def expected_runs(mode):
    seeds = [SMOKE_SEED] if mode == "smoke" else [s for s in SEEDS if s != SMOKE_SEED]
    out = [("b", K, T, N, "BFR", S) for K in B_KS for T in TAGS[K] for N in B_NS for S in seeds]
    if mode == "smoke":
        out.append(("r", 5, "A", 500, "M4", SMOKE_SEED))
    return out


FAIL, WARN = [], []


def bad(msg):
    print("  FAIL: " + msg)
    FAIL.append(msg)


def jobid(f):
    m = re.search(r"\.o(\d+)", f)
    return int(m.group(1)) if m else 0


pats = ["v16bs*.o*", "v16br_*.o*"] if MODE == "smoke" else ["v16bg*.o*"]
files = sorted({f for p in pats for f in glob.glob(p)}, key=jobid)
if not files:
    print("No %s logs found in pbs/. Check qstat." % MODE)
    sys.exit(1)

RX_HEAD = re.compile(r"^\[V16B\] ARM=(\w) K=(\d+) TAG=(\w+) N=(\d+) CLASSES=\S+ MODEL=(BFR|M4) SEED=(\d+) OUT=(\S+) CONT=(\d+)", re.M)
RX_SPLIT = re.compile(r"^\[V14\] SPLIT (train|val|test)\s+n=(\d+) labels=(\S+) per_class=(\S+) min=(\d+) max=(\d+) hash=(\w+)", re.M)
RX_OVER = re.compile(r"^\[V14\] SPLIT overlap train&val=(\d+) train&test=(\d+) val&test=(\d+)", re.M)

runs, dups = {}, defaultdict(int)
for f in files:
    txt = open(f, errors="replace").read()
    h = RX_HEAD.search(txt)
    if not h:
        bad("%s: no [V16B] header line (job died before start? cat it)" % f)
        continue
    key = (h.group(1), int(h.group(2)), h.group(3), int(h.group(4)), h.group(5), int(h.group(6)))
    dups[key] += 1
    runs[key] = dict(f=f, txt=txt, OUT=h.group(7), CONT=int(h.group(8)))
for k, n in dups.items():
    if n > 1:
        WARN.append("%s: %d logs (resubmitted); the latest (%s) is used" % (k, n, runs[k]["f"]))

print("Checking %d %s run(s) from %d log file(s)." % (len(runs), MODE, len(files)))
splits, results, sidecars, caps = {}, {}, {}, []
for key, r in sorted(runs.items()):
    ARM, K, TAG, N, M, S = key
    f, txt = r["f"], r["txt"]
    if not re.search(r"Job end.*rc=0", txt):
        bad("%s: no 'Job end rc=0'" % f); continue
    if r["CONT"] != 0 or re.search(r"CONT=[1-9]", txt) or "continuation" in txt:
        bad("%s: continuation (time budget hit, not bit-reproducible)" % f)
    if not re.search(r"PROTOCOL RESOLVED.*loss=ce", txt): bad("%s: loss is not ce" % f)
    if not re.search(r"PROTOCOL RESOLVED.*n_classes=%d\b" % K, txt): bad("%s: n_classes != %d" % (f, K))
    if not re.search(r"PROTOCOL RESOLVED.*classes=\[" + re.escape(CLS[(K, TAG)]) + r"\]", txt):
        bad("%s: class list != [%s]" % (f, CLS[(K, TAG)]))
    if not re.search(r"PROTOCOL RESOLVED.*head_variant=linear", txt): bad("%s: head_variant is not linear" % f)
    if not re.search(r"\[MULTICLASS\].*n_obs=%d\b" % W, txt): bad("%s: n_obs != %d" % (f, W))
    if not re.search(r"\[VERIFIED\] M4 params: .*total=%d \(expected %d\) OK" % (P_M4[K], P_M4[K]), txt):
        bad("%s: trainer parameter check for M4 (%d) not VERIFIED OK" % (f, P_M4[K]))
    if not re.search(r"DETERMINISM.*ENABLED", txt, re.I): bad("%s: determinism not enabled" % f)
    if not re.search(r"max_epochs=%d\b" % CAP[N], txt): bad("%s: max_epochs is not the pre-registered %d for N=%d" % (f, CAP[N], N))
    if "[V135] warmup: model=M4 min_epochs=0" not in txt: bad("%s: warmup is not M4's (min_epochs=0)" % f)
    if "[V15] NOISE" in txt: bad("%s: [V15] noise was installed (must be off)" % f)
    if re.search(r"\[V16\] (HEAD|RUN|SAVEZ)", txt): bad("%s: a v16a line appeared (v16a flags must be off)" % f)
    if re.search(r"\[V1[56]B?\] FATAL|GRADCHECK FAILED|PARAM COUNT MISMATCH|Traceback", txt): bad("%s: FATAL / traceback in log" % f)
    if M == "BFR":
        m = re.search(r"\[V16B\] HEAD bfr installed model=M4 seed=(\d+) K=(\d+) fc_mid_width=(\d+) params_M4=(\d+) "
                      r"params_BFR=(\d+) expected_BFR=(\d+) .*s0=1\.570796 g0=0\.0 theta0=\[0\.0, 0\.0, 0\.0, 0\.0\] .*VERIFIED", txt)
        if not m:
            bad("%s: [V16B] HEAD ... VERIFIED line missing (or init s/g/theta not as pre-registered)" % f)
        elif tuple(int(x) for x in m.groups()) != (S, K, WIDTH[K], P_M4[K], P_BFR[K], P_BFR[K]):
            bad("%s: HEAD line %s, expected seed %d K %d w %d M4 %d BFR %d" % (f, m.groups(), S, K, WIDTH[K], P_M4[K], P_BFR[K]))
        if not ("[V16B] GRADCHECK probe with g=1" in txt and "[GRADCHECK] M4: all trainable tensors receive gradient. OK" in txt
                and "[V16B] GRADCHECK g restored to 0.0 (bit-exact: True)" in txt):
            bad("%s: gradient probe / restore lines missing" % f)
        if not re.search(r"\[V16B\] GS best\(epoch \d+\): g=\S+ s=\S+ theta=.*\| stop\(epoch \d+\): g=\S+ s=\S+", txt):
            bad("%s: [V16B] GS line missing" % f)
        if not re.search(r"\[V16B\] RUN model=BFR arch=M4 head=bfr seed=%d params=%d selfcheck=MATCH" % (S, P_BFR[K]), txt):
            bad("%s: [V16B] RUN self-check not MATCH (%s)" % (f, "FAILED" if "RUN FAILED" in txt else "skipped" if "RUN skipped" in txt else "see log"))
    elif "[V16B] HEAD" in txt or "[V16B] RUN" in txt:
        bad("%s: BFR head active in an M4 reproduction job" % f)
    # ---- split audit
    sp = {mm.group(1): mm for mm in RX_SPLIT.finditer(txt)}
    ov = RX_OVER.search(txt)
    if len(sp) != 3 or not ov:
        bad("%s: [V14] SPLIT audit lines missing" % f)
    else:
        for name, n_exp in (("train", N * K), ("val", VAL * K), ("test", TEST * K)):
            mm = sp[name]
            if int(mm.group(2)) != n_exp: bad("%s: %s n=%s, expected %d" % (f, name, mm.group(2), n_exp))
            if len(mm.group(3).split(",")) != K or int(mm.group(5)) == 0: bad("%s: %s does not cover all %d classes" % (f, name, K))
        if any(int(x) for x in ov.groups()): bad("%s: images shared between splits %s" % (f, ov.groups()))
        splits[key] = {name: sp[name].group(7) for name in sp}
    # ---- result JSON and sidecar
    jf = os.path.join("..", r["OUT"], "result_M4_mnist_seed%d_alpha0.0.json" % S)
    if not os.path.exists(jf):
        bad("%s: result JSON missing (%s)" % (f, jf)); continue
    j = json.load(open(jf))
    results[key] = j
    if j.get("stop_epoch") is not None and int(j["stop_epoch"]) >= CAP[N]:
        caps.append(key)
        msg = "%s: stopped at the %d-epoch cap (best epoch %s)" % (key, CAP[N], j.get("best_epoch"))
        (bad if MODE == "smoke" else WARN.append)(msg + (" -- pre-registered smoke stop rule" if MODE == "smoke" else ""))
    exp_np = P_BFR[K] if M == "BFR" else P_M4[K]
    if j.get("n_params") != exp_np: bad("%s: result JSON n_params %s != %d" % (f, j.get("n_params"), exp_np))
    if len(j.get("test_predictions") or []) != TEST * K: bad("%s: per-image predictions missing" % f)
    if M == "BFR":
        sf = os.path.join("..", r["OUT"], "v16b_run_BFR_seed%d.json" % S)
        if not os.path.exists(sf):
            bad("%s: run sidecar missing (%s)" % (f, sf)); continue
        s = json.load(open(sf))
        sidecars[key] = s
        ja = float(j.get("test_acc", "nan"))
        if s.get("selfcheck") != "MATCH": bad("%s: sidecar selfcheck %s" % (f, s.get("selfcheck")))
        if not abs(s.get("acc", -1) - ja) < 1e-6: bad("%s: sidecar acc %s != result test_acc %s" % (f, s.get("acc"), ja))
        if s.get("labels") != j.get("test_labels"): bad("%s: sidecar labels != result JSON test_labels" % f)
        if s.get("params") != P_BFR[K]: bad("%s: sidecar params %s != %d" % (f, s.get("params"), P_BFR[K]))
        if s.get("n_state_loads") != 1 or not s.get("best") or not s.get("stop"):
            bad("%s: sidecar best/stop state incomplete (state restores = %s)" % (f, s.get("n_state_loads")))

# ---- cross-log checks ------------------------------------------------------------
groups = defaultdict(list)
for (ARM, K, TAG, N, M, S), s in splits.items():
    groups[(K, TAG, N, S)].append(((ARM, M), s))
for cell, lst in sorted(groups.items()):
    for part in ("train", "val", "test"):
        if len({s[part] for _, s in lst}) != 1:
            bad("%s: runs saw DIFFERENT %s sets" % (cell, part))
tests = defaultdict(set)
for (ARM, K, TAG, N, M, S), s in splits.items():
    tests[(K, TAG)].add(s["test"])
for kt, hs in tests.items():
    if len(hs) != 1: bad("%s: test set differs between runs (%d hashes)" % (kt, len(hs)))
trains = defaultdict(dict)
for (K, TAG, N, S), lst in groups.items():
    trains[(K, TAG, N)][S] = lst[0][1]["train"]
for c, d in trains.items():
    if len(set(d.values())) != len(d): bad("%s: %d seeds but only %d distinct train sets" % (c, len(d), len(set(d.values()))))
print("Splits: %d (K, subset, N, seed) groups; runs within a group saw identical images; one test set per (K, subset)." % len(groups))
# pairing with the reused comparator: the stored v14 M4 log of the same cell and seed
n_found = n_same = 0
for (ARM, K, TAG, N, M, S), s in splits.items():
    pat = ["%s/pbs/v14g%d%s%dM4s%d.o*" % (V14, K, TL[TAG], N, S)] + (["%s/pbs/v14s_%d%s%dM4.o*" % (V14, K, TL[TAG], N)] if S == SMOKE_SEED else [])
    cand = sorted({g for p in pat for g in glob.glob(p)}, key=jobid)
    if not cand:
        continue
    old = {mm.group(1): mm.group(7) for mm in RX_SPLIT.finditer(open(cand[-1], errors="replace").read())}
    if len(old) != 3:
        continue
    n_found += 1
    if old == s:
        n_same += 1
    else:
        bad("%s: split hashes differ from the stored v14 M4 log %s" % ((ARM, K, TAG, N, M, S), os.path.basename(cand[-1])))
print("Runs whose split equals the stored v14 M4 log of the same cell and seed: %d/%d found%s" %
      (n_same, n_found, "" if n_found else " (no v14 logs found; the reproduction carries the pairing)"))
if MODE == "grid" and n_found == 0:
    WARN.append("no stored v14 M4 logs found for the split cross-check (check V14_ROOT/pbs)")

# ---- completeness ------------------------------------------------------------------
missing = [k for k in expected_runs(MODE) if k not in results]
if missing:
    bad("%d expected run(s) missing, e.g. %s" % (len(missing), missing[:3]))

# ---- reproduction (gated) ----------------------------------------------------------
print("\nReproduction (gated: must equal the stored v14 run EXACTLY):")
nr = 0
for key in sorted(k for k in results if k[0] == "r"):
    ARM, K, TAG, N, M, S = key
    ref = "%s/results_v14_k%d_%s_n%d/result_M4_mnist_seed%d_alpha0.0.json" % (V14, K, TAG, N, S)
    nr += 1
    if not os.path.exists(ref):
        bad("reproduction %s: reference missing: %s" % (key, ref)); continue
    old, new = json.load(open(ref)), results[key]
    for k in ("test_acc", "best_epoch", "stop_epoch", "test_predictions"):
        same = old.get(k) == new.get(k)
        shown = ("%d per-image" % len(new.get(k) or [])) if k == "test_predictions" else "ref=%-8s new=%-8s" % (old.get(k), new.get(k))
        if k == "test_acc":
            shown = "(value not printed)"
        print("    M4 K=%d %s N=%d s%d %-17s %-28s %s" % (K, TAG, N, S, k, shown, "MATCH" if same else "DIFFERENT"))
        if not same:
            bad("reproduction %s: %s DIFFERENT -- pre-registered stop rule" % (key, k))
if MODE == "smoke" and nr == 0:
    bad("the M4 reproduction job (v16br_M4) has no result")
print("  %d reproduction run(s) checked." % nr)

# ---- verdict -------------------------------------------------------------------------
if caps and MODE == "grid":
    per = defaultdict(int)
    for k in caps:
        per[k[1:4]] += 1
    for c, n in per.items():
        WARN.append("%s: %d grid run(s) at the cap%s" % (c, n, " -> cell labelled budget-limited" if n > 2 else ""))
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
    print("Sanity only (not a result): epochs, job times, and the gate/scale at the best checkpoint:")
    for k, j in sorted(results.items()):
        t = re.findall(r"=== Job (?:start|end): (.*?)(?: rc=\d+)? ===", runs[k]["txt"])
        b = sidecars.get(k, {}).get("best") or {}
        gs = ("g=%+.3f s=%.3f" % (b["g"], b["s"])) if b else ""
        print("  %s K=%-2d %-3s N=%-3d %-3s best=%-4s stop=%-4s %-18s %s" % (k[0], k[1], k[2], k[3], k[4], j.get("best_epoch"),
                                                                         j.get("stop_epoch"), gs, " -> ".join(t)))
    print("Then: ./submit_v16b_grid.sh")
else:
    print("Then: cd .. && python3 scripts/collect_v16b.py | tee collect_v16b.log && python3 scripts/plot_v16b.py")
