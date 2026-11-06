#!/usr/bin/env python3
"""
v16a integrity gate. Called by pbs/check_v16a.sh from inside pbs/:
    python3 ../scripts/gate_v16a.py smoke|grid
Standard library only.

Every job log (the latest per run if a job was resubmitted):
  'Job end rc=0'; CONT=0 (no continuation); loss=ce; n_classes=K; class list; n_obs=15;
  trainer parameter check VERIFIED OK; determinism ENABLED; max_epochs = 200*500/N;
  warmup min_epochs=30 (M2, M3, M2c all have the Circuit-4 front-end);
  [V14] SPLIT: train n = N*K, val 125*K, test 200*K, every class present, zero overlap;
  no [V15] noise line (noise off) and no [V15]/[V16] FATAL;
  M2c: '[V16] HEAD m2c installed ... wiring=identity params_M2=<1686|1766>
        params_M2c=<1665|1730> ... VERIFIED' (pre-registered counts);
  every run: '[V16] RUN ... selfcheck=MATCH' and its sidecar (v16_run_*.json) agrees with
        the result JSON (test_acc, labels present, MATCH);
  arm z: '[V16] SAVEZ ... n=200*K dim=2 ... selfcheck=MATCH', z file present, hash in sidecar.
Reproductions (gated in BOTH smoke and grid): arm z (M2, M3, K=10, N=50/500, seeds 42-44)
  and the repro job (M2, K=5 A, N=500, seed 42) must equal the stored v13 (N=500) / v14
  (N=50) runs EXACTLY: test_acc, best_epoch, stop_epoch and every prediction.
Across logs: every run at the same (K, subset, N, seed) saw identical images; one test set
  per (K, subset); different seeds drew different train sets; at N=50 every run's split hashes
  equal the stored v14 M2 log of the same cell (where that log is found).
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
    sys.exit("usage: gate_v16a.py smoke|grid")
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/../pbs")


def sh(cmd):
    return subprocess.run(["bash", "-c", "source ./config16a.sh; " + cmd], stdout=subprocess.PIPE,
                          universal_newlines=True, check=True).stdout


def cfg(*names):
    return sh("; ".join('echo "${%s}"' % n for n in names)).split("\n")[:len(names)]


(SEEDS, SMOKE_SEED, C_KS, C_NS, Z_MODELS, Z_K, Z_NS, Z_SEEDS, W, VAL, TEST, V13, V14) = cfg(
    "GRID_SEEDS", "SMOKE_SEED", "C_KS", "C_NS", "Z_MODELS", "Z_K", "Z_NS", "Z_SEEDS", "W_OBS",
    "PER_CLASS_VAL", "PER_CLASS_TEST", "V13_ROOT", "V14_ROOT")
SEEDS = [int(s) for s in SEEDS.split()]
SMOKE_SEED, W, VAL, TEST, Z_K = int(SMOKE_SEED), int(W), int(VAL), int(TEST), int(Z_K)
C_KS, C_NS, Z_NS = [int(x) for x in C_KS.split()], [int(x) for x in C_NS.split()], [int(x) for x in Z_NS.split()]
Z_MODELS, Z_SEEDS = Z_MODELS.split(), [int(s) for s in Z_SEEDS.split()]
NS_ALL = sorted(set(C_NS + Z_NS))
CAP = {n: int(sh("max_epochs_for %d" % n).strip()) for n in NS_ALL}
TAGS = {5: ["A", "B"], 10: ["all"]}
CLS = {(5, "A"): "1, 3, 5, 8, 9", (5, "B"): "0, 2, 6, 8, 9", (10, "all"): "0, 1, 2, 3, 4, 5, 6, 7, 8, 9"}
P_M2 = {5: 1686, 10: 1766}
P_M2C = {5: 1665, 10: 1730}
TL = {"A": "A", "B": "B", "all": "X"}


def stored(K, TAG, N, M, S):
    d = ("%s/results_v13_k%d_%s_w15" % (V13, K, TAG)) if N == 500 else ("%s/results_v14_k%d_%s_n%d" % (V14, K, TAG, N))
    return os.path.join(d, "result_%s_mnist_seed%d_alpha0.0.json" % (M, S)), ("v13" if N == 500 else "v14")


def expected_runs(mode):
    seeds_c = [SMOKE_SEED] if mode == "smoke" else [s for s in SEEDS if s != SMOKE_SEED]
    seeds_z = [SMOKE_SEED] if mode == "smoke" else [s for s in Z_SEEDS if s != SMOKE_SEED]
    out = [("c", K, T, N, "M2c", S) for K in C_KS for T in TAGS[K] for N in C_NS for S in seeds_c]
    out += [("z", Z_K, "all", N, M, S) for N in Z_NS for M in Z_MODELS for S in seeds_z]
    if mode == "smoke":
        out.append(("r", 5, "A", 500, "M2", SMOKE_SEED))
    return out


FAIL, WARN = [], []


def bad(msg):
    print("  FAIL: " + msg)
    FAIL.append(msg)


def jobid(f):
    m = re.search(r"\.o(\d+)", f)
    return int(m.group(1)) if m else 0


pats = ["v16sc*.o*", "v16sz*.o*", "v16r_*.o*"] if MODE == "smoke" else ["v16gc*.o*", "v16gz*.o*"]
files = sorted({f for p in pats for f in glob.glob(p)}, key=jobid)
if not files:
    print("No %s logs found in pbs/. Check qstat." % MODE)
    sys.exit(1)

RX_HEAD = re.compile(r"^\[V16A\] ARM=(\w) K=(\d+) TAG=(\w+) N=(\d+) CLASSES=\S+ MODEL=(M2c|M2|M3) SEED=(\d+) OUT=(\S+) CONT=(\d+)", re.M)
RX_SPLIT = re.compile(r"^\[V14\] SPLIT (train|val|test)\s+n=(\d+) labels=(\S+) per_class=(\S+) min=(\d+) max=(\d+) hash=(\w+)", re.M)
RX_OVER = re.compile(r"^\[V14\] SPLIT overlap train&val=(\d+) train&test=(\d+) val&test=(\d+)", re.M)

runs, dups = {}, defaultdict(int)
for f in files:
    txt = open(f, errors="replace").read()
    h = RX_HEAD.search(txt)
    if not h:
        bad("%s: no [V16A] header line (job died before start? cat it)" % f)
        continue
    key = (h.group(1), int(h.group(2)), h.group(3), int(h.group(4)), h.group(5), int(h.group(6)))
    dups[key] += 1
    runs[key] = dict(f=f, txt=txt, OUT=h.group(7), CONT=int(h.group(8)))
for k, n in dups.items():
    if n > 1:
        WARN.append("%s: %d logs (resubmitted); the latest (%s) is used" % (k, n, runs[k]["f"]))

print("Checking %d %s run(s) from %d log file(s)." % (len(runs), MODE, len(files)))
splits, results, caps = {}, {}, []
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
    if not re.search(r"\[MULTICLASS\].*n_obs=%d\b" % W, txt): bad("%s: n_obs != %d" % (f, W))
    if not re.search(r"VERIFIED.*OK", txt): bad("%s: trainer parameter check not VERIFIED OK" % f)
    if not re.search(r"DETERMINISM.*ENABLED", txt, re.I): bad("%s: determinism not enabled" % f)
    if not re.search(r"max_epochs=%d\b" % CAP[N], txt): bad("%s: max_epochs is not the pre-registered %d for N=%d" % (f, CAP[N], N))
    arch = "M2" if M == "M2c" else M
    if "[V135] warmup: model=%s min_epochs=30" % arch not in txt: bad("%s: warmup for %s should be 30 epochs" % (f, arch))
    if "[V15] NOISE" in txt: bad("%s: [V15] noise was installed (must be off)" % f)
    if re.search(r"\[V1[56]\] FATAL", txt): bad("%s: FATAL line in log" % f)
    if M == "M2c":
        m = re.search(r"\[V16\] HEAD m2c installed model=M2 seed=(\d+) .*wiring=identity params_M2=(\d+) "
                      r"params_M2c=(\d+) expected_M2c=(\S+) VERIFIED", txt)
        if not m:
            bad("%s: [V16] HEAD m2c ... VERIFIED line missing" % f)
        elif int(m.group(1)) != S or int(m.group(2)) != P_M2[K] or int(m.group(3)) != P_M2C[K]:
            bad("%s: M2c head line seed/params %s, expected seed %d, %d -> %d" % (f, m.groups(), S, P_M2[K], P_M2C[K]))
    elif "[V16] HEAD" in txt:
        bad("%s: M2c head installed in a %s job" % (f, M))
    if not re.search(r"\[V16\] RUN model=%s arch=%s .*seed=%d .*selfcheck=MATCH" % (M, arch, S), txt):
        bad("%s: [V16] RUN self-check not MATCH (%s)" % (f, "FAILED" if "RUN FAILED" in txt else "skipped" if "RUN skipped" in txt else "see log"))
    if ARM == "z":
        if not re.search(r"\[V16\] SAVEZ model=%s seed=%d n=%d dim=2 .*selfcheck=MATCH" % (M, S, TEST * K), txt):
            bad("%s: [V16] SAVEZ line missing or not MATCH" % f)
    elif "[V16] SAVEZ" in txt:
        bad("%s: z saved in a non-z job" % f)
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
    # ---- result JSON and sidecars
    jf = os.path.join("..", r["OUT"], "result_%s_mnist_seed%d_alpha0.0.json" % (arch, S))
    if not os.path.exists(jf):
        bad("%s: result JSON missing (%s)" % (f, jf)); continue
    j = json.load(open(jf))
    results[key] = j
    if j.get("stop_epoch") is not None and int(j["stop_epoch"]) >= CAP[N]:
        caps.append(key)
        msg = "%s: stopped at the %d-epoch cap (best epoch %s)" % (key, CAP[N], j.get("best_epoch"))
        (bad if MODE == "smoke" else WARN.append)(msg + (" -- pre-registered smoke stop rule" if MODE == "smoke" else ""))
    sf = os.path.join("..", r["OUT"], "v16_run_%s_seed%d.json" % (M, S))
    if not os.path.exists(sf):
        bad("%s: run sidecar missing (%s)" % (f, sf)); continue
    s = json.load(open(sf))
    ja = float(j.get("test_acc", "nan"))
    if s.get("selfcheck") != "MATCH": bad("%s: sidecar selfcheck %s" % (f, s.get("selfcheck")))
    if not (abs(s.get("acc", -1) - ja) < 1e-6 or abs(100 * s.get("acc", -1) - ja) < 1e-4):
        bad("%s: sidecar acc %s != result test_acc %s" % (f, s.get("acc"), ja))
    if len(s.get("labels", [])) != TEST * K: bad("%s: sidecar labels missing" % f)
    if s.get("head") != ("m2c" if M == "M2c" else "native"): bad("%s: sidecar head %s" % (f, s.get("head")))
    if M == "M2c" and s.get("params") != P_M2C[K]: bad("%s: sidecar params %s != %d" % (f, s.get("params"), P_M2C[K]))
    if ARM == "z":
        zf = os.path.join("..", r["OUT"], "v16_z_%s_seed%d.npz" % (M, S))
        if not (s.get("z_saved") and os.path.exists(zf) and s.get("z_shape") == [TEST * K, 2] and s.get("z_sha256")):
            bad("%s: z file or its sidecar record missing (%s)" % (f, zf))

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
# against the stored v14 logs at N=50 (the M2 runs M2c is paired with)
n_found = n_same = 0
for (ARM, K, TAG, N, M, S), s in splits.items():
    if N == 500:
        continue
    pat = ["%s/pbs/v14g%d%s%dM2s%d.o*" % (V14, K, TL[TAG], N, S)] + (["%s/pbs/v14s_%d%s%dM2.o*" % (V14, K, TL[TAG], N)] if S == SMOKE_SEED else [])
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
        bad("%s: split hashes differ from the stored v14 M2 log %s" % ((ARM, K, TAG, N, M, S), os.path.basename(cand[-1])))
print("N=50 runs whose split equals the stored v14 M2 log of the same cell: %d/%d found%s" %
      (n_same, n_found, "" if n_found else " (no v14 logs found; the reproductions carry the pairing)"))

# ---- completeness ------------------------------------------------------------------
missing = [k for k in expected_runs(MODE) if k not in results]
if missing:
    bad("%d expected run(s) missing, e.g. %s" % (len(missing), missing[:3]))


# ---- reproductions (gated) ---------------------------------------------------------
def compare(key, verbose):
    ARM, K, TAG, N, M, S = key
    ref, src = stored(K, TAG, N, M, S)
    if not os.path.exists(ref):
        return None, "reference missing: %s" % ref
    old, new = json.load(open(ref)), results[key]
    ok = all(old.get(k) == new.get(k) for k in ("test_acc", "best_epoch", "stop_epoch"))
    pk = next((k for k in ("predictions", "test_predictions", "preds") if k in old and k in new), None)
    ok = ok and pk is not None and old[pk] == new[pk]
    if verbose:
        for k in ("test_acc", "best_epoch", "stop_epoch"):
            print("    %-3s K=%-2d %-3s N=%-3d s%d %-10s ref(%s)=%-20s new=%-20s %s" % (
                M, K, TAG, N, S, k, src, old.get(k), new.get(k), "MATCH" if old.get(k) == new.get(k) else "DIFFERENT"))
        if pk:
            print("    %-3s K=%-2d %-3s N=%-3d s%d %-10s %d per-image  %s" % (
                M, K, TAG, N, S, pk, len(new[pk]), "MATCH" if old[pk] == new[pk] else "DIFFERENT"))
    return ok, src


print("\nReproductions (gated: must equal the stored run EXACTLY):")
nr = 0
for key in sorted(k for k in results if k[0] in ("z", "r")):
    ok, src = compare(key, True)
    nr += 1
    if ok is None:
        bad("reproduction %s: %s" % (key, src))
    elif not ok:
        bad("reproduction %s vs %s: DIFFERENT -- pre-registered stop rule" % (key, src))
print("  %d reproduction run(s) checked." % nr)

# ---- verdict -------------------------------------------------------------------------
if caps and MODE == "grid":
    per = defaultdict(int)
    for k in caps:
        per[k[:4]] += 1
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
    print("Sanity only (not a result): epochs per run, and time per run for the walltime decision:")
    for k, j in sorted(results.items()):
        t = re.findall(r"=== Job (?:start|end): (.*?)(?: rc=\d+)? ===", runs[k]["txt"])
        print("  %s K=%-2d %-3s N=%-3d %-3s best=%-4s stop=%-4s  %s" % (k[0], k[1], k[2], k[3], k[4], j.get("best_epoch"),
                                                                     j.get("stop_epoch"), " -> ".join(t)))
    print("Then: ./submit_v16a_grid.sh")
else:
    print("Then: cd .. && python3 scripts/collect_v16a.py | tee collect_v16a.log && python3 scripts/plot_v16a.py")
