#!/usr/bin/env python3
"""
v15 integrity gate. Called by pbs/check_v15.sh from inside pbs/:
    python3 ../scripts/gate_v15.py smoke|grid
Standard library only (runs on any python3 >= 3.6).

Every job log (latest log per run if a job was resubmitted):
  rc=0; CONT=0; loss=ce; n_classes=K; class list; n_obs=15; params VERIFIED OK;
  determinism on; max_epochs=200; warmup min_epochs = 30 (M1, M2, M3, M5) or 0 (M4);
  [V14] SPLIT: train n = 500*K, val n = 125*K, test n = 200*K, K labels in each,
  zero overlap; M5: freeze banner + exit self-check UNCHANGED OK;
  [V15] NOISE installed: right model, right front-end kind (M1 henderson, M2/M3/M5
  circuit4, M4 none), alpha = the job's alpha, calibration = the pre-registered
  sha256; install self-checks PASS; no [V15] FATAL;
  [V15] XEVAL self-check MATCH (the trained model, evaluated at its own alpha,
  reproduces the saved per-image predictions); sidecar present with every
  evaluation alpha, and its accuracy at the training alpha == the result JSON's.
Across logs:
  - every model and every alpha at the same (K, subset, seed) used the SAME
    train/val/test images (noise never touches the data);
  - one fixed test set per (K, subset); different seeds drew different train sets.
Smoke only:
  - REPRODUCTIONS: alpha=0, seed 42, K=5 A must equal the stored runs EXACTLY
    (test_acc, best/stop epoch, every prediction) for all five models:
    M1, M4 = v14 (N=500); M2, M3 = v13 (w15); M5 = v13.5 (w15). Other cells are
    compared and reported.
  - any run at the 200-epoch cap STOPS the gate (pre-registered stop rule).
Grid: every (K, subset, alpha, model) cell has all 10 seeds; alpha=0 runs compared
  with the stored runs and the exact-match count reported; cap hits reported.
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
    sys.exit("usage: gate_v15.py smoke|grid")
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/../pbs")


def cfg(*names):
    cmd = "source ./config15.sh; " + "; ".join('echo "${%s}"' % n for n in names)
    out = subprocess.run(["bash", "-c", cmd], stdout=subprocess.PIPE, universal_newlines=True, check=True).stdout
    return out.split("\n")[:len(names)]


(KS, ALPHAS, EVALS, NOISE_MODELS, CTRL_MODELS, SEEDS, SMOKE_SEED, NPC, VAL, TEST, W, MAXEP,
 SHA, V14, V13, V135) = cfg("GRID_KS", "GRID_ALPHAS", "EVAL_ALPHAS", "NOISE_MODELS", "CTRL_MODELS", "GRID_SEEDS",
                            "SMOKE_SEED", "N_PER_CLASS", "PER_CLASS_VAL", "PER_CLASS_TEST", "W_OBS", "MAX_EPOCHS",
                            "CALIB_SHA", "V14_ROOT", "V13_ROOT", "V135_ROOT")
KS = [int(x) for x in KS.split()]
ALPHAS = ALPHAS.split()                       # strings, as in the job header
EVALS = sorted(float(x) for x in EVALS.split())
NOISE_MODELS, CTRL_MODELS = NOISE_MODELS.split(), CTRL_MODELS.split()
SEEDS = [int(x) for x in SEEDS.split()]
SMOKE_SEED, NPC, VAL, TEST, W, MAXEP = int(SMOKE_SEED), int(NPC), int(VAL), int(TEST), int(W), int(MAXEP)
TAGS = {5: ["A", "B"], 10: ["all"]}
CLS = {(5, "A"): "1, 3, 5, 8, 9", (5, "B"): "0, 2, 6, 8, 9", (10, "all"): "0, 1, 2, 3, 4, 5, 6, 7, 8, 9"}
KIND = {"M1": "henderson", "M2": "circuit4", "M3": "circuit4", "M4": "none", "M5": "circuit4"}


def models_for(a):
    return NOISE_MODELS + CTRL_MODELS if float(a) == 0 else NOISE_MODELS


def stored_ref(K, TAG, M, S):
    d = {"M1": "%s/results_v14_k%d_%s_n500" % (V14, K, TAG), "M4": "%s/results_v14_k%d_%s_n500" % (V14, K, TAG),
         "M2": "%s/results_v13_k%d_%s_w15" % (V13, K, TAG), "M3": "%s/results_v13_k%d_%s_w15" % (V13, K, TAG),
         "M5": "%s/results_v135_k%d_%s_w15" % (V135, K, TAG)}[M]
    src = {"M1": "v14", "M4": "v14", "M2": "v13", "M3": "v13", "M5": "v13.5"}[M]
    return os.path.join(d, "result_%s_mnist_seed%d_alpha0.0.json" % (M, S)), src


FAIL, WARN = [], []


def bad(msg):
    print("  FAIL: " + msg)
    FAIL.append(msg)


def jobid(f):
    m = re.search(r"\.o(\d+)", f)
    return int(m.group(1)) if m else 0


pats = ["v15s_*.o*"] if MODE == "smoke" else ["v15g_*.o*"]
files = sorted({f for p in pats for f in glob.glob(p)}, key=jobid)
if not files:
    print("No %s logs found in pbs/. Check qstat." % MODE)
    sys.exit(1)

RX_HEAD = re.compile(r"^\[V15\] K=(\d+) TAG=(\w+) ALPHA=(\S+) CLASSES=\S+ MODEL=(M\d) SEED=(\d+) OUT=(\S+) CONT=(\d+)", re.M)
RX_SPLIT = re.compile(r"^\[V14\] SPLIT (train|val|test)\s+n=(\d+) labels=(\S+) per_class=(\S+) min=(\d+) max=(\d+) hash=(\w+)", re.M)
RX_OVER = re.compile(r"^\[V14\] SPLIT overlap train&val=(\d+) train&test=(\d+) val&test=(\d+)", re.M)

runs, dups = {}, defaultdict(int)
for f in files:
    txt = open(f, errors="replace").read()
    h = RX_HEAD.search(txt)
    if not h:
        bad("%s: no [V15] header line (job died before start? cat it)" % f)
        continue
    K, TAG, A, M, S, OUT, CONT = int(h.group(1)), h.group(2), h.group(3), h.group(4), int(h.group(5)), h.group(6), int(h.group(7))
    key = (K, TAG, A, M, S)
    dups[key] += 1
    runs[key] = dict(f=f, txt=txt, OUT=OUT, CONT=CONT)
for k, n in dups.items():
    if n > 1:
        WARN.append("%s: %d logs (resubmitted); the latest (%s) is used" % (k, n, runs[k]["f"]))

print("Checking %d %s run(s) from %d log file(s)." % (len(runs), MODE, len(files)))
splits, results, xevals, caps = {}, {}, {}, []
for key, r in sorted(runs.items()):
    K, TAG, A, M, S = key
    f, txt = r["f"], r["txt"]
    if not re.search(r"Job end.*rc=0", txt):
        bad("%s: no 'Job end rc=0'" % f); continue
    if r["CONT"] != 0 or re.search(r"CONT=[1-9]", txt):
        bad("%s: continuation (time budget hit, not bit-reproducible)" % f)
    if not re.search(r"PROTOCOL RESOLVED.*loss=ce", txt): bad("%s: loss is not ce" % f)
    if not re.search(r"PROTOCOL RESOLVED.*n_classes=%d " % K, txt): bad("%s: n_classes != %d" % (f, K))
    if not re.search(r"PROTOCOL RESOLVED.*classes=\[" + re.escape(CLS[(K, TAG)]) + r"\]", txt):
        bad("%s: class list != [%s]" % (f, CLS[(K, TAG)]))
    if not re.search(r"\[MULTICLASS\].*n_obs=%d " % W, txt): bad("%s: n_obs != %d" % (f, W))
    if not re.search(r"VERIFIED.*OK", txt): bad("%s: parameter check not OK" % f)
    if not re.search(r"DETERMINISM.*ENABLED", txt, re.I): bad("%s: determinism not enabled" % f)
    if "max_epochs=%d " % MAXEP not in txt: bad("%s: max_epochs is not the pre-registered %d" % (f, MAXEP))
    want_warm = 0 if M == "M4" else 30
    if "[V135] warmup: model=%s min_epochs=%d" % (M, want_warm) not in txt:
        bad("%s: warmup for %s should be %d epochs" % (f, M, want_warm))
    if M == "M5":
        if "[V135] M5 = M3 with front-end frozen" not in txt: bad("%s: M5 freeze banner missing" % f)
        if "[V135] M5 front-end after run: UNCHANGED OK" not in txt: bad("%s: M5 exit self-check missing" % f)
        if "CHANGED -- FAIL" in txt: bad("%s: M5 FRONT-END CHANGED DURING TRAINING" % f)
    # ---- v15 noise lines
    m = re.search(r"\[V15\] NOISE installed model=(M\d) kind=(\w+) alpha=(\S+) calib=(\w+)", txt)
    if not m:
        bad("%s: [V15] NOISE installed line missing" % f)
    else:
        if m.group(1) != M: bad("%s: noise installed on %s, job is %s" % (f, m.group(1), M))
        if m.group(2) != KIND[M]: bad("%s: front-end kind %s, expected %s" % (f, m.group(2), KIND[M]))
        if float(m.group(3)) != float(A): bad("%s: noise alpha %s != job alpha %s" % (f, m.group(3), A))
        if m.group(4) != SHA[:12]: bad("%s: calibration %s is not the pre-registered %s" % (f, m.group(4), SHA[:12]))
    if "[V15] FATAL" in txt: bad("%s: [V15] FATAL in log" % f)
    if M != "M4" and not re.search(r"\[V15\] NOISE selfcheck PTM path at alpha=0 vs exact forward: .* PASS", txt):
        bad("%s: noise self-check (PTM at alpha=0) missing or not PASS" % f)
    if M == "M1" and not re.search(r"\[V15\] NOISE selfcheck M1 RandomLayers decomposition .* PASS", txt):
        bad("%s: M1 decomposition self-check missing or not PASS" % f)
    if not re.search(r"\[V15\] XEVAL model=%s seed=%d .*selfcheck=MATCH" % (M, S), txt):
        bad("%s: cross-evaluation self-check not MATCH (%s)" % (f, "XEVAL FAILED" if "XEVAL FAILED" in txt else
                                                               "skipped" if "XEVAL skipped" in txt else "see log"))
    # ---- split audit
    sp = {mm.group(1): mm for mm in RX_SPLIT.finditer(txt)}
    ov = RX_OVER.search(txt)
    if len(sp) != 3 or not ov:
        bad("%s: [V14] SPLIT audit lines missing" % f)
    else:
        for name, n_exp in (("train", NPC * K), ("val", VAL * K), ("test", TEST * K)):
            mm = sp[name]
            if int(mm.group(2)) != n_exp: bad("%s: %s n=%s, expected %d" % (f, name, mm.group(2), n_exp))
            if len(mm.group(3).split(",")) != K or int(mm.group(5)) == 0: bad("%s: %s does not cover all %d classes" % (f, name, K))
        if any(int(x) for x in ov.groups()): bad("%s: images shared between splits %s" % (f, ov.groups()))
        splits[key] = {name: sp[name].group(7) for name in sp}
    # ---- result JSON and cross-eval sidecar
    jf = os.path.join("..", r["OUT"], "result_%s_mnist_seed%d_alpha0.0.json" % (M, S))
    if not os.path.exists(jf):
        bad("%s: result JSON missing (%s)" % (f, jf)); continue
    j = json.load(open(jf))
    results[key] = j
    if j.get("stop_epoch") is not None and int(j["stop_epoch"]) >= MAXEP:
        msg = "%s: stopped at the %d-epoch cap (best epoch %s)" % (key, MAXEP, j.get("best_epoch"))
        caps.append(key)
        (bad if MODE == "smoke" else WARN.append)(msg + (" -- pre-registered smoke stop rule" if MODE == "smoke" else ""))
    sf = os.path.join("..", r["OUT"], "v15_xeval_%s_seed%d.json" % (M, S))
    if not os.path.exists(sf):
        bad("%s: cross-eval sidecar missing (%s)" % (f, sf)); continue
    s = json.load(open(sf))
    xevals[key] = s
    if s.get("selfcheck") != "MATCH": bad("%s: sidecar selfcheck %s" % (f, s.get("selfcheck")))
    if float(s.get("alpha_train", -1)) != float(A): bad("%s: sidecar alpha_train %s != %s" % (f, s.get("alpha_train"), A))
    if sorted(float(x) for x in s.get("acc", {})) != EVALS: bad("%s: sidecar eval alphas %s" % (f, sorted(s.get("acc", {}))))
    if s.get("calib_sha") != SHA: bad("%s: sidecar calibration differs" % f)
    a_tr = s.get("acc", {}).get("%g" % float(A))
    ja = float(j.get("test_acc", "nan"))
    if a_tr is None or not (abs(a_tr - ja) < 1e-6 or abs(100 * a_tr - ja) < 1e-4):
        bad("%s: sidecar accuracy at the training alpha (%s) != result test_acc (%s)" % (f, a_tr, ja))

# ---- cross-log checks ------------------------------------------------------------
same = defaultdict(list)
for (K, TAG, A, M, S), s in splits.items():
    same[(K, TAG, S)].append(((A, M), s))
for cell, lst in sorted(same.items()):
    for part in ("train", "val", "test"):
        hs = {s[part] for _, s in lst}
        if len(hs) != 1:
            bad("%s: models/alphas saw DIFFERENT %s sets %s" % (cell, part, sorted(set((am, s[part]) for am, s in lst))[:4]))
n_cells = len(same)
tests = defaultdict(set)
for (K, TAG, A, M, S), s in splits.items():
    tests[(K, TAG)].add(s["test"])
for kt, hs in tests.items():
    if len(hs) != 1: bad("%s: test set differs between runs (%d hashes)" % (kt, len(hs)))
trains = defaultdict(set)
for (K, TAG, S), lst in same.items():
    trains[(K, TAG)].add(lst[0][1]["train"])
for kt, hs in trains.items():
    nseeds = len({S for (K, TAG, S) in same if (K, TAG) == kt})
    if len(hs) != nseeds: bad("%s: %d seeds but only %d distinct train sets" % (kt, nseeds, len(hs)))
print("Splits: %d (K, subset, seed) groups; every model and alpha within a group saw identical images; "
      "one test set per (K, subset)." % n_cells)

# ---- completeness ------------------------------------------------------------------
want_seeds = [SMOKE_SEED] if MODE == "smoke" else [s for s in SEEDS if s != SMOKE_SEED]
missing = [(K, TAG, A, M, S) for K in KS for TAG in TAGS[K] for A in ALPHAS for M in models_for(A)
           for S in want_seeds if (K, TAG, A, M, S) not in results]
if missing:
    bad("%d expected run(s) missing, e.g. %s" % (len(missing), missing[:3]))


# ---- reproductions -------------------------------------------------------------
def compare(key, verbose):
    K, TAG, A, M, S = key
    ref, src = stored_ref(K, TAG, M, S)
    if not os.path.exists(ref):
        return None, "reference missing: %s" % ref
    old, new = json.load(open(ref)), results[key]
    ok = all(old.get(k) == new.get(k) for k in ("test_acc", "best_epoch", "stop_epoch"))
    pk = next((k for k in ("predictions", "test_predictions", "preds") if k in old and k in new), None)
    if pk:
        ok = ok and old[pk] == new[pk]
    if verbose:
        for k in ("test_acc", "best_epoch", "stop_epoch"):
            print("    %s K=%d %s %-10s ref(%s)=%s  new=%s  %s" % (M, K, TAG, k, src, old.get(k), new.get(k),
                                                             "MATCH" if old.get(k) == new.get(k) else "DIFFERENT"))
        if pk:
            print("    %s K=%d %s %-10s %d per-image  %s" % (M, K, TAG, pk, len(new[pk]), "MATCH" if old[pk] == new[pk] else "DIFFERENT"))
    return ok, src


zero = [k for k in results if float(k[2]) == 0]
if MODE == "smoke":
    print("\nReproductions, alpha=0 seed 42 (pre-registered gate: K=5 A, all five models, EXACT):")
    for M in ("M1", "M2", "M3", "M4", "M5"):
        key = (5, "A", "0", M, SMOKE_SEED)
        if key not in results:
            bad("reproduction %s: no alpha=0 smoke result" % M); continue
        ok, src = compare(key, True)
        if ok is None: bad("reproduction %s: %s" % (M, src))
        elif not ok: bad("reproduction %s vs %s: DIFFERENT -- pre-registered stop rule" % (M, src))
    others = [k for k in zero if (k[0], k[1]) != (5, "A")]
    if others:
        print("  Other cells (reported, not gated):")
        for k in sorted(others):
            ok, src = compare(k, False)
            print("    %s K=%d %s vs %s: %s" % (k[3], k[0], k[1], src, "MATCH" if ok else ("no reference" if ok is None else "DIFFERENT")))
            if ok is False: WARN.append("alpha=0 %s K=%d %s differs from %s (reported; the gate is K=5 A)" % (k[3], k[0], k[1], src))
else:
    n_ok = n_ref = 0
    for k in zero:
        ok, _ = compare(k, False)
        if ok is not None:
            n_ref += 1; n_ok += int(ok)
    print("alpha=0 grid runs reproducing the stored v13/v13.5/v14 runs exactly: %d/%d (with a reference)" % (n_ok, n_ref))
    if n_ok < n_ref:
        WARN.append("%d alpha=0 grid run(s) differ from the stored runs (reported; the pre-registered gate is the smoke)" % (n_ref - n_ok))

# ---- verdict -------------------------------------------------------------------------
if caps and MODE == "grid":
    per = defaultdict(int)
    for k in caps: per[k[:4]] += 1
    for c, n in per.items():
        if n > 2: WARN.append("%s: %d/9 grid runs at the cap -> cell labelled budget-limited" % (c, n))
if WARN:
    print("\nNotes:")
    for w in WARN: print("  - " + w)
flag = ".smoke_ok" if MODE == "smoke" else ".grid_ok"
if FAIL:
    print("\nGATE FAILED (%d problem(s)) -- do not read accuracies. Paste this output." % len(FAIL))
    if os.path.exists(flag): os.remove(flag)
    sys.exit(1)
open(flag, "w").write("ok\n")
print("\nALL CHECKS PASSED")
if MODE == "smoke":
    print("Sanity only (not a result): smoke runs must sit well above chance (1/K) and the cross-eval must exist:")
    for k, j in sorted(results.items()):
        print("  K=%-2d %-3s a=%-4s %s  test_acc=%.4f  best=%s stop=%s" % (k[0], k[1], k[2], k[3], float(j.get("test_acc")),
                                                                     j.get("best_epoch"), j.get("stop_epoch")))
    print("Then: ./submit_v15_grid.sh")
else:
    print("Then: cd .. && python3 scripts/collect_v15.py | tee collect_v15.log && python3 scripts/plot_v15.py")
