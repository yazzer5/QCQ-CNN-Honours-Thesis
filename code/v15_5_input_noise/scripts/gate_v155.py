#!/usr/bin/env python3
"""
v15.5 integrity gate. Called by pbs/check_v155.sh from inside pbs/:
    python3 ../scripts/gate_v155.py smoke|grid
Standard library only (runs on any python3 >= 3.6).

Every job log (latest log per run if a job was resubmitted):
  rc=0; CONT=0; loss=ce; n_classes=K; class list; n_obs=15; params VERIFIED OK with the
  pre-registered total; determinism on; max_epochs=200; warmup min_epochs = 30 (M1, M2,
  M3, M5) or 0 (M4); [V14] SPLIT: train n = 500*K, val n = 125*K, test n = 200*K, K labels
  in each, zero overlap; M5: freeze banner + exit self-check UNCHANGED OK;
  [V155] header sigma == result folder sigma == [V155] NOISE line sigma == sidecar sigma;
  [V155] NOISE line with the pre-registered generator seeds (sigma > 0) or "untouched"
  (sigma = 0); self-checks PASS (images in range; global RNG untouched); no [V155] FATAL;
  [V155] XEVAL self-check MATCH; sidecar with every Gaussian sigma and salt-and-pepper p,
  its accuracy at the job's sigma == the result JSON's test_acc; training noise drawn
  (sigma > 0).
Across logs:
  - clean [V14] split hashes identical across models and sigmas at each (K, subset, seed);
    one clean test set per (K, subset); different seeds drew different train sets;
  - ONE noisy test-set hash per (K, subset, sigma) across all models and seeds (log), and
    one per (K, subset, eval level) across all sidecars;
  - every sigma=0 run that has a stored v14 / v13 / v13.5 counterpart reproduces it EXACTLY
    (test_acc, best/stop epoch, every prediction). Smoke: K=5 A seed 42 must exist for all
    five models (pre-registered stop rule).
Smoke only: any run at the 200-epoch cap STOPS the gate (pre-registered stop rule).
Grid: every (K, subset, sigma, model) cell has all 9 grid seeds; cap hits reported.
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
    sys.exit("usage: gate_v155.py smoke|grid")
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/../pbs")


def cfg(*names):
    cmd = "source ./config155.sh; " + "; ".join('echo "${%s}"' % n for n in names)
    out = subprocess.run(["bash", "-c", cmd], stdout=subprocess.PIPE, universal_newlines=True, check=True).stdout
    return out.split("\n")[:len(names)]


(KS, SIGMAS, EVALS, EVAL_SP, MODELS, SEEDS, SMOKE_SEED, NPC, VAL, TEST, W, MAXEP,
 V14, V13, V135) = cfg("GRID_KS", "GRID_SIGMAS", "EVAL_SIGMAS", "EVAL_SP", "MODELS", "GRID_SEEDS",
                       "SMOKE_SEED", "N_PER_CLASS", "PER_CLASS_VAL", "PER_CLASS_TEST", "W_OBS", "MAX_EPOCHS",
                       "V14_ROOT", "V13_ROOT", "V135_ROOT")
KS = [int(x) for x in KS.split()]
SIGMAS = SIGMAS.split()                       # strings, as in the job header and folder names
EVALS = sorted(float(x) for x in EVALS.split())
EVAL_SP = sorted(float(x) for x in EVAL_SP.split())
MODELS = MODELS.split()
SEEDS = [int(x) for x in SEEDS.split()]
SMOKE_SEED, NPC, VAL, TEST, W, MAXEP = int(SMOKE_SEED), int(NPC), int(VAL), int(TEST), int(W), int(MAXEP)
TAGS = {5: ["A", "B"], 10: ["all"]}
CLS = {(5, "A"): "1, 3, 5, 8, 9", (5, "B"): "0, 2, 6, 8, 9", (10, "all"): "0, 1, 2, 3, 4, 5, 6, 7, 8, 9"}
PARAMS = {5: dict(M1=1680, M2=1686, M3=1725, M4=1719, M5=1725), 10: dict(M1=1760, M2=1766, M3=1805, M4=1799, M5=1805)}


def code(x):
    return int(round(1000 * float(x)))


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


pats = ["v155s_*.o*"] if MODE == "smoke" else ["v155g_*.o*"]
files = sorted({f for p in pats for f in glob.glob(p)}, key=jobid)
if not files:
    print("No %s logs found in pbs/. Check qstat." % MODE)
    sys.exit(1)

RX_HEAD = re.compile(r"^\[V155\] K=(\d+) TAG=(\w+) SIGMA=(\S+) CLASSES=\S+ MODEL=(M\d) SEED=(\d+) OUT=(\S+) CONT=(\d+)", re.M)
RX_SPLIT = re.compile(r"^\[V14\] SPLIT (train|val|test)\s+n=(\d+) labels=(\S+) per_class=(\S+) min=(\d+) max=(\d+) hash=(\w+)", re.M)
RX_OVER = re.compile(r"^\[V14\] SPLIT overlap train&val=(\d+) train&test=(\d+) val&test=(\d+)", re.M)
RX_TEST = re.compile(r"^\[V155\] TESTSET sigma=(\S+) n=(\d+) sha=(\w+)", re.M)

runs, dups = {}, defaultdict(int)
for f in files:
    txt = open(f, errors="replace").read()
    h = RX_HEAD.search(txt)
    if not h:
        bad("%s: no [V155] header line (job died before start? cat it)" % f)
        continue
    K, TAG, SG, M, S, OUT, CONT = int(h.group(1)), h.group(2), h.group(3), h.group(4), int(h.group(5)), h.group(6), int(h.group(7))
    key = (K, TAG, SG, M, S)
    dups[key] += 1
    runs[key] = dict(f=f, txt=txt, OUT=OUT, CONT=CONT)
for k, n in dups.items():
    if n > 1:
        WARN.append("%s: %d logs (resubmitted); the latest (%s) is used" % (k, n, runs[k]["f"]))

print("Checking %d %s run(s) from %d log file(s)." % (len(runs), MODE, len(files)))
splits, results, sides, caps, testhash = {}, {}, {}, [], defaultdict(set)
sidehash = defaultdict(set)
for key, r in sorted(runs.items()):
    K, TAG, SG, M, S = key
    f, txt = r["f"], r["txt"]
    sg = float(SG)
    if not re.search(r"Job end.*rc=0", txt):
        bad("%s: no 'Job end rc=0'" % f); continue
    if r["CONT"] != 0 or re.search(r"CONT=[1-9]", txt):
        bad("%s: continuation (time budget hit, not bit-reproducible)" % f)
    if (K, TAG) not in CLS or SG not in SIGMAS or M not in MODELS:
        bad("%s: cell (K=%d, %s, sigma=%s, %s) is not in the pre-registered design" % (f, K, TAG, SG, M)); continue
    m_out = re.match(r"results_v155_k(\d+)_(\w+?)_s([\d.]+)$", r["OUT"])
    if not m_out or int(m_out.group(1)) != K or m_out.group(2) != TAG or float(m_out.group(3)) != sg:
        bad("%s: result folder %s does not match K=%d %s sigma=%s" % (f, r["OUT"], K, TAG, SG))
    if not re.search(r"PROTOCOL RESOLVED.*loss=ce", txt): bad("%s: loss is not ce" % f)
    if not re.search(r"PROTOCOL RESOLVED.*n_classes=%d " % K, txt): bad("%s: n_classes != %d" % (f, K))
    if not re.search(r"PROTOCOL RESOLVED.*classes=\[" + re.escape(CLS[(K, TAG)]) + r"\]", txt):
        bad("%s: class list != [%s]" % (f, CLS[(K, TAG)]))
    if not re.search(r"\[MULTICLASS\].*n_obs=%d " % W, txt): bad("%s: n_obs != %d" % (f, W))
    mv = re.search(r"VERIFIED.*total=(\d+) .*OK", txt)
    if not mv: bad("%s: parameter check not OK" % f)
    elif int(mv.group(1)) != PARAMS[K][M]: bad("%s: %s params %s != pre-registered %d" % (f, M, mv.group(1), PARAMS[K][M]))
    if not re.search(r"DETERMINISM.*ENABLED", txt, re.I): bad("%s: determinism not enabled" % f)
    if "max_epochs=%d " % MAXEP not in txt: bad("%s: max_epochs is not the pre-registered %d" % (f, MAXEP))
    want_warm = 0 if M == "M4" else 30
    if "[V135] warmup: model=%s min_epochs=%d" % (M, want_warm) not in txt:
        bad("%s: warmup for %s should be %d epochs" % (f, M, want_warm))
    if M == "M5":
        if "[V135] M5 = M3 with front-end frozen" not in txt: bad("%s: M5 freeze banner missing" % f)
        if "[V135] M5 front-end after run: UNCHANGED OK" not in txt: bad("%s: M5 exit self-check missing" % f)
        if "CHANGED -- FAIL" in txt: bad("%s: M5 FRONT-END CHANGED DURING TRAINING" % f)
    # ---- v15.5 noise lines
    if "[V155] FATAL" in txt: bad("%s: [V155] FATAL in log" % f)
    if sg == 0:
        if not re.search(r"\[V155\] NOISE sigma=0 model=%s seed=%d datasets untouched" % (M, S), txt):
            bad("%s: sigma=0 run without the 'datasets untouched' line" % f)
    else:
        m = re.search(r"\[V155\] NOISE sigma=(\S+) model=(M\d) seed=(\d+) .*train_gen_seed=(\d+) val_seed=(\d+) test_seed=(\d+)", txt)
        if not m:
            bad("%s: [V155] NOISE line missing" % f)
        else:
            if float(m.group(1)) != sg: bad("%s: noise sigma %s != job sigma %s" % (f, m.group(1), SG))
            if m.group(2) != M or int(m.group(3)) != S: bad("%s: noise line is for %s seed %s" % (f, m.group(2), m.group(3)))
            want = (155000000 + 10000 * S + code(SG), 15600000 + code(SG), 15500000 + code(SG))
            got = (int(m.group(4)), int(m.group(5)), int(m.group(6)))
            if got != want: bad("%s: generator seeds %s != pre-registered %s" % (f, got, want))
        if not re.search(r"\[V155\] selfcheck noisy images in \[0,1\], finite, same shape: PASS", txt):
            bad("%s: noisy-image self-check missing or not PASS" % f)
    if not re.search(r"\[V155\] selfcheck global RNG .* unchanged by noise setup: PASS", txt):
        bad("%s: global-RNG self-check missing or not PASS" % f)
    mt = RX_TEST.search(txt)
    if not mt or float(mt.group(1)) != sg or int(mt.group(2)) != TEST * K:
        bad("%s: [V155] TESTSET line missing or wrong (sigma/size)" % f)
    else:
        testhash[(K, TAG, SG)].add(mt.group(3))
    if not re.search(r"\[V155\] XEVAL model=%s seed=%d sigma_train=%g .*selfcheck=MATCH" % (M, S, sg), txt):
        bad("%s: cross-evaluation self-check not MATCH (%s)" % (f, "XEVAL FAILED" if "XEVAL FAILED" in txt else
                                                               "skipped" if "XEVAL skipped" in txt else "see log"))
    # ---- split audit (clean data)
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
        splits[key] = dict((name, sp[name].group(7)) for name in sp)
    # ---- result JSON and cross-eval sidecar
    jf = os.path.join("..", r["OUT"], "result_%s_mnist_seed%d_alpha0.0.json" % (M, S))
    if not os.path.exists(jf):
        bad("%s: result JSON missing (%s)" % (f, jf)); continue
    j = json.load(open(jf))
    results[key] = j
    if j.get("stop_epoch") is not None and int(j["stop_epoch"]) >= MAXEP:
        msg = "%s: stopped at the %d-epoch cap (best epoch %s)" % (key, MAXEP, j.get("best_epoch"))
        caps.append(key)
        if MODE == "smoke":
            bad(msg + " -- pre-registered smoke stop rule")
        else:
            WARN.append(msg)
    sf = os.path.join("..", r["OUT"], "v155_xeval_%s_seed%d.json" % (M, S))
    if not os.path.exists(sf):
        bad("%s: cross-eval sidecar missing (%s)" % (f, sf)); continue
    s = json.load(open(sf))
    sides[key] = s
    if s.get("selfcheck") != "MATCH": bad("%s: sidecar selfcheck %s" % (f, s.get("selfcheck")))
    if float(s.get("sigma_train", -1)) != sg: bad("%s: sidecar sigma_train %s != %s" % (f, s.get("sigma_train"), SG))
    if sorted(float(x) for x in s.get("acc", {})) != EVALS: bad("%s: sidecar eval sigmas %s" % (f, sorted(s.get("acc", {}))))
    if sorted(float(x) for x in s.get("acc_sp", {})) != EVAL_SP: bad("%s: sidecar salt-and-pepper levels %s" % (f, sorted(s.get("acc_sp", {}))))
    a_tr = s.get("acc", {}).get("%g" % sg)
    ja = float(j.get("test_acc", "nan"))
    if a_tr is None or not (abs(a_tr - ja) < 1e-6 or abs(100 * a_tr - ja) < 1e-4):
        bad("%s: sidecar accuracy at the job's sigma (%s) != result test_acc (%s)" % (f, a_tr, ja))
    if s.get("predictions", {}).get("g%g" % sg) != j.get("test_predictions"):
        bad("%s: sidecar predictions at the job's sigma differ from the result JSON's" % f)
    if mt and s.get("test_hash", {}).get("g%g" % sg) != mt.group(3):
        bad("%s: sidecar test hash at the job's sigma != the log's TESTSET hash" % f)
    if sg > 0 and not (s.get("train_fetches", 0) > 0 and s.get("train_gen_seed") == 155000000 + 10000 * S + code(SG)):
        bad("%s: training noise not drawn, or wrong generator seed in sidecar" % f)
    for lv, h_ in s.get("test_hash", {}).items():
        sidehash[(K, TAG, lv)].add(h_)

# ---- cross-log checks ------------------------------------------------------------
same = defaultdict(list)
for (K, TAG, SG, M, S), s in splits.items():
    same[(K, TAG, S)].append(((SG, M), s))
for cell, lst in sorted(same.items()):
    for part in ("train", "val", "test"):
        hs = set(s[part] for _, s in lst)
        if len(hs) != 1:
            bad("%s: models/sigmas saw DIFFERENT clean %s sets %s" % (cell, part, sorted(set((am, s[part]) for am, s in lst))[:4]))
tests = defaultdict(set)
for (K, TAG, SG, M, S), s in splits.items():
    tests[(K, TAG)].add(s["test"])
for kt, hs in tests.items():
    if len(hs) != 1: bad("%s: clean test set differs between runs (%d hashes)" % (kt, len(hs)))
trains = defaultdict(set)
for (K, TAG, S), lst in same.items():
    trains[(K, TAG)].add(lst[0][1]["train"])
for kt, hs in trains.items():
    nseeds = len(set(S for (K, TAG, S) in same if (K, TAG) == kt))
    if len(hs) != nseeds: bad("%s: %d seeds but only %d distinct train sets" % (kt, nseeds, len(hs)))
for c, hs in sorted(testhash.items()):
    if len(hs) != 1: bad("%s: noisy test set differs between runs (%d hashes) -- contrasts would not be paired" % (c, len(hs)))
for c, hs in sorted(sidehash.items()):
    if len(hs) != 1: bad("%s: cross-eval test set differs between sidecars (%d hashes)" % (c, len(hs)))
n_lv = len(set(c[2] for c in sidehash))
print("Splits: %d (K, subset, seed) groups; every model and sigma within a group saw identical clean images; "
      "one clean test set per (K, subset)." % len(same))
print("Noisy test sets: %d (K, subset, sigma) cells, one hash each; cross-eval sets: %d levels x %d (K, subset), one hash each."
      % (len(testhash), n_lv, len(set(c[:2] for c in sidehash))))

# ---- completeness ------------------------------------------------------------------
want_seeds = [SMOKE_SEED] if MODE == "smoke" else [s for s in SEEDS if s != SMOKE_SEED]
missing = [(K, TAG, SG, M, S) for K in KS for TAG in TAGS[K] for SG in SIGMAS for M in MODELS
           for S in want_seeds if (K, TAG, SG, M, S) not in results]
if missing:
    bad("%d expected run(s) missing, e.g. %s" % (len(missing), missing[:3]))


# ---- reproductions -------------------------------------------------------------
def compare(key, verbose):
    K, TAG, SG, M, S = key
    ref, src = stored_ref(K, TAG, M, S)
    if not os.path.exists(ref):
        return None, "reference missing: %s" % ref
    old, new = json.load(open(ref)), results[key]
    ok = all(old.get(k) == new.get(k) for k in ("test_acc", "best_epoch", "stop_epoch"))
    pk = "test_predictions" if "test_predictions" in old else "predictions"
    ok = ok and pk in old and old.get(pk) == new.get("test_predictions")
    if verbose:
        for k in ("test_acc", "best_epoch", "stop_epoch"):
            print("    %s K=%d %s %-16s ref(%s)=%s  new=%s  %s" % (M, K, TAG, k, src, old.get(k), new.get(k),
                                                               "MATCH" if old.get(k) == new.get(k) else "DIFFERENT"))
        print("    %s K=%d %s %-16s %d per-image  %s" % (M, K, TAG, "test_predictions", len(new.get("test_predictions") or []),
                                                     "MATCH" if old.get(pk) == new.get("test_predictions") else "DIFFERENT"))
    return ok, src


zero = sorted(k for k in results if float(k[2]) == 0)
if MODE == "smoke":
    print("\nReproductions, sigma=0 seed 42 (pre-registered stop rule: K=5 A, all five models, EXACT):")
    for M in MODELS:
        key = [k for k in zero if k[:2] == (5, "A") and k[3] == M and k[4] == SMOKE_SEED]
        if not key:
            bad("reproduction %s: no sigma=0 smoke result for K=5 A" % M); continue
        ok, src = compare(key[0], True)
        if ok is None: bad("reproduction %s: %s" % (M, src))
        elif not ok: bad("reproduction %s vs %s: DIFFERENT -- pre-registered stop rule" % (M, src))
n_ok = n_ref = 0
for k in zero:
    if MODE == "smoke" and k[:2] == (5, "A"):
        continue
    ok, src = compare(k, False)
    if ok is None:
        WARN.append("sigma=0 %s K=%d %s seed %d: no stored reference (%s)" % (k[3], k[0], k[1], k[4], src))
        continue
    n_ref += 1; n_ok += int(ok)
    if not ok:
        bad("sigma=0 %s K=%d %s seed %d differs from the stored %s run (pre-registered: must reproduce exactly)" % (
            k[3], k[0], k[1], k[4], src))
print("sigma=0 %s runs reproducing the stored v13/v13.5/v14 runs exactly: %d/%d with a reference%s"
      % (MODE, n_ok, n_ref, " (plus the K=5 A gate above)" if MODE == "smoke" else ""))

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
    print("Sanity only (not a result): smoke runs must sit well above chance (1/K):")
    for k, j in sorted(results.items(), key=lambda kv: (kv[0][0], kv[0][1], float(kv[0][2]), kv[0][3])):
        print("  K=%-2d %-3s sigma=%-4s %s  test_acc=%.4f  best=%s stop=%s" % (k[0], k[1], k[2], k[3], float(j.get("test_acc")),
                                                                          j.get("best_epoch"), j.get("stop_epoch")))
    print("Then: ./submit_v155_grid.sh")
else:
    print("Then: cd .. && python3 scripts/collect_v155.py | tee collect_v155.log && python3 scripts/plot_v155.py")
