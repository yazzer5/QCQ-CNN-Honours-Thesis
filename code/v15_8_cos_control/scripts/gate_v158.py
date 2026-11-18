#!/usr/bin/env python3
"""
v15.8 integrity gate. Called by pbs/check_v158.sh from inside pbs/:
    python3 ../scripts/gate_v158.py smoke|grid
Standard library only (python3 >= 3.6). Prints no accuracy unless everything passes.

Every job log (latest per run): rc=0; CONT=0; loss=ce; n_classes=K; class list; n_obs=15;
params VERIFIED with the pre-registered total (M6 = M4's); determinism; max_epochs=200;
warmup 30 (M3, M5) or 0 (M4, M6); [V14] split sizes/coverage/no overlap; M5 freeze checks;
M6: [V158] front-end self-check PASS; [V155] noise line with the pre-registered generator
seeds, image and RNG self-checks PASS; cross-eval MATCH; sidecar complete and consistent.
Pairing with v15.5 (the stored M3/M4/M5 runs this analysis reuses):
  - every sidecar's test-set hash at every eval level == v15.5's (same noisy test images);
  - clean [V14] train/val/test hashes == v15.5's M4 log for the same (K, subset, sigma, seed)
    when that log is still on disk (otherwise a note);
Smoke: M3/M4/M5 seed 42 at K=5 A sigma=0.3 reproduce v15.5 BIT-EXACTLY (test_acc, best/stop
  epoch, every test prediction, every cross-eval prediction) -- stop rule; any cap hit stops.
Grid: all M6 grid runs present; the stored v15.5 M3/M4/M5 result + sidecar exist for every
  (cell, sigma, seed) the analysis uses (seeds in GRID_SEEDS).
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
    sys.exit("usage: gate_v158.py smoke|grid")
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/../pbs")


def cfg(*names):
    cmd = "source ./config158.sh; " + "; ".join('echo "${%s}"' % n for n in names)
    out = subprocess.run(["bash", "-c", cmd], stdout=subprocess.PIPE, universal_newlines=True, check=True).stdout
    return out.split("\n")[:len(names)]


(KS, SIGMAS, EVALS, EVAL_SP, NEW, REUSED, SEEDS, SMOKE_SEED, NPC, VAL, TEST, W, MAXEP, V155,
 RK, RTAG, RSG) = cfg("GRID_KS", "GRID_SIGMAS", "EVAL_SIGMAS", "EVAL_SP", "NEW_MODELS", "REUSED_MODELS",
                      "GRID_SEEDS", "SMOKE_SEED", "N_PER_CLASS", "PER_CLASS_VAL", "PER_CLASS_TEST", "W_OBS",
                      "MAX_EPOCHS", "V155_ROOT", "REPRO_K", "REPRO_TAG", "REPRO_SIGMA")
KS = [int(x) for x in KS.split()]
SIGMAS = SIGMAS.split()
EVALS = sorted(float(x) for x in EVALS.split())
EVAL_SP = sorted(float(x) for x in EVAL_SP.split())
NEW, REUSED = NEW.split(), REUSED.split()
SEEDS = [int(x) for x in SEEDS.split()]
SMOKE_SEED, NPC, VAL, TEST, W, MAXEP, RK = int(SMOKE_SEED), int(NPC), int(VAL), int(TEST), int(W), int(MAXEP), int(RK)
TAGS = {5: ["A", "B"], 10: ["all"]}
CLS = {(5, "A"): "1, 3, 5, 8, 9", (5, "B"): "0, 2, 6, 8, 9", (10, "all"): "0, 1, 2, 3, 4, 5, 6, 7, 8, 9"}
PARAMS = {5: dict(M3=1725, M4=1719, M5=1725, M6=1719), 10: dict(M3=1805, M4=1799, M5=1805, M6=1799)}

FAIL, WARN = [], []


def bad(msg):
    print("  FAIL: " + msg)
    FAIL.append(msg)


def code(x):
    return int(round(1000 * float(x)))


def jobid(f):
    m = re.search(r"\.o(\d+)", f)
    return int(m.group(1)) if m else 0


def v155_paths(K, TAG, SG, M, S):
    d = os.path.join(V155, "results_v155_k%d_%s_s%s" % (K, TAG, SG))
    return (os.path.join(d, "result_%s_mnist_seed%d_alpha0.0.json" % (M, S)),
            os.path.join(d, "v155_xeval_%s_seed%d.json" % (M, S)))


if not os.path.isdir(V155):
    print("FATAL: V155_ROOT %s not found (edit pbs/config158.sh)" % V155)
    sys.exit(1)

files = sorted(glob.glob("v158s_*.o*" if MODE == "smoke" else "v158g_*.o*"), key=jobid)
if not files:
    print("No %s logs found in pbs/. Check qstat." % MODE)
    sys.exit(1)

RX_HEAD = re.compile(r"^\[V158\] K=(\d+) TAG=(\w+) SIGMA=(\S+) CLASSES=\S+ MODEL=(M\d) SEED=(\d+) OUT=(\S+) CONT=(\d+)", re.M)
RX_SPLIT = re.compile(r"^\[V14\] SPLIT (train|val|test)\s+n=(\d+) labels=(\S+) per_class=(\S+) min=(\d+) max=(\d+) hash=(\w+)", re.M)
RX_OVER = re.compile(r"^\[V14\] SPLIT overlap train&val=(\d+) train&test=(\d+) val&test=(\d+)", re.M)
RX_TEST = re.compile(r"^\[V155\] TESTSET sigma=(\S+) n=(\d+) sha=(\w+)", re.M)

runs, dups = {}, defaultdict(int)
for f in files:
    txt = open(f, errors="replace").read()
    h = RX_HEAD.search(txt)
    if not h:
        bad("%s: no [V158] header line (job died before start? cat it)" % f)
        continue
    K, TAG, SG, M, S, OUT, CONT = int(h.group(1)), h.group(2), h.group(3), h.group(4), int(h.group(5)), h.group(6), int(h.group(7))
    key = (K, TAG, SG, M, S)
    dups[key] += 1
    runs[key] = dict(f=f, txt=txt, OUT=OUT, CONT=CONT)
for k, n in dups.items():
    if n > 1:
        WARN.append("%s: %d logs (resubmitted); the latest (%s) is used" % (k, n, runs[k]["f"]))

print("Checking %d %s run(s) from %d log file(s); v15.5 reference: %s" % (len(runs), MODE, len(files), V155))
results, sides, splits, caps = {}, {}, {}, []
v155_side_hash = {}


def v155_hashes(K, TAG):
    if (K, TAG) not in v155_side_hash:
        hs = None
        for f in sorted(glob.glob(os.path.join(V155, "results_v155_k%d_%s_s*" % (K, TAG), "v155_xeval_M4_seed%d.json" % SMOKE_SEED))):
            hs = json.load(open(f)).get("test_hash")
            break
        v155_side_hash[(K, TAG)] = hs
    return v155_side_hash[(K, TAG)]


def v155_log_splits(K, TAG, SG, S):
    pats = ["v155s_k%d%s_s%s_M4.o*" % (K, TAG, SG)] if S == SMOKE_SEED else ["v155g_k%d%s_s%s_M4_r%d.o*" % (K, TAG, SG, S)]
    fs = sorted([f for p in pats for f in glob.glob(os.path.join(V155, "pbs", p))], key=jobid)
    if not fs:
        return None
    return dict((m.group(1), m.group(7)) for m in RX_SPLIT.finditer(open(fs[-1], errors="replace").read()))


for key, r in sorted(runs.items()):
    K, TAG, SG, M, S = key
    f, txt = r["f"], r["txt"]
    sg = float(SG)
    if not re.search(r"Job end.*rc=0", txt):
        bad("%s: no 'Job end rc=0'" % f); continue
    if r["CONT"] != 0 or re.search(r"CONT=[1-9]", txt):
        bad("%s: continuation (time budget hit, not bit-reproducible)" % f)
    in_design = (K, TAG) in CLS and SG in SIGMAS and (M in NEW or (MODE == "smoke" and M in REUSED and (K, TAG, SG) == (RK, RTAG, RSG)))
    if not in_design:
        bad("%s: run (K=%d, %s, sigma=%s, %s) is not in the pre-registered %s design" % (f, K, TAG, SG, M, MODE)); continue
    if r["OUT"] != "results_v158_k%d_%s_s%s" % (K, TAG, SG):
        bad("%s: result folder %s does not match K/TAG/sigma" % (f, r["OUT"]))
    if not re.search(r"PROTOCOL RESOLVED.*loss=ce", txt): bad("%s: loss is not ce" % f)
    if not re.search(r"PROTOCOL RESOLVED.*n_classes=%d " % K, txt): bad("%s: n_classes != %d" % (f, K))
    if not re.search(r"PROTOCOL RESOLVED.*classes=\[" + re.escape(CLS[(K, TAG)]) + r"\]", txt):
        bad("%s: class list != [%s]" % (f, CLS[(K, TAG)]))
    if not re.search(r"\[MULTICLASS\].*n_obs=%d " % W, txt): bad("%s: n_obs != %d" % (f, W))
    mv = re.search(r"VERIFIED\] (M\d) params.*total=(\d+) .*OK", txt)
    if not mv: bad("%s: parameter check not OK" % f)
    elif mv.group(1) != ("M3" if M == "M5" else M) or int(mv.group(2)) != PARAMS[K][M]:   # M5 builds M3, then freezes
        bad("%s: %s params %s != pre-registered %d" % (f, mv.group(1), mv.group(2), PARAMS[K][M]))
    if not re.search(r"DETERMINISM.*ENABLED", txt, re.I): bad("%s: determinism not enabled" % f)
    if "max_epochs=%d " % MAXEP not in txt: bad("%s: max_epochs is not %d" % (f, MAXEP))
    want_warm = 0 if M in ("M4", "M6") else 30
    if "[V135] warmup: model=%s min_epochs=%d" % (M, want_warm) not in txt:
        bad("%s: warmup for %s should be %d epochs" % (f, M, want_warm))
    if "[GRADCHECK] %s: all trainable tensors receive gradient. OK" % M not in txt:
        bad("%s: gradient check line missing" % f)
    if M == "M5":
        if "[V135] M5 front-end after run: UNCHANGED OK" not in txt or "CHANGED -- FAIL" in txt:
            bad("%s: M5 freeze self-check missing or FAILED" % f)
    if M == "M6":
        if not re.search(r"\[V158\] M6 front-end = Circuit 4 at w=0 .* PASS", txt):
            bad("%s: [V158] M6 front-end self-check missing or not PASS" % f)
    if "[V158] FATAL" in txt or "[V155] FATAL" in txt:
        bad("%s: FATAL in log" % f)
    if sg == 0:
        if not re.search(r"\[V155\] NOISE sigma=0 model=%s seed=%d datasets untouched" % (M, S), txt):
            bad("%s: sigma=0 run without the 'datasets untouched' line" % f)
    else:
        m = re.search(r"\[V155\] NOISE sigma=(\S+) model=(M\d) seed=(\d+) .*train_gen_seed=(\d+) val_seed=(\d+) test_seed=(\d+)", txt)
        want = (155000000 + 10000 * S + code(SG), 15600000 + code(SG), 15500000 + code(SG))
        if not m or float(m.group(1)) != sg or m.group(2) != M or int(m.group(3)) != S \
                or tuple(int(m.group(i)) for i in (4, 5, 6)) != want:
            bad("%s: [V155] NOISE line missing or generator seeds != %s" % (f, want))
        if "[V155] selfcheck noisy images in [0,1], finite, same shape: PASS" not in txt:
            bad("%s: noisy-image self-check missing or not PASS" % f)
    if not re.search(r"\[V155\] selfcheck global RNG .* unchanged by noise setup: PASS", txt):
        bad("%s: global-RNG self-check missing or not PASS" % f)
    mt = RX_TEST.search(txt)
    if not mt or float(mt.group(1)) != sg or int(mt.group(2)) != TEST * K:
        bad("%s: [V155] TESTSET line missing or wrong" % f)
    if not re.search(r"\[V155\] XEVAL model=%s seed=%d sigma_train=%g .*selfcheck=MATCH" % (M, S, sg), txt):
        bad("%s: cross-evaluation self-check not MATCH" % f)
    sp = {mm.group(1): mm for mm in RX_SPLIT.finditer(txt)}
    ov = RX_OVER.search(txt)
    if len(sp) != 3 or not ov:
        bad("%s: [V14] SPLIT audit lines missing" % f)
    else:
        for name, n_exp in (("train", NPC * K), ("val", VAL * K), ("test", TEST * K)):
            if int(sp[name].group(2)) != n_exp: bad("%s: %s n=%s, expected %d" % (f, name, sp[name].group(2), n_exp))
            if len(sp[name].group(3).split(",")) != K or int(sp[name].group(5)) == 0:
                bad("%s: %s does not cover all %d classes" % (f, name, K))
        if any(int(x) for x in ov.groups()): bad("%s: images shared between splits %s" % (f, ov.groups()))
        splits[key] = dict((n_, sp[n_].group(7)) for n_ in sp)
        ref_sp = v155_log_splits(K, TAG, SG, S)
        if ref_sp is None:
            WARN.append("%s: v15.5 M4 log not on disk; clean-split pairing checked through the test hashes only" % (key,))
        elif ref_sp != splits[key]:
            bad("%s: clean splits differ from v15.5's M4 run at the same (K, subset, sigma, seed) -- not paired" % f)
    jf = os.path.join("..", r["OUT"], "result_%s_mnist_seed%d_alpha0.0.json" % (M, S))
    sf = os.path.join("..", r["OUT"], "v155_xeval_%s_seed%d.json" % (M, S))
    if not (os.path.exists(jf) and os.path.exists(sf)):
        bad("%s: result JSON or cross-eval sidecar missing" % f); continue
    j, s = json.load(open(jf)), json.load(open(sf))
    results[key], sides[key] = j, s
    if j.get("model") != M or int(j.get("n_params", -1)) != PARAMS[K][M]:
        bad("%s: result JSON model/n_params %s/%s" % (f, j.get("model"), j.get("n_params")))
    if j.get("stop_epoch") is not None and int(j["stop_epoch"]) >= MAXEP:
        caps.append(key)
        msg = "%s: stopped at the %d-epoch cap" % (key, MAXEP)
        if MODE == "smoke": bad(msg + " -- smoke stop rule")
        else: WARN.append(msg)
    if s.get("selfcheck") != "MATCH": bad("%s: sidecar selfcheck %s" % (f, s.get("selfcheck")))
    if float(s.get("sigma_train", -1)) != sg: bad("%s: sidecar sigma_train %s" % (f, s.get("sigma_train")))
    if sorted(float(x) for x in s.get("acc", {})) != EVALS or sorted(float(x) for x in s.get("acc_sp", {})) != EVAL_SP:
        bad("%s: sidecar eval levels incomplete" % f)
    a_tr, ja = s.get("acc", {}).get("%g" % sg), float(j.get("test_acc", "nan"))
    if a_tr is None or not (abs(a_tr - ja) < 1e-6 or abs(100 * a_tr - ja) < 1e-4):
        bad("%s: sidecar accuracy at the job's sigma != result test_acc" % f)
    if s.get("predictions", {}).get("g%g" % sg) != j.get("test_predictions"):
        bad("%s: sidecar predictions at the job's sigma differ from the result JSON's" % f)
    if mt and s.get("test_hash", {}).get("g%g" % sg) != mt.group(3):
        bad("%s: sidecar test hash != log TESTSET hash" % f)
    if sg > 0 and not (s.get("train_fetches", 0) > 0 and s.get("train_gen_seed") == 155000000 + 10000 * S + code(SG)):
        bad("%s: training noise not drawn, or wrong generator seed" % f)
    ref_h = v155_hashes(K, TAG)
    if ref_h is None:
        bad("%s: no v15.5 sidecar found for K=%d %s to pair test sets against" % (f, K, TAG))
    elif s.get("test_hash") != ref_h:
        bad("%s: noisy test sets differ from v15.5's at some eval level -- contrasts would not be paired" % f)

print("Pairing: every run's noisy test sets (all %d Gaussian + %d salt-and-pepper levels) checked against v15.5's."
      % (len(EVALS), len(EVAL_SP)))
n_split = sum(1 for k in splits if v155_log_splits(k[0], k[1], k[2], k[4]) is not None)
print("Clean splits checked against v15.5's M4 logs: %d/%d runs (rest: log not on disk, see notes)." % (n_split, len(splits)))
same = defaultdict(set)
for (K, TAG, SG, M, S), sp_ in splits.items():
    same[(K, TAG, S)].add((sp_["train"], sp_["val"], sp_["test"]))
for c, hs in same.items():
    if len(hs) != 1: bad("%s: models/sigmas saw different clean splits" % (c,))

# ---- completeness ------------------------------------------------------------------
if MODE == "smoke":
    want = [(K, TAG, SG, M, SMOKE_SEED) for K in KS for TAG in TAGS[K] for SG in SIGMAS for M in NEW] + \
           [(RK, RTAG, RSG, M, SMOKE_SEED) for M in REUSED]
else:
    want = [(K, TAG, SG, M, S) for K in KS for TAG in TAGS[K] for SG in SIGMAS for M in NEW
            for S in SEEDS if S != SMOKE_SEED]
missing = [k for k in want if k not in results]
if missing:
    bad("%d expected run(s) missing, e.g. %s" % (len(missing), missing[:3]))
if MODE == "grid":
    if not os.path.exists(".smoke_ok"):
        bad("pbs/.smoke_ok missing: the smoke gate never passed")
    sm = [(K, TAG, SG, SMOKE_SEED) for K in KS for TAG in TAGS[K] for SG in SIGMAS]
    for K, TAG, SG, S in sm:
        for M in NEW:
            p = os.path.join("..", "results_v158_k%d_%s_s%s" % (K, TAG, SG), "v155_xeval_%s_seed%d.json" % (M, S))
            if not os.path.exists(p): bad("smoke run %s %s missing from the results folder" % (M, (K, TAG, SG, S)))
    n_ref = 0
    for K in KS:
        for TAG in TAGS[K]:
            for SG in SIGMAS:
                for M in REUSED:
                    for S in SEEDS:
                        jf, sf = v155_paths(K, TAG, SG, M, S)
                        if not (os.path.exists(jf) and os.path.exists(sf)):
                            bad("stored v15.5 run missing: %s" % jf)
                        else:
                            n_ref += 1
                            sv = json.load(open(sf))
                            if sv.get("selfcheck") != "MATCH" or sv.get("test_hash") != v155_hashes(K, TAG):
                                bad("stored v15.5 sidecar not MATCH or not paired: %s" % sf)
    print("Stored v15.5 M3/M4/M5 runs available for the analysis: %d (expect %d)"
          % (n_ref, sum(len(TAGS[K]) for K in KS) * len(SIGMAS) * len(REUSED) * len(SEEDS)))

# ---- reproductions (smoke) -----------------------------------------------------------
if MODE == "smoke":
    print("\nReproductions of v15.5 (seed %d, K=%d %s, sigma=%s; must be EXACT):" % (SMOKE_SEED, RK, RTAG, RSG))
    for M in REUSED:
        key = (RK, RTAG, RSG, M, SMOKE_SEED)
        if key not in results:
            bad("reproduction %s: run missing" % M); continue
        jf, sf = v155_paths(RK, RTAG, RSG, M, SMOKE_SEED)
        if not (os.path.exists(jf) and os.path.exists(sf)):
            bad("reproduction %s: v15.5 reference missing (%s)" % (M, jf)); continue
        old, olds = json.load(open(jf)), json.load(open(sf))
        new, news = results[key], sides[key]
        pk = "test_predictions" if "test_predictions" in old else "predictions"
        parts = [(k, old.get(k) == new.get(k)) for k in ("test_acc", "best_epoch", "stop_epoch")]
        parts.append(("test_predictions (%d)" % len(new.get("test_predictions") or []), old.get(pk) == new.get("test_predictions")))
        parts.append(("cross-eval predictions, all %d levels" % len(news.get("predictions", {})),
                      olds.get("predictions") == news.get("predictions")))
        for name, ok in parts:
            print("    %s %-36s %s" % (M, name, "MATCH" if ok else "DIFFERENT"))
        if not all(ok for _, ok in parts):
            bad("reproduction %s vs v15.5: DIFFERENT -- pre-registered stop rule" % M)

# ---- verdict -------------------------------------------------------------------------
if WARN:
    print("\nNotes:")
    for w_ in WARN[:12]: print("  - " + w_)
    if len(WARN) > 12: print("  - ... %d more" % (len(WARN) - 12))
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
        print("  K=%-2d %-3s sigma_train=%-4s %s  test_acc=%.4f  best=%s stop=%s" % (
            k[0], k[1], k[2], k[3], float(j.get("test_acc")), j.get("best_epoch"), j.get("stop_epoch")))
    print("Then: ./submit_v158_grid.sh")
else:
    print("Then: cd .. && python3 scripts/analyse_v158.py | tee analyse_v158.log")
