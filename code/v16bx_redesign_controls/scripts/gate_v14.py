#!/usr/bin/env python3
"""
v14 INTEGRITY GATE. Called by pbs/check_v14.sh from inside pbs/:
    python3 ../scripts/gate_v14.py smoke|grid

Every job log (latest log per run if a job was resubmitted):
  rc=0; loss=ce; n_classes=K; class list; n_obs=15; params VERIFIED OK;
  determinism on; CONT=0; warmup min_epochs = 30 (M1, M2, M3, M5) or 0 (M4);
  [V14] SPLIT: train n = N*K, val n = 125*K, test n = 200*K, K labels in each,
  zero overlap between splits; M5: freeze banner + exit self-check UNCHANGED OK;
  result JSON present.
Across logs:
  - every model at the same (K, subset, N, seed) trained/validated/tested on the
    SAME images (train/val/test hashes equal);
  - the test set is identical across every N, seed and model within (K, subset);
  - different seeds drew different train sets.
Smoke only: the two N=500 reproductions (w15: M3 vs v13, M5 vs v13.5) must match
  EXACTLY (test_acc, best/stop epoch, per-image predictions); any run at the
  max-epoch cap (200*500/N) STOPS the gate (pre-registered stop rule).
Grid: every (K, subset, N, model) cell must have all 10 seeds; cap hits reported.
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
    sys.exit("usage: gate_v14.py smoke|grid")
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/../pbs")


def cfg(*names):
    cmd = "source ./config14.sh; " + "; ".join(f'echo "${n}"' for n in names)
    return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, check=True).stdout.split("\n")[:len(names)]


KS, NS, MODELS, SEEDS, SMOKE_SEED, VAL, TEST, V13, V135, W = cfg(
    "GRID_KS", "GRID_NS", "GRID_MODELS", "GRID_SEEDS", "SMOKE_SEED",
    "PER_CLASS_VAL", "PER_CLASS_TEST", "V13_ROOT", "V135_ROOT", "W_OBS")
KS, NS, SEEDS = [int(x) for x in KS.split()], [int(x) for x in NS.split()], [int(x) for x in SEEDS.split()]
MODELS, SMOKE_SEED, VAL, TEST, W = MODELS.split(), int(SMOKE_SEED), int(VAL), int(TEST), int(W)
MODELS_FOR = {int(n): ms.split() for n, ms in (l.split(":") for l in subprocess.run(
    ["bash", "-c", "source ./config14.sh; for n in $GRID_NS; do echo $n:$(models_for $n); done"],
    capture_output=True, text=True, check=True).stdout.strip().split("\n"))}
CAP = {int(n): int(c) for n, c in (l.split() for l in subprocess.run(
    ["bash", "-c", "source ./config14.sh; for n in $GRID_NS 500; do echo $n $(max_epochs_for $n); done"],
    capture_output=True, text=True, check=True).stdout.strip().split("\n"))}
TAGS = {5: ["A", "B"], 10: ["all"]}
CLS = {(5, "A"): "1, 3, 5, 8, 9", (5, "B"): "0, 2, 6, 8, 9", (10, "all"): "0, 1, 2, 3, 4, 5, 6, 7, 8, 9"}
REPRO_REF = {"M3": f"{V13}/results_v13_k5_A_w{W}/result_M3_mnist_seed42_alpha0.0.json",
             "M5": f"{V135}/results_v135_k5_A_w{W}/result_M5_mnist_seed42_alpha0.0.json"}

FAIL = []
WARN = []


def bad(msg):
    print(f"  FAIL: {msg}")
    FAIL.append(msg)


def jobid(f):
    m = re.search(r"\.o(\d+)", f)
    return int(m.group(1)) if m else 0


pats = ["v14s_*.o*", "v14r_*.o*"] if MODE == "smoke" else ["v14g*.o*"]
files = sorted({f for p in pats for f in glob.glob(p)}, key=jobid)
if not files:
    print(f"No {MODE} logs found in pbs/. Check qstat.")
    sys.exit(1)

RX_HEAD = re.compile(r"^\[V14\] K=(\d+) TAG=(\w+) N=(\d+) CLASSES=\S+ MODEL=(M\d) SEED=(\d+) OUT=(\S+) CONT=(\d+)", re.M)
RX_SPLIT = re.compile(r"^\[V14\] SPLIT (train|val|test)\s+n=(\d+) labels=(\S+) per_class=(\S+) min=(\d+) max=(\d+) hash=(\w+)", re.M)
RX_OVER = re.compile(r"^\[V14\] SPLIT overlap train&val=(\d+) train&test=(\d+) val&test=(\d+)", re.M)

runs = {}                                         # (K, TAG, N, M, SEED) -> info; latest log wins
dups = defaultdict(int)
for f in files:
    txt = open(f, errors="replace").read()
    h = RX_HEAD.search(txt)
    if not h:
        bad(f"{f}: no [V14] header line (job died before start? cat it)")
        continue
    K, TAG, N, M, SEED, OUT, CONT = int(h[1]), h[2], int(h[3]), h[4], int(h[5]), h[6], int(h[7])
    key = (K, TAG, N, M, SEED)
    dups[key] += 1
    runs[key] = dict(f=f, txt=txt, OUT=OUT, CONT=CONT)
for k, n in dups.items():
    if n > 1:
        WARN.append(f"{k}: {n} logs (resubmitted); the latest ({runs[k]['f']}) is used")

print(f"Checking {len(runs)} {MODE} run(s) from {len(files)} log file(s).")
splits = {}
results = {}
for key, r in sorted(runs.items()):
    K, TAG, N, M, SEED = key
    f, txt = r["f"], r["txt"]
    if f"max_epochs={CAP[N]} " not in txt:
        bad(f"{f}: max_epochs is not the pre-registered {CAP[N]} for N={N}")
    if not re.search(r"Job end.*rc=0", txt):
        bad(f"{f}: no 'Job end rc=0'"); continue
    if r["CONT"] != 0 or re.search(r"CONT=[1-9]", txt):
        bad(f"{f}: continuation (time budget hit, not bit-reproducible)")
    if not re.search(r"PROTOCOL RESOLVED.*loss=ce", txt): bad(f"{f}: loss is not ce")
    if not re.search(rf"PROTOCOL RESOLVED.*n_classes={K} ", txt): bad(f"{f}: n_classes != {K}")
    if not re.search(r"PROTOCOL RESOLVED.*classes=\[" + re.escape(CLS[(K, TAG)]) + r"\]", txt):
        bad(f"{f}: class list != [{CLS[(K, TAG)]}]")
    if not re.search(rf"\[MULTICLASS\].*n_obs={W} ", txt): bad(f"{f}: n_obs != {W}")
    if not re.search(r"VERIFIED.*OK", txt): bad(f"{f}: parameter check not OK")
    if not re.search(r"DETERMINISM.*ENABLED", txt, re.I): bad(f"{f}: determinism not enabled")
    want_warm = 0 if M == "M4" else 30
    if f"[V135] warmup: model={M} min_epochs={want_warm}" not in txt:
        got = re.findall(r"\[V135\] warmup: model=\S+ min_epochs=\d+", txt)
        bad(f"{f}: warmup for {M} should be {want_warm} epochs, log says {got or 'nothing'}")
    if M == "M5":
        if "[V135] M5 = M3 with front-end frozen" not in txt: bad(f"{f}: M5 freeze banner missing")
        if "[V135] M5 front-end after run: UNCHANGED OK" not in txt: bad(f"{f}: M5 exit self-check missing")
        if "CHANGED -- FAIL" in txt: bad(f"{f}: M5 FRONT-END CHANGED DURING TRAINING")
    sp = {m[1]: m for m in RX_SPLIT.finditer(txt)}
    ov = RX_OVER.search(txt)
    if len(sp) != 3 or not ov:
        bad(f"{f}: [V14] SPLIT audit lines missing ({'SPLIT-AUDIT FAILED' if 'SPLIT-AUDIT FAILED' in txt else 'not printed'})")
    else:
        for name, n_exp in (("train", N * K), ("val", VAL * K), ("test", TEST * K)):
            m = sp[name]
            if int(m[2]) != n_exp: bad(f"{f}: {name} n={m[2]}, expected {n_exp}")
            if len(m[3].split(",")) != K or int(m[5]) == 0: bad(f"{f}: {name} does not cover all {K} classes ({m[4]})")
        if any(int(x) for x in ov.groups()): bad(f"{f}: images shared between splits {ov.groups()}")
        splits[key] = {**{name: sp[name][7] for name in sp}, "train_pc": sp["train"][4]}
    jf = os.path.join("..", r["OUT"], f"result_{M}_mnist_seed{SEED}_alpha0.0.json")
    if not os.path.exists(jf):
        bad(f"{f}: result JSON missing ({jf})"); continue
    j = json.load(open(jf))
    results[key] = j
    if j.get("stop_epoch") is not None and int(j["stop_epoch"]) >= CAP[N]:
        msg = f"{key}: stopped at the {CAP[N]}-epoch cap (best epoch {j.get('best_epoch')})"
        (bad if MODE == "smoke" else WARN.append)(
            msg + (" -- pre-registered smoke stop rule: do not submit the grid, paste this" if MODE == "smoke" else ""))

# ---- cross-log checks ------------------------------------------------------------
same = defaultdict(list)
for (K, TAG, N, M, SEED), s in splits.items():
    same[(K, TAG, N, SEED)].append((M, s))
n_shared = 0
for cell, lst in same.items():
    for part in ("train", "val", "test"):
        hs = {s[part] for _, s in lst}
        if len(hs) != 1:
            bad(f"{cell}: models saw DIFFERENT {part} sets: " + ", ".join(f"{m}={s[part]}" for m, s in lst))
    n_shared += len(lst) > 1
tests = defaultdict(set)
for (K, TAG, N, M, SEED), s in splits.items():
    tests[(K, TAG)].add(s["test"])
for kt, hs in sorted(tests.items()):
    if len(hs) != 1: bad(f"K={kt[0]} {kt[1]}: test set changed across runs ({len(hs)} different test hashes)")
seedtrain = defaultdict(dict)
for (K, TAG, N, M, SEED), s in splits.items():
    seedtrain[(K, TAG, N)][SEED] = s["train"]
for cell, d in seedtrain.items():
    if len(d) > 1 and len(set(d.values())) != len(d):
        bad(f"{cell}: two seeds drew the identical train set")
print(f"Same-data check: {n_shared} (K, subset, N, seed) group(s) with >1 model, all models on identical splits"
      if not any("DIFFERENT" in x for x in FAIL) else "Same-data check FAILED (see above)")
print(f"Fixed test set: {len(tests)} (K, subset) group(s), one test hash each" if all(len(h) == 1 for h in tests.values()) else "")

# ---- completeness ------------------------------------------------------------------
print("\nCompleteness (result JSONs on disk, all seeds incl. smoke):")
for K in KS:
    for TAG in TAGS[K]:
        for N in NS:
            row = []
            for M in MODELS_FOR[N]:
                d = os.path.join("..", f"results_v14_k{K}_{TAG}_n{N}")
                have = len(glob.glob(os.path.join(d, f"result_{M}_mnist_seed*_alpha0.0.json")))
                need = 1 if MODE == "smoke" else len(SEEDS)
                row.append(f"{M}:{have}")
                if have < need: bad(f"K={K} {TAG} N={N} {M}: {have} result(s), expected >= {need}")
            print(f"  K={K:<2} {TAG:>3} N={N:<3}  " + "  ".join(row))
if MODE == "grid":
    exp = {(K, T, N, M, S) for K in KS for T in TAGS[K] for N in NS for M in MODELS_FOR[N] for S in SEEDS if S != SMOKE_SEED}
    missing = exp - set(runs)
    if missing: bad(f"{len(missing)} grid run(s) have no log, e.g. {sorted(missing)[:3]}")

# ---- reproductions (smoke) -----------------------------------------------------------
if MODE == "smoke":
    print("\nReproductions at N=500, w15 (K=5 A, seed 42, patched repo; M3 vs v13, M5 vs v13.5) -- must match EXACTLY:")
    for M, ref in REPRO_REF.items():
        key = (5, "A", 500, M, 42)
        if key not in results:
            bad(f"reproduction {M}: no result (log v14r_{M}.o*)"); continue
        if not os.path.exists(ref):
            bad(f"reproduction {M}: reference missing {ref}"); continue
        old, new = json.load(open(ref)), results[key]
        for k in ("test_acc", "best_epoch", "stop_epoch"):
            ok = old.get(k) == new.get(k)
            print(f"  {M} {k:<11} ref={old.get(k)}  new={new.get(k)}  {'MATCH' if ok else 'DIFFERENT'}")
            if not ok: bad(f"reproduction {M}: {k} differs")
        pk = next((k for k in ("predictions", "test_predictions", "preds") if k in old and k in new), None)
        if pk:
            ok = old[pk] == new[pk]
            print(f"  {M} {pk:<11} {len(new[pk])} per-image predictions  {'MATCH' if ok else 'DIFFERENT'}")
            if not ok: bad(f"reproduction {M}: predictions differ")
        else:
            print(f"  {M} (no per-image predictions key in both files; matched on scalars only)")
    if (5, "A", 500, "M3", 42) in splits and (5, "A", 500, "M5", 42) in splits:
        print("  (M3 and M5 reproductions saw identical splits: checked above)")

# ---- verdict -------------------------------------------------------------------------
if WARN:
    print("\nNotes:")
    for w in WARN: print("  - " + w)
print("\nSmoke train class balance at N=50 (seed 42): " + "; ".join(
    f"K={k[0]} {k[1]} [{s['train_pc']}]" for k, s in sorted(splits.items()) if k[2] == 50 and k[3] == "M3")
      if MODE == "smoke" else "")
flag = ".smoke_ok" if MODE == "smoke" else ".grid_ok"
if FAIL:
    print(f"\nGATE FAILED ({len(FAIL)} problem(s)) -- do not read accuracies. Paste this output.")
    if os.path.exists(flag): os.remove(flag)
    sys.exit(1)
open(flag, "w").write("ok\n")
print("\nALL CHECKS PASSED")
if MODE == "smoke":
    print("Sanity only (not a result): smoke test_acc must sit well above chance (1/K):")
    for k, j in sorted(results.items()):
        print(f"  K={k[0]:<2} {k[1]:>3} N={k[2]:<3} {k[3]}  test_acc={j.get('test_acc')}  best={j.get('best_epoch')} stop={j.get('stop_epoch')}")
    print("Then: ./submit_v14_grid.sh")
else:
    print("Then: cd .. && python3 scripts/collect_v14.py && python3 scripts/plot_v14.py")
