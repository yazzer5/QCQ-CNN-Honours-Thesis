#!/usr/bin/env python3
"""gate_v156.py smoke|grid <repo156> <repo155>

Prints counts and checks only; never an accuracy or a metric value.
Writes <repo156>/.smoke_ok or .grid_ok only on a full pass.

Per job (model x arm x cell x seed):
  G1 result JSON, v155 xeval JSON, geom npz present
  G2 reproduction: test_acc, best_epoch, stop_epoch, test_predictions == stored v15.5 run (bit-exact)
  G3 xeval predictions at every eval sigma == stored v15.5 xeval; selfcheck MATCH
  G4 geom meta: model/seed/sigma_train/K match the cell; n_sub = 50K; R = 20; clean_mismatch = 0;
     A1 draw-0 mismatch = 0 at every sigma
  G5 sigma=0 sanity at every layer: trSN = 0, eta = 0, Jtr_ret = 1; no NaN where a value is defined
Across jobs:
  G6 test_hash, subset_hash and noise_hash identical within each (cell, seed)
  G7 Lpix metrics identical within each (cell, seed) across models and arms
  G8 PBS logs: no Traceback / GEOM FAILED / XEVAL FAILED; one GEOM OK per job
"""
import glob
import json
import math
import os
import sys

import numpy as np

MODELS = ("M3", "M4", "M5")
ARMS = ("0", "0.3")
CELLS = (("k5_A", 5), ("k5_B", 5), ("k10_all", 10))
LAYERS = ("Lpix", "L0", "L1", "L2", "L3")
FAILS = []


def fail(msg):
    FAILS.append(msg)


def rdir(repo, tag, cell, arm):
    return os.path.join(repo, "results_%s_%s_s%s" % (tag, cell, arm))


def one(pattern):
    g = sorted(glob.glob(pattern))
    return g[-1] if g else None


def main():
    mode, repo156, repo155 = sys.argv[1], sys.argv[2], sys.argv[3]
    seeds = [42] if mode == "smoke" else list(range(42, 52))
    tag156 = "v156" if glob.glob(os.path.join(repo156, "results_v156_*")) else "v155"
    groups = {}
    n_ok = 0
    secs = []
    for cell, K in CELLS:
        for arm in ARMS:
            for M in MODELS:
                for S in seeds:
                    job = "%s s%s %s seed%d" % (cell, arm, M, S)
                    d6, d5 = rdir(repo156, tag156, cell, arm), rdir(repo155, "v155", cell, arm)
                    r6 = one(os.path.join(d6, "result_%s_mnist_seed%d_*.json" % (M, S)))
                    x6 = os.path.join(d6, "v155_xeval_%s_seed%d.json" % (M, S))
                    g6 = os.path.join(d6, "geom_%s_seed%d.npz" % (M, S))
                    r5 = one(os.path.join(d5, "result_%s_mnist_seed%d_*.json" % (M, S)))
                    x5 = os.path.join(d5, "v155_xeval_%s_seed%d.json" % (M, S))
                    miss = [n for n, p in (("result", r6), ("xeval", x6), ("geom", g6)) if not (p and os.path.exists(p))]
                    if miss:
                        fail("G1 %s missing %s" % (job, ",".join(miss)))
                        continue
                    if not (r5 and os.path.exists(x5)):
                        fail("G2 %s: stored v15.5 run not found in %s" % (job, d5))
                        continue
                    a, b = json.load(open(r6)), json.load(open(r5))
                    for k in ("test_acc", "best_epoch", "stop_epoch", "test_predictions"):
                        if a.get(k) != b.get(k):
                            fail("G2 %s: %s differs from v15.5" % (job, k))
                    xa, xb = json.load(open(x6)), json.load(open(x5))
                    if xa.get("selfcheck") != "MATCH":
                        fail("G3 %s: selfcheck %s" % (job, xa.get("selfcheck")))
                    if xa.get("predictions") != xb.get("predictions"):
                        fail("G3 %s: xeval predictions differ from v15.5" % job)
                    z = np.load(g6)
                    meta, met = json.loads(str(z["meta"])), json.loads(str(z["metrics"]))
                    exp = dict(model=M, seed=S, sigma_train=float(arm), K=K, n_sub=50 * K, R=20, clean_mismatch=0)
                    for k, v in exp.items():
                        if meta.get(k) != v:
                            fail("G4 %s: meta %s=%s expected %s" % (job, k, meta.get(k), v))
                    if any(v != 0 for v in meta.get("a1_mismatch", {"?": -1}).values()):
                        fail("G4 %s: A1 draw-0 mismatch %s" % (job, meta.get("a1_mismatch")))
                    for L in LAYERS:
                        m0 = met.get("%s|0" % L, {})
                        if not (m0.get("trSN") == 0 and m0.get("eta") == 0 and abs(m0.get("Jtr_ret", 9) - 1) < 1e-12):
                            fail("G5 %s: sigma=0 sanity at %s" % (job, L))
                        for s in meta["sigmas"]:
                            for k, v in met["%s|%g" % (L, s)].items():
                                defined = not (k.startswith("J_") and L != "L2")
                                if defined and (v is None or (isinstance(v, float) and math.isnan(v))):
                                    fail("G5 %s: NaN %s at %s sigma=%g" % (job, k, L, s))
                    gk = (cell, S)
                    groups.setdefault(gk, []).append((job, meta["test_hash"], meta["subset_hash"],
                                                      json.dumps(meta["noise_hash"], sort_keys=True),
                                                      json.dumps(dict((k, v) for k, v in met.items()
                                                                      if k.startswith("Lpix|")), sort_keys=True)))
                    secs.append(meta.get("seconds", 0))
                    n_ok += 1
    for gk, rows in groups.items():
        for i, nm in ((1, "test_hash"), (2, "subset_hash"), (3, "noise_hash")):
            if len(set(r[i] for r in rows)) != 1:
                fail("G6 %s seed%d: %s differs across jobs" % (gk[0], gk[1], nm))
        if len(set(r[4] for r in rows)) != 1:
            fail("G7 %s seed%d: Lpix metrics differ across models/arms" % gk)
    logs = glob.glob(os.path.join(repo156, "pbs", "*.o*"))
    if logs:
        n_geom = 0
        for lf in logs:
            t = open(lf, errors="replace").read()
            n_geom += t.count("[V156] GEOM OK")
            for bad in ("Traceback", "GEOM FAILED", "XEVAL FAILED", "GEOM CHECKFAIL"):
                if bad in t:
                    fail("G8 %s contains '%s'" % (os.path.basename(lf), bad))
        if n_geom < n_ok:
            fail("G8 only %d '[V156] GEOM OK' lines in %d logs for %d jobs" % (n_geom, len(logs), n_ok))
    else:
        print("note: no PBS logs found in %s/pbs (G8 skipped)" % repo156)
    expected = len(CELLS) * len(ARMS) * len(MODELS) * len(seeds)
    print("gate %s: %d/%d jobs complete and checked; (cell, seed) groups: %d; geometry pass %.0f-%.0f s"
          % (mode, n_ok, expected, len(groups), min(secs) if secs else 0, max(secs) if secs else 0))
    print("=" * 78)
    if FAILS or n_ok != expected:
        for f in FAILS[:60]:
            print("FAIL", f)
        if len(FAILS) > 60:
            print("... and %d more" % (len(FAILS) - 60))
        print("GATE FAILED (%d problems)" % len(FAILS))
        sys.exit(1)
    open(os.path.join(repo156, ".%s_ok" % mode), "w").write("ok %d jobs\n" % n_ok)
    print("ALL CHECKS PASSED  (wrote .%s_ok)" % mode)


if __name__ == "__main__":
    main()
