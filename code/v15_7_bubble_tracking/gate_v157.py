#!/usr/bin/env python3
"""gate_v157.py smoke|grid <repo157> <repo156> <repo155>   (prints checks only, never a metric)
G1 files present  G2 bit-exact v15.5 reproduction  G3 xeval predictions == v15.5, MATCH
G4 final geometry == stored v15.6 geometry (exact)  G5 tracker: >=8 records, first epoch 0, increasing,
epoch count within 2 of stop_epoch, no error/NaN, sigma=0 sanity, subset hash == v15.6
G6 PBS logs clean, one TRACK OK per job. Writes <repo157>/.smoke_ok / .grid_ok only on a full pass."""
import glob
import json
import math
import os
import sys

import numpy as np

MODELS, ARMS, CELL = ("M3", "M4", "M5"), ("0", "0.3"), "k5_A"
F = []


def one(p):
    g = sorted(glob.glob(p))
    return g[-1] if g else None


def main():
    mode, r7, r6, r5 = sys.argv[1:5]
    seeds = [42] if mode == "smoke" else list(range(42, 52))
    ok, secs, nrec = 0, [], []
    for arm in ARMS:
        d7, d6, d5 = ["%s/results_%s_%s_s%s" % (r, t, CELL, arm) for r, t in ((r7, "v157"), (r6, "v156"), (r5, "v155"))]
        for M in MODELS:
            for S in seeds:
                job = "s%s %s seed%d" % (arm, M, S)
                f = dict(res=one("%s/result_%s_mnist_seed%d_*.json" % (d7, M, S)), xe="%s/v155_xeval_%s_seed%d.json" % (d7, M, S),
                         geo="%s/geom_%s_seed%d.npz" % (d7, M, S), trk="%s/track_%s_seed%d.json" % (d7, M, S),
                         res5=one("%s/result_%s_mnist_seed%d_*.json" % (d5, M, S)), xe5="%s/v155_xeval_%s_seed%d.json" % (d5, M, S),
                         geo6="%s/geom_%s_seed%d.npz" % (d6, M, S))
                miss = [k for k, v in f.items() if not (v and os.path.exists(v))]
                if miss:
                    F.append("G1 %s missing %s" % (job, ",".join(miss)))
                    continue
                a, b = json.load(open(f["res"])), json.load(open(f["res5"]))
                for k in ("test_acc", "best_epoch", "stop_epoch", "test_predictions"):
                    if a.get(k) != b.get(k):
                        F.append("G2 %s: %s differs from v15.5" % (job, k))
                xa, xb = json.load(open(f["xe"])), json.load(open(f["xe5"]))
                if xa.get("selfcheck") != "MATCH" or xa.get("predictions") != xb.get("predictions"):
                    F.append("G3 %s: xeval selfcheck/predictions" % job)
                g7, g6 = np.load(f["geo"]), np.load(f["geo6"])
                if str(g7["metrics"]) != str(g6["metrics"]):
                    F.append("G4 %s: final geometry differs from v15.6" % job)
                m6 = json.loads(str(g6["meta"]))
                t = json.load(open(f["trk"]))
                mt, rec = t["meta"], t["records"]
                ep = [r["epoch"] for r in rec]
                if mt.get("error"):
                    F.append("G5 %s: tracker error %s" % (job, mt["error"]))
                if len(rec) < 8 or not ep or ep[0] != 0 or any(y <= x for x, y in zip(ep, ep[1:])):
                    F.append("G5 %s: records/epochs %s" % (job, ep[:12]))
                if abs(mt.get("epoch_starts", -99) - int(a.get("stop_epoch", -999))) > 2:
                    F.append("G5 %s: epoch_starts %s vs stop_epoch %s" % (job, mt.get("epoch_starts"), a.get("stop_epoch")))
                if mt.get("subset_hash") != m6.get("subset_hash"):
                    F.append("G5 %s: subset hash differs from v15.6" % job)
                for r in rec:
                    for L in ("L0", "L1", "L2", "L3"):
                        z = r["%s|0" % L]
                        if z["eta"] != 0 or z["trSN"] != 0:
                            F.append("G5 %s: sigma=0 sanity epoch %d %s" % (job, r["epoch"], L))
                        v = r["%s|0.3" % L]["radius"]
                        if v is None or math.isnan(v):
                            F.append("G5 %s: NaN radius epoch %d %s" % (job, r["epoch"], L))
                secs.append(mt.get("seconds", 0))
                nrec.append(len(rec))
                ok += 1
    logs = glob.glob(os.path.join(r7, "pbs", "v157*.o*"))
    nt = 0
    for lf in logs:
        s = open(lf, errors="replace").read()
        nt += s.count("[V157] TRACK OK")
        for bad in ("Traceback", "FAILED", "CHECKFAIL", "continuation"):
            if bad in s:
                F.append("G6 %s contains '%s'" % (os.path.basename(lf), bad))
    if logs and nt < ok:
        F.append("G6 only %d TRACK OK lines for %d jobs" % (nt, ok))
    if not logs:
        print("note: no v157 PBS logs found (G6 skipped)")
    exp = len(ARMS) * len(MODELS) * len(seeds)
    print("gate %s: %d/%d jobs checked; checkpoints per job %s-%s; tracking time %.0f-%.0f s"
          % (mode, ok, exp, min(nrec or [0]), max(nrec or [0]), min(secs or [0]), max(secs or [0])))
    print("=" * 78)
    if F or ok != exp:
        for x in F[:60]:
            print("FAIL", x)
        print("GATE FAILED (%d problems)" % len(F))
        sys.exit(1)
    open(os.path.join(r7, ".%s_ok" % mode), "w").write("ok\n")
    print("ALL CHECKS PASSED  (wrote .%s_ok)" % mode)


if __name__ == "__main__":
    main()
