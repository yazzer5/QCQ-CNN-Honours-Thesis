#!/usr/bin/env python3
"""
v16b-ext: collect and run the PRE-REGISTERED analysis (PREREGISTRATION_v16b.md amendment A3). Run from the v16bx repo
root after the grid gate passes:   python3 scripts/collect_v16bx.py | tee collect_v16bx.log

Cell: K=10, N=50, seeds 42-51, paired. New: M4-trim, BFR-ZZ. Stored: M4, M2-w15 (v14), BFR (v16b).
PRIMARY (Holm over the 3, two-sided paired t):  C1 = BFR - M4trim,  C2 = M4trim - M4,  C3 = BFRZZ - BFR.
   C1 + C2 = BFR - M4 = D4 exactly (checked).
E1  M4trim - M4 explains >= 50% of BFR - M4:  mean C2 <= -2.3 pp  AND  C2 detected < 0 (Holm p < .05).
E2  BFRZZ - BFR <= -5 pp:                      mean C3 <= -5.0 pp  AND  C3 detected < 0 (Holm p < .05).
    A threshold met without detection is reported "direction only (not confirmed)".
Secondary (uncorrected): fit / gap decomposition of each contrast (train_acc_at_best, eval mode, best checkpoint);
seed SD per model; g and s of BFR-ZZ vs BFR; descriptive BFRZZ - M2, BFRZZ - M4, M4trim - M2. MDE with every null.
Writes v16bx_runs.csv, v16bx_contrasts.csv, v16bx_decomp.csv, v16bx_gs.csv.
"""
import csv
import json
import os
import subprocess
import sys

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
if not (os.path.exists("pbs/.smoke_ok") and os.path.exists("pbs/.grid_ok")):
    sys.exit("REFUSED: pbs/.smoke_ok and pbs/.grid_ok must both exist (run the gates first).")
out = subprocess.run(["bash", "-c", 'cd pbs && source ./config16bx.sh; echo "$GRID_SEEDS"; echo "$V14_ROOT"; echo "$V16B_ROOT"'],
                     stdout=subprocess.PIPE, universal_newlines=True, check=True).stdout.split("\n")
SEEDS, V14, V16B = [int(s) for s in out[0].split()], out[1], out[2]
MODELS = ["M2", "M4", "M4trim", "BFR", "BFRZZ"]
PARAMS = {"M2": 1766, "M4": 1799, "M4trim": 1643, "BFR": 1799, "BFRZZ": 1799}
D4_V16B = -4.66


def path(M, S):
    return {"M2": "%s/results_v14_k10_all_n50/result_M2_mnist_seed%d_alpha0.0.json" % (V14, S),
            "M4": "%s/results_v14_k10_all_n50/result_M4_mnist_seed%d_alpha0.0.json" % (V14, S),
            "BFR": "%s/results_v16b_bfr_k10_all_n50/result_M4_mnist_seed%d_alpha0.0.json" % (V16B, S),
            "M4trim": "results_v16bx_m4trim_k10_all_n50/result_M4_mnist_seed%d_alpha0.0.json" % S,
            "BFRZZ": "results_v16bx_bfrzz_k10_all_n50/result_M4_mnist_seed%d_alpha0.0.json" % S}[M]


R, miss = {}, []
for M in MODELS:
    for S in SEEDS:
        p = path(M, S)
        if os.path.exists(p):
            R[(M, S)] = json.load(open(p))
        else:
            miss.append(p)
if miss:
    sys.exit("MISSING %d files, e.g. %s" % (len(miss), miss[:3]))
labs = {tuple(j["test_labels"]) for j in R.values()}
if len(labs) != 1:
    sys.exit("test labels differ between runs -- stop")
LAB = np.array(labs.pop())
worst = max(abs(100 * np.mean(np.array(j["test_predictions"]) == LAB) - 100 * j["test_acc"]) for j in R.values())
print("Loaded %d runs (M4trim, BFRZZ new; M2, M4 v14; BFR v16b). Max |acc from predictions - test_acc| = %.1e pp" % (len(R), worst))
if worst > 1e-3:
    sys.exit("prediction/label alignment failed -- stop")


def v(M, key="test_acc"):
    return np.array([100 * float(R[(M, S)][key]) for S in SEEDS])


def paired(d):
    d = np.asarray(d, float); n = len(d); m = d.mean(); sd = d.std(ddof=1); se = sd / np.sqrt(n)
    tc = stats.t.ppf(0.975, n - 1)
    p = 2 * stats.t.sf(abs(m / se), n - 1) if se > 0 else (0.0 if m else 1.0)
    return dict(mean=m, sd=sd, lo=m - tc * se, hi=m + tc * se, p=p, mde=(tc + stats.t.ppf(0.8, n - 1)) * se, n=n)


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


rows = [dict(model=M, seed=S, test_acc=100 * R[(M, S)]["test_acc"], train_acc_at_best=100 * R[(M, S)]["train_acc_at_best"],
             gap_at_best=100 * R[(M, S)]["gap_at_best"], best_epoch=R[(M, S)]["best_epoch"], stop_epoch=R[(M, S)]["stop_epoch"],
             params=PARAMS[M]) for M in MODELS for S in SEEDS]
with open("v16bx_runs.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

print("\nK=10, N=50: accuracy (%%), mean +- seed SD, n = %d; params" % len(SEEDS))
for M in MODELS:
    a = v(M)
    print("  %-7s %6.2f +- %.2f   [min %.1f, max %.1f]   params %d   train@best %.1f  gap %.1f" % (
        M, a.mean(), a.std(ddof=1), a.min(), a.max(), PARAMS[M], v(M, "train_acc_at_best").mean(), v(M, "gap_at_best").mean()))

C = [("C1 = BFR - M4trim", "BFR", "M4trim"), ("C2 = M4trim - M4", "M4trim", "M4"), ("C3 = BFRZZ - BFR", "BFRZZ", "BFR")]
res = []
for name, a, b in C:
    r = paired(v(a) - v(b)); r.update(contrast=name, a=a, b=b); res.append(r)
for r, ph in zip(res, holm(np.array([r["p"] for r in res]))):
    r["p_holm"] = ph
d4 = paired(v("BFR") - v("M4"))
print("\nPRIMARY (paired over seeds, Holm over the 3 contrasts)")
for r in res:
    print("  %-18s %+6.2f [%+.2f, %+.2f]  p=%.3g  p_holm=%.3g  MDE=%.2f  %s" % (
        r["contrast"], r["mean"], r["lo"], r["hi"], r["p"], r["p_holm"], r["mde"],
        "detected" if r["p_holm"] < 0.05 else "not detected"))
chk = abs(res[0]["mean"] + res[1]["mean"] - d4["mean"])
print("  check: C1 + C2 = %+.2f = BFR - M4 = %+.2f (v16b: %+.2f); |diff| %.1e" % (res[0]["mean"] + res[1]["mean"], d4["mean"], D4_V16B, chk))
if chk > 1e-9 or abs(d4["mean"] - D4_V16B) > 0.02:
    sys.exit("decomposition check failed -- stop")

print("\nPRE-REGISTERED PREDICTIONS (A3)")


def verdict(r, thr):
    if r["mean"] <= thr and r["p_holm"] < 0.05 and r["mean"] < 0:
        return "HOLDS"
    if r["mean"] <= thr:
        return "direction only (threshold met, not detected) -- not confirmed"
    return "FAILS"


c1, c2, c3 = res
share = c2["mean"] / d4["mean"] if d4["mean"] else float("nan")
print("  E1  M4trim - M4 = %+.2f (needs <= -2.3 and detected; Holm p=%.3g); share of BFR - M4 explained by the trim = %.0f%% -> %s"
      % (c2["mean"], c2["p_holm"], 100 * share, verdict(c2, -2.3)))
print("  E2  BFRZZ - BFR = %+.2f (needs <= -5.0 and detected; Holm p=%.3g) -> %s" % (c3["mean"], c3["p_holm"], verdict(c3, -5.0)))

print("\nSECONDARY (uncorrected): fit / gap decomposition, contrast = fit part + gap part")
DEC = []
for name, a, b in C + [("D4 = BFR - M4", "BFR", "M4"), ("BFRZZ - M4", "BFRZZ", "M4")]:
    fit = paired(v(a, "train_acc_at_best") - v(b, "train_acc_at_best"))
    gap = paired(-(v(a, "gap_at_best") - v(b, "gap_at_best")))
    DEC.append(dict(contrast=name, fit=fit["mean"], fit_lo=fit["lo"], fit_hi=fit["hi"], fit_p=fit["p"],
                    gap=gap["mean"], gap_lo=gap["lo"], gap_hi=gap["hi"], gap_p=gap["p"]))
    print("  %-18s fit %+6.2f [%+.2f, %+.2f] p=%.2g   gap part %+6.2f [%+.2f, %+.2f] p=%.2g" % (
        name, fit["mean"], fit["lo"], fit["hi"], fit["p"], gap["mean"], gap["lo"], gap["hi"], gap["p"]))
print("  seed SD of test accuracy: " + "  ".join("%s %.2f" % (M, v(M).std(ddof=1)) for M in MODELS))
print("  descriptive: BFRZZ - M2 %+.2f, BFRZZ - M4 %+.2f, M4trim - M2 %+.2f, BFR - M2 %+.2f" % (
    (v("BFRZZ") - v("M2")).mean(), (v("BFRZZ") - v("M4")).mean(), (v("M4trim") - v("M2")).mean(), (v("BFR") - v("M2")).mean()))

GS = []
for M, d in (("BFR", "%s/results_v16b_bfr_k10_all_n50/v16b_run_BFR_seed%%d.json" % V16B),
             ("BFRZZ", "results_v16bx_bfrzz_k10_all_n50/v16bx_run_BFRZZ_seed%d.json")):
    for S in SEEDS:
        s = json.load(open(d % S))
        GS.append(dict(model=M, seed=S, g=s["best"]["g"], s=s["best"]["s"], wq_norm=s["best"]["wq_norm"],
                       eff=abs(s["best"]["g"]) * s["best"]["wq_norm"]))
for M in ("BFR", "BFRZZ"):
    g = [x for x in GS if x["model"] == M]
    print("  %-6s best checkpoint: |g| median %.2f, s median %.3f [%.3f, %.3f], |g|.|W_q| median %.2f [%.2f, %.2f]" % (
        M, np.median([abs(x["g"]) for x in g]), np.median([x["s"] for x in g]), min(x["s"] for x in g), max(x["s"] for x in g),
        np.median([x["eff"] for x in g]), min(x["eff"] for x in g), max(x["eff"] for x in g)))
with open("v16bx_contrasts.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["contrast", "a", "b", "mean", "sd", "lo", "hi", "p", "p_holm", "mde", "n"], extrasaction="ignore")
    w.writeheader(); w.writerows(res + [dict(d4, contrast="D4 = BFR - M4 (reference)", a="BFR", b="M4", p_holm="")])
with open("v16bx_decomp.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(DEC[0])); w.writeheader(); w.writerows(DEC)
with open("v16bx_gs.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(GS[0])); w.writeheader(); w.writerows(GS)

print("\nReading (pre-registered, A3):")
if verdict(c2, -2.3) == "HOLDS":
    print("  E1 holds -> at least half of BFR's K10 N50 deficit is the fc_mid trim (narrower classical parameterisation),")
    print("  not the quantum branch. The quantum branch's own effect is C1 = %+.2f [%+.2f, %+.2f]." % (c1["mean"], c1["lo"], c1["hi"]))
else:
    print("  E1 not confirmed -> the trim does not explain half of the deficit; C1 = BFR - M4trim = %+.2f [%+.2f, %+.2f] (%s)"
          % (c1["mean"], c1["lo"], c1["hi"], "detected" if c1["p_holm"] < 0.05 else "not detected, MDE %.2f" % c1["mde"]))
if verdict(c3, -5.0) == "HOLDS":
    print("  E2 holds -> restoring the ZZ product term (on bounded inputs) brings back a large part of the ZZ head's cost:")
    print("  the product term is causally implicated (with the stated confound: it also activates the 6 single-Y Paulis).")
elif c3["p_holm"] < 0.05 and c3["mean"] > 0:
    print("  BFR-ZZ > BFR detected -> the product term (on bounded inputs) helps; the v16b recovery is not from removing it.")
else:
    print("  E2 not confirmed -> no evidence that the product term on bounded inputs carries the ZZ cost; the v16b recovery")
    print("  is then attributed to bounding / gating / the residual form (C3 = %+.2f [%+.2f, %+.2f], MDE %.2f)."
          % (c3["mean"], c3["lo"], c3["hi"], c3["mde"]))
print("  Scope: one cell (K=10, N=50), n = 10, MNIST. A3 was written after v16b's results were read, before any ext job.")
print("\nWrote v16bx_runs.csv, v16bx_contrasts.csv, v16bx_decomp.csv, v16bx_gs.csv")
