#!/usr/bin/env python3
"""
v16a: collect results and run the PRE-REGISTERED analysis (PREREGISTRATION_v16a.md section 5).
Run from the v16a repo root AFTER the grid gate passes (pbs/.grid_ok):

    python3 scripts/collect_v16a.py | tee collect_v16a.log

Arms: M2c (this grid, seeds 42-51) vs the stored M2(w15) and M3(w15) runs: v13 at N=500,
v14 at N=50. K=5 is pooled over subsets A and B (per-seed mean, df = 9); A and B are also
reported. All contrasts are paired over seeds.

PRIMARY  D = M2c - M2 at each (K, N). TOST +-1 pp; Holm within K over the 2 N cells
         (applied to the TOST p-values for P1, and separately to the difference tests).
P1       equivalent (Holm-adjusted TOST p < .05) in all 4 (K, N) cells.
P2       per-image error agreement (Jaccard of the two error sets):
         mean over seeds of J(M2c_s, M2_s)  >=  mean over seed pairs s != t of J(M2_s, M2_t),
         in every (K, N) cell. Cross-seed J(M2c_s, M2_t) and Cohen's kappa reported alongside.
P3       BEc = M3 - M2c reproduces BE = M3 - M2: BEc at N=50 detected > 0 (p < .05) at both K,
         and the collapse is present: BEc(50) - BEc(500) detected > 0 at both K.
Writes v16a_runs.csv, v16a_summary.csv, v16a_contrasts.csv, v16a_agreement.csv, v16a_params.csv.
"""
import csv
import json
import os
import subprocess
import sys
from itertools import combinations

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
if not (os.path.exists("pbs/.smoke_ok") and os.path.exists("pbs/.grid_ok")):
    sys.exit("REFUSED: pbs/.smoke_ok and pbs/.grid_ok must both exist (run the gates first).")


def cfg(*names):
    out = subprocess.run(["bash", "-c", "cd pbs && source ./config16a.sh; " + "; ".join('echo "${%s}"' % n for n in names)],
                         stdout=subprocess.PIPE, universal_newlines=True, check=True).stdout
    return out.split("\n")[:len(names)]


SEEDS, V13, V14, C_NS = cfg("GRID_SEEDS", "V13_ROOT", "V14_ROOT", "C_NS")
SEEDS = [int(s) for s in SEEDS.split()]
KS = (5, 10)
NS = tuple(sorted(int(n) for n in C_NS.split()))     # (50, 500) as pre-registered
NLO, NHI = NS[0], NS[-1]
TAGS = {5: ["A", "B"], 10: ["all"]}
CLASSES = {(5, "A"): [1, 3, 5, 8, 9], (5, "B"): [0, 2, 6, 8, 9], (10, "all"): list(range(10))}
PARAMS = {("M2", 5): 1686, ("M2c", 5): 1665, ("M3", 5): 1725, ("M2", 10): 1766, ("M2c", 10): 1730, ("M3", 10): 1805}
MARGIN = 1.0


def path(M, K, TAG, N, S):
    if M == "M2c":
        return "results_v16a_c_k%d_%s_n%d/result_M2_mnist_seed%d_alpha0.0.json" % (K, TAG, N, S)
    d = ("%s/results_v13_k%d_%s_w15" % (V13, K, TAG)) if N == 500 else ("%s/results_v14_k%d_%s_n%d" % (V14, K, TAG, N))
    return "%s/result_%s_mnist_seed%d_alpha0.0.json" % (d, M, S)


def pct(a):
    a = float(a)
    return 100 * a if a <= 1.0 else a


def preds_of(j):
    for k in ("predictions", "test_predictions", "preds"):
        if k in j:
            v = j[k]
            return [max(range(len(r)), key=r.__getitem__) for r in v] if v and isinstance(v[0], list) else list(map(int, v))
    raise KeyError("no predictions")


# ---- load -------------------------------------------------------------------------------
R, missing = {}, []
for K in KS:
    for TAG in TAGS[K]:
        for N in NS:
            for M in ("M2", "M2c", "M3"):
                for S in SEEDS:
                    p = path(M, K, TAG, N, S)
                    if not os.path.exists(p):
                        missing.append(p); continue
                    R[(M, K, TAG, N, S)] = json.load(open(p))
if missing:
    print("MISSING %d result files, e.g.:\n  " % len(missing) + "\n  ".join(missing[:5]))
    sys.exit(1)
# labels and prediction encoding from the v16a sidecars (one fixed test set per (K, subset))
LAB, ENC = {}, {}
for K in KS:
    for TAG in TAGS[K]:
        sc = [os.path.join("results_v16a_c_k%d_%s_n%d" % (K, TAG, N), f) for N in NS
              for f in os.listdir("results_v16a_c_k%d_%s_n%d" % (K, TAG, N)) if f.startswith("v16_run_")]
        js = [json.load(open(f)) for f in sc]
        labs = {tuple(j["labels"]) for j in js}
        encs = {j["pred_encoding"] for j in js}
        if len(labs) != 1 or len(encs) != 1:
            sys.exit("inconsistent labels/encodings in the sidecars for K=%d %s" % (K, TAG))
        lab = np.array(labs.pop())
        ENC[(K, TAG)] = encs.pop()
        LAB[(K, TAG)] = np.array(CLASSES[(K, TAG)])[lab] if ENC[(K, TAG)] == "class" else lab


def acc(M, K, TAG, N, S):
    return pct(R[(M, K, TAG, N, S)]["test_acc"])


def err(M, K, TAG, N, S):
    return np.array(preds_of(R[(M, K, TAG, N, S)])) != LAB[(K, TAG)]


# sanity: recomputed accuracy from predictions == stored test_acc
worst = max(abs(100 * (1 - err(*k).mean()) - acc(*k)) for k in R)
print("Loaded %d runs. Max |accuracy from predictions - test_acc| = %.2e pp" % (len(R), worst))
if worst > 1e-3:
    sys.exit("prediction/label alignment failed -- stop")


def seedvec(M, K, N, TAG=None):
    if TAG is not None:
        return np.array([acc(M, K, TAG, N, S) for S in SEEDS])
    return np.mean([[acc(M, K, T, N, S) for S in SEEDS] for T in TAGS[K]], axis=0)


def paired(d):
    d = np.asarray(d, float)
    n = len(d); m = d.mean(); sd = d.std(ddof=1); se = sd / np.sqrt(n); df = n - 1
    tc = stats.t.ppf(0.975, df)
    p = 2 * stats.t.sf(abs(m / se), df) if se > 0 else (0.0 if m != 0 else 1.0)
    p_tost = max(stats.t.sf((m + MARGIN) / se, df), stats.t.cdf((m - MARGIN) / se, df)) if se > 0 else float(abs(m) >= MARGIN)
    mde = (stats.t.ppf(0.975, df) + stats.t.ppf(0.80, df)) * se
    return dict(mean=m, sd=sd, lo=m - tc * se, hi=m + tc * se, p=p, p_tost=p_tost, mde=mde, n=n)


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1.0, run)
    return adj


# ---- tables -----------------------------------------------------------------------------
rows = []
for (M, K, TAG, N, S), j in sorted(R.items()):
    rows.append(dict(model=M, K=K, subset=TAG, N=N, seed=S, test_acc=acc(M, K, TAG, N, S), best_epoch=j.get("best_epoch"),
                     stop_epoch=j.get("stop_epoch"), params=PARAMS[(M, K)]))
with open("v16a_runs.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

print("\nParameter counts (verified in every job log):")
print("  K=5 : M2(w15) %d   M2c %d   M3 %d" % (PARAMS[("M2", 5)], PARAMS[("M2c", 5)], PARAMS[("M3", 5)]))
print("  K=10: M2(w15) %d   M2c %d   M3 %d" % (PARAMS[("M2", 10)], PARAMS[("M2c", 10)], PARAMS[("M3", 10)]))
print("  M2c = M2 - 6 circuit parameters - Linear(15->K) + Linear(12->K)")
with open("v16a_params.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["model", "K", "params"])
    for (M, K), v in sorted(PARAMS.items()):
        w.writerow([M, K, v])

summ = []
print("\nAccuracy (%%), mean +- seed SD, n = %d seeds per cell" % len(SEEDS))
print("  %-10s %-4s %-16s %-16s %-16s" % ("K", "N", "M2 (w15)", "M2c", "M3"))
for K in KS:
    for lab, TAG in ([("5 pooled", None), ("5 A", "A"), ("5 B", "B")] if K == 5 else [("10", None)]):
        for N in NS:
            cells = []
            for M in ("M2", "M2c", "M3"):
                v = seedvec(M, K, N, TAG)
                cells.append("%.2f +- %.2f" % (v.mean(), v.std(ddof=1)))
                summ.append(dict(K=K, subset=TAG or ("pooled" if K == 5 else "all"), N=N, model=M, mean=v.mean(),
                                 sd=v.std(ddof=1), min=v.min(), max=v.max()))
            print("  %-10s %-4d %-16s %-16s %-16s" % (lab, N, *cells))
with open("v16a_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summ[0])); w.writeheader(); w.writerows(summ)

# ---- contrasts --------------------------------------------------------------------------
CON = []


def add(name, K, N, TAG, d, family=None):
    r = paired(d); r.update(contrast=name, K=K, N=N, subset=TAG or ("pooled" if K == 5 else "all"), family=family or "")
    CON.append(r)
    return r


print("\nPRIMARY: D = M2c - M2(w15), paired over seeds; TOST +-%.0f pp; Holm within K (2 cells)" % MARGIN)
P1 = {}
for K in KS:
    rs = [add("D", K, N, None, seedvec("M2c", K, N) - seedvec("M2", K, N), "primary") for N in NS]
    adj_t = holm(np.array([r["p_tost"] for r in rs]))
    adj_p = holm(np.array([r["p"] for r in rs]))
    for r, at, ap in zip(rs, adj_t, adj_p):
        r["p_tost_holm"], r["p_holm"] = at, ap
        P1[(K, r["N"])] = at < 0.05
        print("  K=%-2d N=%-3d D=%+.2f [%+.2f, %+.2f]  p=%.4f p_holm=%.4f  MDE=%.2f  TOST p=%.4f p_holm=%.4f  %s" % (
            K, r["N"], r["mean"], r["lo"], r["hi"], r["p"], ap, r["mde"], r["p_tost"], at,
            "EQUIVALENT" if at < 0.05 else ("DIFFERENT" if ap < 0.05 else "not shown either way")))
    if K == 5:
        for TAG in TAGS[5]:
            for N in NS:
                r = add("D", K, N, TAG, seedvec("M2c", K, N, TAG) - seedvec("M2", K, N, TAG), "per-subset")
                print("      subset %s N=%-3d D=%+.2f [%+.2f, %+.2f] TOST p=%.4f" % (TAG, N, r["mean"], r["lo"], r["hi"], r["p_tost"]))

print("\nSECONDARY (uncorrected): back-end cost with each head")
BE, BEC = {}, {}
for K in KS:
    for N in NS:
        BE[(K, N)] = add("BE=M3-M2", K, N, None, seedvec("M3", K, N) - seedvec("M2", K, N), "secondary")
        BEC[(K, N)] = add("BEc=M3-M2c", K, N, None, seedvec("M3", K, N) - seedvec("M2c", K, N), "secondary")
        print("  K=%-2d N=%-3d BE=%+.2f [%+.2f, %+.2f] p=%.2g   BEc=%+.2f [%+.2f, %+.2f] p=%.2g   BEc/BE=%.2f" % (
            K, N, BE[(K, N)]["mean"], BE[(K, N)]["lo"], BE[(K, N)]["hi"], BE[(K, N)]["p"],
            BEC[(K, N)]["mean"], BEC[(K, N)]["lo"], BEC[(K, N)]["hi"], BEC[(K, N)]["p"],
            BEC[(K, N)]["mean"] / BE[(K, N)]["mean"] if BE[(K, N)]["mean"] else float("nan")))
COL = {}
for K in KS:
    dc = (seedvec("M3", K, NLO) - seedvec("M2c", K, NLO)) - (seedvec("M3", K, NHI) - seedvec("M2c", K, NHI))
    d0 = (seedvec("M3", K, NLO) - seedvec("M2", K, NLO)) - (seedvec("M3", K, NHI) - seedvec("M2", K, NHI))
    COL[K] = add("BEc(50)-BEc(500)", K, 0, None, dc, "secondary")
    r0 = add("BE(50)-BE(500)", K, 0, None, d0, "secondary")
    print("  K=%-2d collapse BEc(50)-BEc(500) = %+.2f [%+.2f, %+.2f] p=%.2g   (M2: BE(50)-BE(500) = %+.2f)" % (
        K, COL[K]["mean"], COL[K]["lo"], COL[K]["hi"], COL[K]["p"], r0["mean"]))
with open("v16a_contrasts.csv", "w", newline="") as f:
    keys = ["contrast", "family", "K", "N", "subset", "mean", "sd", "lo", "hi", "p", "p_holm", "p_tost", "p_tost_holm", "mde", "n"]
    w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(CON)

# ---- P2: per-image error agreement -------------------------------------------------------


def jac(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else 1.0


def kappa(a, b):
    a, b = a.astype(int), b.astype(int)
    po = (a == b).mean(); pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


print("\nP2: per-image error agreement (Jaccard of error sets; kappa on correct/incorrect)")
AG, P2 = [], {}
for K in KS:
    for N in NS:
        same, cross_m2, cross_c, k_same, k_m2 = [], [], [], [], []
        for TAG in TAGS[K]:
            E2 = {S: err("M2", K, TAG, N, S) for S in SEEDS}
            Ec = {S: err("M2c", K, TAG, N, S) for S in SEEDS}
            same.append([jac(Ec[S], E2[S]) for S in SEEDS])
            k_same.append([kappa(Ec[S], E2[S]) for S in SEEDS])
            cross_m2.append(np.mean([jac(E2[s], E2[t]) for s, t in combinations(SEEDS, 2)]))
            k_m2.append(np.mean([kappa(E2[s], E2[t]) for s, t in combinations(SEEDS, 2)]))
            cross_c.append(np.mean([jac(Ec[s], E2[t]) for s in SEEDS for t in SEEDS if s != t]))
        js, jm, jc = np.mean(same), np.mean(cross_m2), np.mean(cross_c)
        P2[(K, N)] = js >= jm
        AG.append(dict(K=K, N=N, J_M2c_M2_same_seed=js, J_M2_M2_cross_seed=jm, J_M2c_M2_cross_seed=jc,
                       kappa_same_seed=np.mean(k_same), kappa_M2_cross_seed=np.mean(k_m2), P2=P2[(K, N)]))
        print("  K=%-2d N=%-3d J(M2c_s, M2_s)=%.3f  J(M2_s, M2_t)=%.3f  J(M2c_s, M2_t)=%.3f  kappa same=%.3f M2-M2=%.3f  -> %s" % (
            K, N, js, jm, jc, np.mean(k_same), np.mean(k_m2), "P2 holds" if P2[(K, N)] else "P2 fails"))
with open("v16a_agreement.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(AG[0])); w.writeheader(); w.writerows(AG)

# ---- verdicts ---------------------------------------------------------------------------
print("\nPRE-REGISTERED PREDICTIONS")
p1 = all(P1.values())
print("  P1 (D equivalent within +-1 pp, Holm-adjusted TOST, all 4 cells): %s   [%s]" % (
    "HOLDS" if p1 else "FAILS", ", ".join("K%d N%d %s" % (K, N, "eq" if v else "not eq") for (K, N), v in sorted(P1.items()))))
print("  P2 (same-seed M2c-M2 error agreement >= M2-M2 cross-seed, all 4 cells): %s" % ("HOLDS" if all(P2.values()) else "FAILS"))
p3a = all(BEC[(K, NLO)]["mean"] > 0 and BEC[(K, NLO)]["p"] < 0.05 for K in KS)
p3b = all(COL[K]["mean"] > 0 and COL[K]["p"] < 0.05 for K in KS)
print("  P3 (BEc at N=50 detected > 0 at both K: %s; collapse BEc(50) > BEc(500) detected at both K: %s): %s" % (
    p3a, p3b, "HOLDS" if (p3a and p3b) else "FAILS"))
print("\nReading (PREREGISTRATION_v16a.md section 7):")
if p1:
    print("  P1 holds -> the head's cost belongs to the classical feature map; nothing quantum contributes.")
else:
    ds = [c for c in CON if c["contrast"] == "D" and c["family"] == "primary"]
    if any(c["mean"] < 0 and c.get("p_holm", 1) < 0.05 for c in ds):
        print("  D detected < 0 somewhere -> CHECK THE IMPLEMENTATION FIRST (pre-registered), then read.")
    elif any(c["mean"] > 0 and c.get("p_holm", 1) < 0.05 for c in ds):
        print("  D detected > 0 somewhere -> the circuit parameterisation adds optimisation difficulty on top of the feature map.")
    else:
        print("  P1 not shown and nothing detected -> underpowered at n = 10 in the failing cell(s); report the CI and MDE.")
print("\nWrote v16a_runs.csv, v16a_summary.csv, v16a_contrasts.csv, v16a_agreement.csv, v16a_params.csv")
