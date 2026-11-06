#!/usr/bin/env python3
"""
v16b: collect results and run the PRE-REGISTERED analysis (PREREGISTRATION_v16b.md section 6).
Run from the v16b repo root AFTER the grid gate passes (pbs/.grid_ok):

    python3 scripts/collect_v16b.py | tee collect_v16b.log

Arms: BFR (this grid, seeds 42-51) vs the stored M4 runs (v14, N = 50 and 500) and the stored M2-w15
runs (v14 at N = 50, v13 at N = 500). K = 5 is pooled over subsets A and B (per-seed mean, df = 9);
A and B are reported alongside. All contrasts are paired over seeds.

PRIMARY  D4 = BFR - M4 (TOST +-1 pp) and D2 = BFR - M2 (difference test), at each (K, N).
         Holm within K over the 2 N cells, per contrast, applied separately to the TOST p-values and to
         the two-sided difference p-values (v16a precedent).
P1       D2 >= +1.2 pp at K10 N500, and >= +6 pp at N50 (both K): point estimate at/above the threshold
         AND D2 detected > 0 (Holm-adjusted two-sided p < .05).
P2       N500 (both K): D4 equivalent within +-1 pp (Holm-adjusted TOST p < .05).
         N50 (both K): non-inferiority, H0: D4 <= -1 pp rejected (one-sided p < .05).
P5       seed SD(BFR) <= 1.5 x seed SD(M4) in every (K, N) cell (K = 5: SD of the pooled per-seed means).
Secondary (uncorrected / exploratory): per-subset contrasts; g and s at the best checkpoint and at the stop
epoch; best/stop epochs vs M4; per-image error agreement BFR-M4 (same seed) vs M4-M4 (cross-seed).
Every null comes with its MDE (80% power, alpha .05, paired t, n = 10).
Writes v16b_runs.csv, v16b_summary.csv, v16b_contrasts.csv, v16b_gs.csv, v16b_agreement.csv, v16b_params.csv.
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
    out = subprocess.run(["bash", "-c", "cd pbs && source ./config16b.sh; " + "; ".join('echo "${%s}"' % n for n in names)],
                         stdout=subprocess.PIPE, universal_newlines=True, check=True).stdout
    return out.split("\n")[:len(names)]


SEEDS, V13, V14, B_NS = cfg("GRID_SEEDS", "V13_ROOT", "V14_ROOT", "B_NS")
SEEDS = [int(s) for s in SEEDS.split()]
KS = (5, 10)
NS = tuple(sorted(int(n) for n in B_NS.split()))
TAGS = {5: ["A", "B"], 10: ["all"]}
MODELS = ("M2", "M4", "BFR")
PARAMS = {("M2", 5): 1686, ("M4", 5): 1719, ("BFR", 5): 1720, ("M2", 10): 1766, ("M4", 10): 1799, ("BFR", 10): 1799}
WIDTH = {5: 5, 10: 3}
MARGIN = 1.0
P1_THR = {(10, 500): 1.2, (5, 50): 6.0, (10, 50): 6.0}


def path(M, K, TAG, N, S):
    if M == "BFR":
        return "results_v16b_bfr_k%d_%s_n%d/result_M4_mnist_seed%d_alpha0.0.json" % (K, TAG, N, S)
    if M == "M4":
        d = "%s/results_v14_k%d_%s_n%d" % (V14, K, TAG, N)
    else:
        d = ("%s/results_v13_k%d_%s_w15" % (V13, K, TAG)) if N == 500 else ("%s/results_v14_k%d_%s_n%d" % (V14, K, TAG, N))
    return "%s/result_%s_mnist_seed%d_alpha0.0.json" % (d, M, S)


def pct(a):
    a = float(a)
    return 100 * a if a <= 1.0 else a


# ---- load -------------------------------------------------------------------------------
R, SC, missing = {}, {}, []
for K in KS:
    for TAG in TAGS[K]:
        for N in NS:
            for M in MODELS:
                for S in SEEDS:
                    p = path(M, K, TAG, N, S)
                    if not os.path.exists(p):
                        missing.append(p); continue
                    R[(M, K, TAG, N, S)] = json.load(open(p))
                    if M == "BFR":
                        SC[(K, TAG, N, S)] = json.load(open(os.path.join(os.path.dirname(p), "v16b_run_BFR_seed%d.json" % S)))
if missing:
    print("MISSING %d result files, e.g.:\n  " % len(missing) + "\n  ".join(missing[:5]))
    sys.exit(1)
# one fixed test set per (K, subset): every model's stored labels must agree index-for-index
LAB = {}
for K in KS:
    for TAG in TAGS[K]:
        labs = {tuple(R[(M, K, TAG, N, S)]["test_labels"]) for M in MODELS for N in NS for S in SEEDS}
        if len(labs) != 1:
            sys.exit("test labels differ between runs at K=%d %s -- stop (the comparison would not be paired per image)" % (K, TAG))
        LAB[(K, TAG)] = np.array(labs.pop())


def acc(M, K, TAG, N, S):
    return pct(R[(M, K, TAG, N, S)]["test_acc"])


def err(M, K, TAG, N, S):
    return np.array(R[(M, K, TAG, N, S)]["test_predictions"]) != LAB[(K, TAG)]


worst = max(abs(100 * (1 - err(*k).mean()) - acc(*k)) for k in R)
print("Loaded %d runs (BFR %d new; M4 and M2-w15 stored). Max |accuracy from predictions - test_acc| = %.2e pp" % (
    len(R), sum(1 for k in R if k[0] == "BFR"), worst))
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
    if se > 0:
        p = 2 * stats.t.sf(abs(m / se), df)
        p_lo = stats.t.sf((m + MARGIN) / se, df)         # H0: D <= -margin  (non-inferiority)
        p_hi = stats.t.cdf((m - MARGIN) / se, df)        # H0: D >= +margin
    else:
        p = 0.0 if m != 0 else 1.0
        p_lo, p_hi = float(m <= -MARGIN), float(m >= MARGIN)
    mde = (stats.t.ppf(0.975, df) + stats.t.ppf(0.80, df)) * se
    return dict(mean=m, sd=sd, lo=m - tc * se, hi=m + tc * se, p=p, p_tost=max(p_lo, p_hi), p_noninf=p_lo, mde=mde, n=n)


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
with open("v16b_runs.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

print("\nParameter counts (verified in every job log):")
print("  K    M2-w15  M4     BFR    BFR-M4         BFR fc_mid width")
for K in KS:
    d = PARAMS[("BFR", K)] - PARAMS[("M4", K)]
    print("  %-4d %-7d %-6d %-6d %+d (%+.2f%%)   %d (M4: 15)" % (K, PARAMS[("M2", K)], PARAMS[("M4", K)], PARAMS[("BFR", K)],
                                                              d, 100 * d / PARAMS[("M4", K)], WIDTH[K]))
with open("v16b_params.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["model", "K", "params"])
    for (M, K), v in sorted(PARAMS.items()):
        w.writerow([M, K, v])

summ = []
print("\nAccuracy (%%), mean +- seed SD, n = %d seeds per cell" % len(SEEDS))
print("  %-10s %-4s %-16s %-16s %-16s" % ("K", "N", "M2 (w15)", "M4", "BFR"))
for K in KS:
    for lab, TAG in ([("5 pooled", None), ("5 A", "A"), ("5 B", "B")] if K == 5 else [("10", None)]):
        for N in NS:
            cells = []
            for M in MODELS:
                v = seedvec(M, K, N, TAG)
                cells.append("%.2f +- %.2f" % (v.mean(), v.std(ddof=1)))
                summ.append(dict(K=K, subset=TAG or ("pooled" if K == 5 else "all"), N=N, model=M, mean=v.mean(),
                                 sd=v.std(ddof=1), min=v.min(), max=v.max()))
            print("  %-10s %-4d %-16s %-16s %-16s" % (lab, N, *cells))
with open("v16b_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summ[0])); w.writeheader(); w.writerows(summ)

# ---- primary ------------------------------------------------------------------------------
CON = []


def add(name, K, N, TAG, d, family=None):
    r = paired(d); r.update(contrast=name, K=K, N=N, subset=TAG or ("pooled" if K == 5 else "all"), family=family or "")
    CON.append(r)
    return r


PRIM = {}
print("\nPRIMARY (paired over seeds; Holm within K over the 2 N cells, per contrast)")
for K in KS:
    for name, other in (("D4=BFR-M4", "M4"), ("D2=BFR-M2", "M2")):
        rs = [add(name, K, N, None, seedvec("BFR", K, N) - seedvec(other, K, N), "primary") for N in NS]
        adj_p = holm(np.array([r["p"] for r in rs]))
        adj_t = holm(np.array([r["p_tost"] for r in rs]))
        for r, ap, at in zip(rs, adj_p, adj_t):
            r["p_holm"], r["p_tost_holm"] = ap, at
            PRIM[(name, K, r["N"])] = r
            extra = ("  TOST p=%.4f p_holm=%.4f  non-inf(D>-1) p=%.4f" % (r["p_tost"], at, r["p_noninf"])) if other == "M4" else ""
            print("  K=%-2d N=%-3d %-10s %+6.2f [%+.2f, %+.2f]  p=%.2g p_holm=%.2g  MDE=%.2f%s" % (
                K, r["N"], name, r["mean"], r["lo"], r["hi"], r["p"], ap, r["mde"], extra))
    if K == 5:
        for TAG in TAGS[5]:
            for N in NS:
                for name, other in (("D4=BFR-M4", "M4"), ("D2=BFR-M2", "M2")):
                    r = add(name, K, N, TAG, seedvec("BFR", K, N, TAG) - seedvec(other, K, N, TAG), "per-subset")
                    print("      subset %s N=%-3d %-10s %+6.2f [%+.2f, %+.2f]" % (TAG, N, name, r["mean"], r["lo"], r["hi"]))

# ---- predictions ----------------------------------------------------------------------------
print("\nPRE-REGISTERED PREDICTIONS")
P1 = {}
for (K, N), thr in sorted(P1_THR.items()):
    r = PRIM[("D2=BFR-M2", K, N)]
    P1[(K, N)] = r["mean"] >= thr and r["mean"] > 0 and r["p_holm"] < 0.05
    print("  P1  K=%-2d N=%-3d BFR-M2 = %+.2f (needs >= %+.1f and Holm p < .05: p_holm=%.2g) -> %s" % (
        K, N, r["mean"], thr, r["p_holm"], "holds" if P1[(K, N)] else "fails"))
P2 = {}
for K in KS:
    for N in NS:
        r = PRIM[("D4=BFR-M4", K, N)]
        if N == max(NS):
            P2[(K, N)] = r["p_tost_holm"] < 0.05
            print("  P2  K=%-2d N=%-3d BFR-M4 = %+.2f [%+.2f, %+.2f], equivalence +-1 pp: Holm TOST p=%.4f -> %s" % (
                K, N, r["mean"], r["lo"], r["hi"], r["p_tost_holm"], "holds" if P2[(K, N)] else "fails"))
        else:
            P2[(K, N)] = r["p_noninf"] < 0.05
            print("  P2  K=%-2d N=%-3d BFR-M4 = %+.2f [%+.2f, %+.2f], non-inferiority (D > -1 pp): p=%.4f -> %s" % (
                K, N, r["mean"], r["lo"], r["hi"], r["p_noninf"], "holds" if P2[(K, N)] else "fails"))
P5 = {}
for K in KS:
    for N in NS:
        sb, s4 = seedvec("BFR", K, N).std(ddof=1), seedvec("M4", K, N).std(ddof=1)
        P5[(K, N)] = sb <= 1.5 * s4
        print("  P5  K=%-2d N=%-3d seed SD BFR %.2f vs M4 %.2f (ratio %.2f, needs <= 1.5) -> %s" % (
            K, N, sb, s4, sb / s4 if s4 else float("inf"), "holds" if P5[(K, N)] else "fails"))
print("  SUMMARY: P1 %s | P2 %s | P5 %s" % ("HOLDS" if all(P1.values()) else "FAILS (%d/%d)" % (sum(P1.values()), len(P1)),
                                           "HOLDS" if all(P2.values()) else "FAILS (%d/%d)" % (sum(P2.values()), len(P2)),
                                           "HOLDS" if all(P5.values()) else "FAILS (%d/%d)" % (sum(P5.values()), len(P5))))

# ---- secondary: g, s, epochs -----------------------------------------------------------------
print("\nSECONDARY: learned gate g and scale s (best checkpoint = the reported model; stop = last epoch)")
GS = []
for K in KS:
    for N in NS:
        for TAG in TAGS[K]:
            for S in SEEDS:
                sc = SC[(K, TAG, N, S)]
                GS.append(dict(K=K, subset=TAG, N=N, seed=S, g_best=sc["best"]["g"], s_best=sc["best"]["s"],
                               g_stop=sc["stop"]["g"], s_stop=sc["stop"]["s"], wq_norm_best=sc["best"]["wq_norm"],
                               theta_best=" ".join("%.4f" % t for t in sc["best"]["theta"]),
                               best_epoch=sc["best_epoch"], stop_epoch=sc["stop_epoch"],
                               d4=acc("BFR", K, TAG, N, S) - acc("M4", K, TAG, N, S)))
        g = np.array([r["g_best"] for r in GS if r["K"] == K and r["N"] == N])
        s = np.array([r["s_best"] for r in GS if r["K"] == K and r["N"] == N])
        gq = np.array([abs(r["g_best"]) * r["wq_norm_best"] for r in GS if r["K"] == K and r["N"] == N])
        d4 = np.array([r["d4"] for r in GS if r["K"] == K and r["N"] == N])
        rho = stats.spearmanr(gq, d4).correlation if len(set(gq)) > 1 else float("nan")
        eb = [R[("BFR", K, T, N, S)]["best_epoch"] for T in TAGS[K] for S in SEEDS]
        e4 = [R[("M4", K, T, N, S)]["best_epoch"] for T in TAGS[K] for S in SEEDS]
        print("  K=%-2d N=%-3d g: median %+.3f [min %+.3f, max %+.3f]  s: median %.3f [%.3f, %.3f]  "
              "|g|.|W_q| vs per-run BFR-M4: Spearman %.2f   median best epoch BFR %d vs M4 %d" % (
                  K, N, np.median(g), g.min(), g.max(), np.median(s), s.min(), s.max(), rho, np.median(eb), np.median(e4)))
with open("v16b_gs.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(GS[0])); w.writeheader(); w.writerows(GS)

# ---- exploratory: per-image error agreement --------------------------------------------------


def jac(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else 1.0


print("\nEXPLORATORY: per-image error agreement (Jaccard of error sets)")
AG = []
for K in KS:
    for N in NS:
        same, cross4, same2 = [], [], []
        for TAG in TAGS[K]:
            E4 = {S: err("M4", K, TAG, N, S) for S in SEEDS}
            EB = {S: err("BFR", K, TAG, N, S) for S in SEEDS}
            E2 = {S: err("M2", K, TAG, N, S) for S in SEEDS}
            same.append(np.mean([jac(EB[S], E4[S]) for S in SEEDS]))
            same2.append(np.mean([jac(EB[S], E2[S]) for S in SEEDS]))
            cross4.append(np.mean([jac(E4[s], E4[t]) for s, t in combinations(SEEDS, 2)]))
        AG.append(dict(K=K, N=N, J_BFR_M4_same_seed=np.mean(same), J_M4_M4_cross_seed=np.mean(cross4),
                       J_BFR_M2_same_seed=np.mean(same2)))
        print("  K=%-2d N=%-3d J(BFR_s, M4_s)=%.3f  J(M4_s, M4_t)=%.3f  J(BFR_s, M2_s)=%.3f" % (
            K, N, np.mean(same), np.mean(cross4), np.mean(same2)))
with open("v16b_agreement.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(AG[0])); w.writeheader(); w.writerows(AG)
with open("v16b_contrasts.csv", "w", newline="") as f:
    keys = ["contrast", "family", "K", "N", "subset", "mean", "sd", "lo", "hi", "p", "p_holm", "p_tost", "p_tost_holm",
            "p_noninf", "mde", "n"]
    w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(CON)

# ---- reading (V16_PLAN.md / PREREGISTRATION_v16b.md section 7) --------------------------------
print("\nReading (pre-registered, PREREGISTRATION_v16b.md section 7):")
d2_big = all(P1.values())
d4 = [PRIM[("D4=BFR-M4", K, N)] for K in KS for N in NS]
d4_win = [r for r in d4 if r["mean"] >= 1.0 and r["p_holm"] < 0.05]
d4_loss = [r for r in d4 if r["mean"] < 0 and r["p_holm"] < 0.05]
d2_null = [PRIM[("D2=BFR-M2", K, N)] for K in KS for N in NS if PRIM[("D2=BFR-M2", K, N)]["p_holm"] >= 0.05]
if d2_big and all(P2.values()):
    print("  BFR >> M2 and ~ M4 -> bounding the encoding removes the head's cost. Consistent with the diagnosis;")
    print("  causal attribution to the ZZ product term needs BFR-ZZ (not run in the main v16b grid, A1).")
if d2_null and len(d2_null) == len(NS) * len(KS):
    print("  BFR ~ M2 everywhere -> bounding the phase is not enough; the cost is periodicity or optimisation.")
if d4_win:
    print("  BFR > M4 by >= 1 pp (detected) in %s -> the first fair hybrid win in the project. CONFIRM ON FRESH SEEDS "
          "before claiming it." % ", ".join("K%d N%d" % (r["K"], r["N"]) for r in d4_win))
if d4_loss:
    print("  BFR < M4 detected in %s. Check the implementation and the g/s logs first, then read: the gated "
          "quantum branch costs accuracy against the classical head it starts from." % ", ".join("K%d N%d" % (r["K"], r["N"]) for r in d4_loss))
if not d2_big and not d2_null:
    print("  Mixed: P1 holds in %d/%d cells; report each cell's estimate, CI and MDE." % (sum(P1.values()), len(P1)))
for r in d4:
    if r["p_holm"] >= 0.05 and not P2[(r["K"], r["N"])]:
        print("  K%d N%d: BFR vs M4 neither different nor shown equivalent -> underpowered at n = 10; report CI [%+.2f, %+.2f] "
              "and MDE %.2f, no equivalence claim." % (r["K"], r["N"], r["lo"], r["hi"], r["mde"]))
for K in KS:
    for N in NS:
        eff = [abs(x["g_best"]) * x["wq_norm_best"] for x in GS if x["K"] == K and x["N"] == N]
        off = sum(e < 0.05 for e in eff)
        if off > len(eff) / 2:
            print("  K%d N%d: effective branch weight |g|.|W_q| < 0.05 in %d/%d runs -> BFR collapsed onto its classical path; "
                  "BFR ~ M4 there means the quantum branch was switched off." % (K, N, off, len(eff)))
print("  Caveat (pre-registered): fc_mid is trimmed (15 -> %d at K=5, -> %d at K=10), so BFR starts as a slightly "
      "narrower-parameterised M4 (same function class), not M4 exactly." % (WIDTH[5], WIDTH[10]))
print("\nWrote v16b_runs.csv, v16b_summary.csv, v16b_contrasts.csv, v16b_gs.csv, v16b_agreement.csv, v16b_params.csv")
