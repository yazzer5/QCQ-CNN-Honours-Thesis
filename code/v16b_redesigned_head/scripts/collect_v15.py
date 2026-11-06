#!/usr/bin/env python3
"""
v15 (front-end noise): collect results and run the PRE-REGISTERED contrasts.
Run from the v15 repo root AFTER the grid gate passes (pbs/.grid_ok):

    python3 scripts/collect_v15.py | tee collect_v15.log

Two arms from one grid (PREREGISTRATION_v15.md section 5):
  MATCHED  -- trained AND tested at alpha (noise-aware training; the hardware run).
              = test_acc of the job at alpha.
  DEPLOYED -- trained noiseless (alpha=0), tested at alpha_eval (cross-evaluation
              sidecar of the alpha=0 job). Includes alpha_eval = 8, 16 (stress).
M4 has no quantum front-end: its accuracy is the same at every alpha (alpha=0 job).

Contrasts, paired over seeds (K=5 pooled = per-seed mean of subsets A and B, df=9):
    F  = M3 - M4     trained quantum front-end vs none          (+TOST +/-1 pp)
    T  = M3 - M5     training vs freezing the quantum front-end (+TOST)
    R  = M5 - M4     frozen random quantum front-end vs none    (+TOST)
    BE = M3 - M2     classical vs quantum head
    Q  = M2 - M1     Circuit-4 trained vs RandomLayers frozen, quantum head
    NC_m(a) = m(a) - m(0)              noise cost to model m, matched arm (+TOST)
    DEP_m(a) = m_deployed(a) - m(0)    noise cost when deployed without noise-aware training
    REC_m(a) = m(a) - m_deployed(a)    what noise-aware training recovers
    slope(X) = per-seed OLS slope of X on alpha over the training alphas
               (pp per unit alpha), one-sample t over seeds.

PRIMARY (Holm within each K, 4 tests): NC3@1, F@1, T@1, slope(F).
SECONDARY (uncorrected, all with MDE): every contrast at every alpha, both arms;
    slope(T), slope(R), slope(BE), slope(NC_m); per-subset results.
Writes v15_runs.csv, v15_summary.csv, v15_contrasts.csv.
"""
import csv
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
MARGIN = 1.0


def cfg():
    names = ["GRID_KS", "GRID_ALPHAS", "EVAL_ALPHAS", "GRID_SEEDS", "NOISE_MODELS", "CTRL_MODELS", "CALIB_SHA"]
    cmd = "cd '%s/pbs' && source ./config15.sh && " % ROOT + " && ".join('echo "${%s}"' % n for n in names)
    out = subprocess.run(["bash", "-c", cmd], stdout=subprocess.PIPE, universal_newlines=True, check=True).stdout.split("\n")
    return ([int(x) for x in out[0].split()], [float(x) for x in out[1].split()], [float(x) for x in out[2].split()],
            [int(x) for x in out[3].split()], out[4].split(), out[5].split(), out[6].strip())


KS, ALPHAS, EVALS, SEEDS, NOISE_MODELS, CTRL_MODELS, SHA = cfg()
TAGS = {5: ["A", "B"], 10: ["all"]}
MODELS = ["M1", "M2", "M3", "M4", "M5"]


def load():
    if not (ROOT / "pbs" / ".grid_ok").exists():
        sys.exit("pbs/.grid_ok missing: the grid gate has not passed. Do not read results.")
    matched, deployed, rows = {}, {}, []
    for d in sorted(ROOT.glob("results_v15_k*_*_a*")):
        m = re.match(r"results_v15_k(\d+)_(\w+?)_a([\d.]+)$", d.name)
        if not m:
            continue
        K, TAG, A = int(m.group(1)), m.group(2), float(m.group(3))
        for f in sorted(d.glob("result_M*_*.json")):
            j = json.loads(f.read_text())
            M, S = j.get("model") or re.match(r"result_(M\d)_", f.name).group(1), int(j["seed"])
            if j.get("loss") not in ("ce", None):
                sys.exit("%s: loss=%s. Stop." % (f, j.get("loss")))
            sc = d / ("v15_xeval_%s_seed%d.json" % (M, S))
            if not sc.exists():
                sys.exit("%s missing. Stop: the gate should have caught this." % sc)
            s = json.loads(sc.read_text())
            if s["selfcheck"] != "MATCH" or s["calib_sha"] != SHA or float(s["alpha_train"]) != A:
                sys.exit("%s: cross-eval sidecar invalid (%s). Stop." % (sc, s["selfcheck"]))
            acc = 100 * float(j["test_acc"])
            matched[(K, TAG, A, M, S)] = acc
            for ae, v in s["acc"].items():
                deployed[(K, TAG, A, M, S, float(ae))] = 100 * float(v)
            rows.append(dict(K=K, subset=TAG, alpha_train=A, model=M, seed=S, test_acc=acc,
                             best_epoch=j.get("best_epoch"), stop_epoch=j.get("stop_epoch"),
                             **{"acc_eval_%g" % float(ae): 100 * float(v) for ae, v in s["acc"].items()}))
    return matched, deployed, rows


def paired(d, margin=None):
    d = np.asarray(d, float)
    n = len(d)
    out = dict(n=n, diff=d.mean() if n else np.nan, sd=np.nan, lo=np.nan, hi=np.nan, p=np.nan, mde=np.nan,
               lo90=np.nan, hi90=np.nan, p_tost=np.nan, equivalent="")
    if n < 2:
        return out
    sd = d.std(ddof=1); se = sd / np.sqrt(n); df = n - 1
    t975, t95, t80 = stats.t.ppf(.975, df), stats.t.ppf(.95, df), stats.t.ppf(.80, df)
    out.update(sd=sd, lo=d.mean() - t975 * se, hi=d.mean() + t975 * se, mde=(t975 + t80) * se,
               p=stats.ttest_1samp(d, 0).pvalue if sd > 0 else (1.0 if d.mean() == 0 else 0.0))
    if margin is not None and sd > 0:
        lo90, hi90 = d.mean() - t95 * se, d.mean() + t95 * se
        out.update(lo90=lo90, hi90=hi90,
                   p_tost=max(1 - stats.t.cdf((d.mean() + margin) / se, df), stats.t.cdf((d.mean() - margin) / se, df)),
                   equivalent="yes" if (lo90 > -margin and hi90 < margin) else "no")
    return out


def holm(ps):
    ps = np.array([p if np.isfinite(p) else 1.0 for p in ps]); order = np.argsort(ps)
    adj = np.empty(len(ps)); run = 0
    for i, j in enumerate(order):
        run = max(run, min(1, (len(ps) - i) * ps[j])); adj[j] = run
    return adj


def main():
    matched, deployed, rows = load()
    seeds_present = sorted({k[4] for k in matched})

    def vec(K, sub, A, M, arm="matched"):
        """Per-seed accuracies; K=5 pooled over subsets if sub == 'pooled'. None if incomplete."""
        tags = TAGS[K] if sub == "pooled" else [sub]
        out = []
        for S in seeds_present:
            vals = []
            for T in tags:
                if arm == "matched":
                    a_ = 0.0 if M in CTRL_MODELS else A
                    v = matched.get((K, T, a_, M, S))
                else:                                           # deployed: trained at 0, evaluated at A
                    v = deployed.get((K, T, 0.0, M, S, A))
                if v is None:
                    return None
                vals.append(v)
            out.append(np.mean(vals))
        return np.array(out)

    with open(ROOT / "v15_runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # ---- summary -------------------------------------------------------------------
    summ = []
    subs = lambda K: (["pooled", "A", "B"] if K == 5 else ["all"])  # noqa: E731
    for K in KS:
        for sub in subs(K):
            for arm, alist in (("matched", ALPHAS), ("deployed", EVALS)):
                for A in alist:
                    for M in MODELS:
                        v = vec(K, sub, A, M, arm)
                        if v is not None:
                            summ.append(dict(K=K, subset=sub, arm=arm, alpha=A, model=M, n=len(v), mean=v.mean(),
                                             sd=v.std(ddof=1) if len(v) > 1 else np.nan))
    with open(ROOT / "v15_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader(); w.writerows(summ)

    # ---- contrasts -----------------------------------------------------------------
    CON = {"F": ("M3", "M4"), "T": ("M3", "M5"), "R": ("M5", "M4"), "BE": ("M3", "M2"), "Q": ("M2", "M1")}
    TOST = {"F", "T", "R"}
    crows = []

    def add(K, sub, name, alpha, d, role, arm, margin=None):
        r = paired(d, margin)
        crows.append(dict(K=K, subset=sub, contrast=name, alpha=alpha, arm=arm, role=role, **r))
        return crows[-1]

    for K in KS:
        for sub in subs(K):
            for arm, alist in (("matched", ALPHAS), ("deployed", EVALS)):
                for A in alist:
                    for c, (a, b) in CON.items():
                        va, vb = vec(K, sub, A, a, arm), vec(K, sub, A, b, arm)
                        if va is not None and vb is not None:
                            add(K, sub, c, A, va - vb, "secondary", arm, MARGIN if c in TOST else None)
            for M in NOISE_MODELS:
                v0 = vec(K, sub, 0.0, M)
                for A in ALPHAS[1:]:
                    vA = vec(K, sub, A, M)
                    if v0 is not None and vA is not None:
                        add(K, sub, "NC_" + M, A, vA - v0, "secondary", "matched", MARGIN)
                for A in EVALS[1:]:
                    vD = vec(K, sub, A, M, "deployed")
                    if v0 is not None and vD is not None:
                        add(K, sub, "DEP_" + M, A, vD - v0, "secondary", "deployed", MARGIN)
                for A in ALPHAS[1:]:
                    vA, vD = vec(K, sub, A, M), vec(K, sub, A, M, "deployed")
                    if vA is not None and vD is not None:
                        add(K, sub, "REC_" + M, A, vA - vD, "secondary", "matched-deployed")
            # slopes over the training alphas (matched arm)
            al = np.array(ALPHAS)

            def slope_of(getd):
                ds = [getd(A) for A in ALPHAS]
                if any(d is None for d in ds):
                    return None
                D = np.stack(ds, 1)                                 # seeds x alphas
                X = al - al.mean()
                return (D - D.mean(1, keepdims=True)) @ X / (X @ X)

            for c, (a, b) in CON.items():
                s = slope_of(lambda A, a=a, b=b: (lambda x, y: None if x is None or y is None else x - y)(vec(K, sub, A, a), vec(K, sub, A, b)))
                if s is not None:
                    add(K, sub, "slope_" + c, np.nan, s, "secondary", "matched")
            for M in NOISE_MODELS:
                s = slope_of(lambda A, M=M: vec(K, sub, A, M))
                if s is not None:
                    add(K, sub, "slope_NC_" + M, np.nan, s, "secondary", "matched")

        # ---- primary, Holm within K (pooled for K=5) --------------------------------
        sub = "pooled" if K == 5 else "all"
        prim = []
        for name, al_ in (("NC_M3", 1.0), ("F", 1.0), ("T", 1.0), ("slope_F", None)):
            for r in crows:
                if (r["K"], r["subset"], r["contrast"], r["arm"]) == (K, sub, name, "matched") and \
                        ((al_ is None and np.isnan(r["alpha"])) or r["alpha"] == al_):
                    prim.append(r)
        if len(prim) == 4:
            for r, ph in zip(prim, holm([r["p"] for r in prim])):
                r["role"] = "PRIMARY"; r["p_holm"] = ph
        else:
            print("WARNING K=%d: only %d of 4 primary contrasts computable" % (K, len(prim)))

    keys = list(crows[0].keys()) + ["p_holm"]
    with open(ROOT / "v15_contrasts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in crows:
            w.writerow({k: r.get(k, "") for k in keys})

    # ---- print -------------------------------------------------------------------
    def fmt(r):
        s = "%+6.2f [%+.2f, %+.2f] p=%.3g MDE=%.2f" % (r["diff"], r["lo"], r["hi"], r["p"], r["mde"])
        if r.get("p_holm", "") != "":
            s += " p_holm=%.3g" % r["p_holm"]
        if r.get("equivalent"):
            s += " TOST(+/-1)=%s p=%.3g" % (r["equivalent"], r["p_tost"])
        return s

    print("v15 runs: %d   seeds: %s   calibration: %s" % (len(rows), seeds_present, SHA[:12]))
    for K in KS:
        sub = "pooled" if K == 5 else "all"
        print("\n" + "=" * 92 + "\nK=%d (%s)\n" % (K, sub) + "=" * 92)
        print("Accuracy %, matched arm (trained and tested at alpha), mean +/- seed SD:")
        print("  alpha  " + "".join("%-15s" % M for M in MODELS))
        for A in ALPHAS:
            line = "  %-6g " % A
            for M in MODELS:
                v = vec(K, sub, A, M)
                line += ("%5.2f +/- %4.2f  " % (v.mean(), v.std(ddof=1))) if v is not None and len(v) > 1 else "%-15s" % "-"
            print(line)
        print("Accuracy %, deployed arm (trained at alpha=0, tested at alpha):")
        for A in EVALS:
            line = "  %-6g " % A
            for M in MODELS:
                v = vec(K, sub, A, M, "deployed")
                line += ("%5.2f +/- %4.2f  " % (v.mean(), v.std(ddof=1))) if v is not None and len(v) > 1 else "%-15s" % "-"
            print(line)
        print("PRIMARY (Holm within K, 4 tests):")
        for r in crows:
            if r["K"] == K and r["role"] == "PRIMARY":
                print("  %-8s alpha=%-4s %s" % (r["contrast"], "" if np.isnan(r["alpha"]) else "%g" % r["alpha"], fmt(r)))
        print("Secondary, matched arm:")
        for c in ("F", "T", "R", "BE", "Q", "NC_M1", "NC_M2", "NC_M3", "NC_M5"):
            for r in crows:
                if (r["K"], r["subset"], r["contrast"], r["arm"], r["role"]) == (K, sub, c, "matched", "secondary"):
                    print("  %-8s alpha=%-4g %s" % (c, r["alpha"], fmt(r)))
        print("Secondary, slopes over alpha (pp per unit alpha):")
        for r in crows:
            if r["K"] == K and r["subset"] == sub and r["contrast"].startswith("slope_") and r["role"] == "secondary":
                print("  %-12s %s" % (r["contrast"], fmt(r)))
        print("Secondary, deployed arm (DEP: cost without noise-aware training; REC: recovered by it):")
        for c in ("DEP_M1", "DEP_M2", "DEP_M3", "DEP_M5", "REC_M1", "REC_M2", "REC_M3", "REC_M5", "F"):
            arm = "deployed" if not c.startswith("REC") else "matched-deployed"
            for r in crows:
                if (r["K"], r["subset"], r["contrast"], r["arm"]) == (K, sub, c, arm) and r["alpha"] != 0:
                    print("  %-8s alpha=%-4g %s" % (c, r["alpha"], fmt(r)))
    print("\nWrote v15_runs.csv, v15_summary.csv, v15_contrasts.csv")


if __name__ == "__main__":
    main()
