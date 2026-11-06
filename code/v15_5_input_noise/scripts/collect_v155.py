#!/usr/bin/env python3
"""
v15.5 (pixel noise on the input images): collect results and run the PRE-REGISTERED
contrasts. Run from the v15.5 repo root AFTER the grid gate passes (pbs/.grid_ok):

    python3 scripts/collect_v155.py | tee collect_v155.log

Two arms from one grid (PREREGISTRATION_v15_5.md section 4):
  MATCHED  -- trained, validated AND tested at sigma (noise augmentation) = the job's test_acc.
  DEPLOYED -- trained clean (sigma=0), tested at sigma = cross-evaluation of the sigma=0 job.
              Includes sigma = 0.7, 1.0 (stress) and salt-and-pepper p (exploratory).

Contrasts, paired over seeds (K=5 pooled = per-seed mean of subsets A and B, df=9):
    F  = M3 - M4     trained quantum front-end vs none          (+TOST +/-1 pp)
    T  = M3 - M5     training vs freezing the quantum front-end (+TOST)
    R  = M5 - M4     frozen random quantum front-end vs none    (+TOST)
    BE = M3 - M2     classical vs quantum head
    Q  = M2 - M1     Circuit-4 trained vs RandomLayers frozen, quantum head
    NC_m(s)  = m(s) - m(0)              matched-arm noise cost (+TOST)
    DEP_m(s) = m_deployed(s) - m(0)     cost when trained clean and tested noisy (+TOST)
    REC_m(s) = m(s) - m_deployed(s)     what noise-aware training recovers
    slope(X) = per-seed OLS slope of X on sigma over the training sigmas, pp per 0.1 sigma;
               matched arm, and "deployed" arm at the same sigmas.

PRIMARY (Holm within each K, 4 tests): F deployed @0.3, slope of deployed F,
    F matched @0.3, T deployed @0.3.
SECONDARY (uncorrected, all with MDE): everything at every sigma, both arms; slopes;
    per-subset results.
EXPLORATORY (labelled): salt-and-pepper deployed contrasts; T without seed 46.
Writes v155_runs.csv, v155_summary.csv, v155_contrasts.csv, v155_transfer.csv.
"""
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
MARGIN = 1.0
SLOPE_UNIT = 0.1                    # slopes in pp per 0.1 sigma


def cfg():
    names = ["GRID_KS", "GRID_SIGMAS", "EVAL_SIGMAS", "EVAL_SP", "GRID_SEEDS", "MODELS"]
    cmd = "cd '%s/pbs' && source ./config155.sh && " % ROOT + " && ".join('echo "${%s}"' % n for n in names)
    out = subprocess.run(["bash", "-c", cmd], stdout=subprocess.PIPE, universal_newlines=True, check=True).stdout.split("\n")
    return ([int(x) for x in out[0].split()], [float(x) for x in out[1].split()], [float(x) for x in out[2].split()],
            [float(x) for x in out[3].split()], [int(x) for x in out[4].split()], out[5].split())


KS, SIGMAS, EVALS, EVAL_SP, SEEDS, MODELS = cfg()
TAGS = {5: ["A", "B"], 10: ["all"]}
CON = {"F": ("M3", "M4"), "T": ("M3", "M5"), "R": ("M5", "M4"), "BE": ("M3", "M2"), "Q": ("M2", "M1")}
TOST = {"F", "T", "R"}
PRIMARY = [("F", "deployed", 0.3), ("slope_F", "deployed", None), ("F", "matched", 0.3), ("T", "deployed", 0.3)]


def load():
    if not (ROOT / "pbs" / ".grid_ok").exists():
        sys.exit("pbs/.grid_ok missing: the grid gate has not passed. Do not read results.")
    matched, xe, xsp, rows = {}, {}, {}, []
    for d in sorted(ROOT.glob("results_v155_k*_*_s*")):
        m = re.match(r"results_v155_k(\d+)_(\w+?)_s([\d.]+)$", d.name)
        if not m:
            continue
        K, TAG, SG = int(m.group(1)), m.group(2), float(m.group(3))
        for f in sorted(d.glob("result_M*_*.json")):
            j = json.loads(f.read_text())
            M = j.get("model") or re.match(r"result_(M\d)_", f.name).group(1)
            S = int(j.get("seed", re.search(r"seed(\d+)", f.name).group(1)))
            if j.get("loss") not in ("ce", None):
                sys.exit("%s: loss=%s. Stop." % (f, j.get("loss")))
            sc = d / ("v155_xeval_%s_seed%d.json" % (M, S))
            if not sc.exists():
                sys.exit("%s missing. Stop: the gate should have caught this." % sc)
            s = json.loads(sc.read_text())
            if s["selfcheck"] != "MATCH" or float(s["sigma_train"]) != SG:
                sys.exit("%s: cross-eval sidecar invalid (%s). Stop." % (sc, s["selfcheck"]))
            acc = 100 * float(j["test_acc"])
            matched[(K, TAG, SG, M, S)] = acc
            for e, v in s["acc"].items():
                xe[(K, TAG, SG, M, S, float(e))] = 100 * float(v)
            for p, v in s["acc_sp"].items():
                xsp[(K, TAG, SG, M, S, float(p))] = 100 * float(v)
            row = dict(K=K, subset=TAG, sigma_train=SG, model=M, seed=S, test_acc=acc,
                       best_epoch=j.get("best_epoch"), stop_epoch=j.get("stop_epoch"),
                       train_acc_at_best=(100 * float(j["train_acc_at_best"]) if j.get("train_acc_at_best") is not None else ""))
            row.update(("acc_eval_g%g" % float(e), 100 * float(v)) for e, v in sorted(s["acc"].items(), key=lambda kv: float(kv[0])))
            row.update(("acc_eval_sp%g" % float(p), 100 * float(v)) for p, v in sorted(s["acc_sp"].items(), key=lambda kv: float(kv[0])))
            rows.append(row)
    if not rows:
        sys.exit("no results found under results_v155_k*")
    return matched, xe, xsp, rows


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


class Data(object):
    def __init__(self):
        self.matched, self.xe, self.xsp, self.rows = load()
        self.seeds = sorted({k[4] for k in self.matched})

    def vec(self, K, sub, s, M, arm="matched", seeds=None, train=None):
        """Per-seed accuracies; K=5 pooled over subsets if sub == 'pooled'. None if incomplete.
        arm: matched (trained and tested at s), deployed (trained at 0 / `train`, tested at Gaussian s),
             sp (trained at 0 / `train`, tested at salt-and-pepper p = s)."""
        tags = TAGS[K] if sub == "pooled" else [sub]
        out = []
        for S in (seeds or self.seeds):
            vals = []
            for T in tags:
                if arm == "matched":
                    v = self.matched.get((K, T, s, M, S))
                elif arm == "deployed":
                    v = self.xe.get((K, T, 0.0 if train is None else train, M, S, s))
                else:
                    v = self.xsp.get((K, T, 0.0 if train is None else train, M, S, s))
                if v is None:
                    return None
                vals.append(v)
            out.append(np.mean(vals))
        return np.array(out)


def slope_of(ds):
    """ds: list (one per SIGMAS) of per-seed arrays -> per-seed OLS slopes in pp per SLOPE_UNIT sigma."""
    if any(d is None for d in ds):
        return None
    D = np.stack(ds, 1)
    X = np.array(SIGMAS) - np.mean(SIGMAS)
    return (D - D.mean(1, keepdims=True)) @ X / (X @ X) * SLOPE_UNIT


def subs(K):
    return ["pooled", "A", "B"] if K == 5 else ["all"]


def main():
    D = Data()
    V = D.vec
    with open(ROOT / "v155_runs.csv", "w", newline="") as f:
        keys = list(D.rows[0].keys())
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(D.rows)

    # ---- summary -------------------------------------------------------------------
    summ = []
    for K in KS:
        for sub in subs(K):
            for arm, slist in (("matched", SIGMAS), ("deployed", EVALS), ("sp", EVAL_SP)):
                for s in slist:
                    for M in MODELS:
                        v = V(K, sub, s, M, arm)
                        if v is not None:
                            summ.append(dict(K=K, subset=sub, arm=arm, level=s, model=M, n=len(v), mean=v.mean(),
                                             sd=v.std(ddof=1) if len(v) > 1 else np.nan))
    with open(ROOT / "v155_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summ[0].keys()))
        w.writeheader(); w.writerows(summ)

    # ---- transfer matrix (exploratory): trained at s_t, tested at s_e -----------------
    trows = []
    for K in KS:
        sub = "pooled" if K == 5 else "all"
        for M in MODELS:
            for st in SIGMAS:
                for se in EVALS:
                    v = V(K, sub, se, M, "deployed", train=st)
                    if v is not None:
                        trows.append(dict(K=K, subset=sub, model=M, sigma_train=st, sigma_test=se, n=len(v),
                                          mean=v.mean(), sd=v.std(ddof=1) if len(v) > 1 else np.nan))
    with open(ROOT / "v155_transfer.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(trows[0].keys()))
        w.writeheader(); w.writerows(trows)

    # ---- contrasts -----------------------------------------------------------------
    crows = []

    def add(K, sub, name, level, d, role, arm, margin=None):
        r = paired(d, margin)
        crows.append(dict(K=K, subset=sub, contrast=name, level=level, arm=arm, role=role, **r))
        return crows[-1]

    no46 = [s for s in D.seeds if s != 46]
    for K in KS:
        for sub in subs(K):
            for arm, slist in (("matched", SIGMAS), ("deployed", EVALS)):
                for s in slist:
                    for c, (a, b) in CON.items():
                        va, vb = V(K, sub, s, a, arm), V(K, sub, s, b, arm)
                        if va is not None and vb is not None:
                            add(K, sub, c, s, va - vb, "secondary", arm, MARGIN if c in TOST else None)
            for M in MODELS:
                v0 = V(K, sub, 0.0, M)
                for s in SIGMAS[1:]:
                    vs = V(K, sub, s, M)
                    if v0 is not None and vs is not None:
                        add(K, sub, "NC_" + M, s, vs - v0, "secondary", "matched", MARGIN)
                for s in EVALS[1:]:
                    vd = V(K, sub, s, M, "deployed")
                    if v0 is not None and vd is not None:
                        add(K, sub, "DEP_" + M, s, vd - v0, "secondary", "deployed", MARGIN)
                for s in SIGMAS[1:]:
                    vs, vd = V(K, sub, s, M), V(K, sub, s, M, "deployed")
                    if vs is not None and vd is not None:
                        add(K, sub, "REC_" + M, s, vs - vd, "secondary", "matched-deployed")
            for arm in ("matched", "deployed"):
                for c, (a, b) in CON.items():
                    ds = []
                    for s in SIGMAS:
                        va, vb = V(K, sub, s, a, arm), V(K, sub, s, b, arm)
                        ds.append(None if va is None or vb is None else va - vb)
                    sl = slope_of(ds)
                    if sl is not None:
                        add(K, sub, "slope_" + c, np.nan, sl, "secondary", arm)
                for M in MODELS:
                    sl = slope_of([V(K, sub, s, M, arm) for s in SIGMAS])
                    if sl is not None:
                        add(K, sub, ("slope_NC_" if arm == "matched" else "slope_DEP_") + M, np.nan, sl, "secondary", arm)
            # ---- exploratory: salt-and-pepper (deployed) and T without seed 46
            for p in EVAL_SP:
                for c, (a, b) in CON.items():
                    va, vb = V(K, sub, p, a, "sp"), V(K, sub, p, b, "sp")
                    if va is not None and vb is not None:
                        add(K, sub, c, p, va - vb, "exploratory", "sp-deployed", MARGIN if c in TOST else None)
                for M in MODELS:
                    v0, vp = V(K, sub, 0.0, M), V(K, sub, p, M, "sp")
                    if v0 is not None and vp is not None:
                        add(K, sub, "DEP_" + M, p, vp - v0, "exploratory", "sp-deployed", MARGIN)
            if 46 in D.seeds:
                for arm, slist in (("matched", SIGMAS), ("deployed", EVALS)):
                    for s in slist:
                        va, vb = V(K, sub, s, "M3", arm, seeds=no46), V(K, sub, s, "M5", arm, seeds=no46)
                        if va is not None and vb is not None:
                            add(K, sub, "T_no46", s, va - vb, "exploratory", arm, MARGIN)

        # ---- primary, Holm within K (pooled for K=5) --------------------------------
        sub = "pooled" if K == 5 else "all"
        prim = []
        for name, arm, lv in PRIMARY:
            for r in crows:
                if (r["K"], r["subset"], r["contrast"], r["arm"]) == (K, sub, name, arm) and r["role"] == "secondary" and \
                        ((lv is None and np.isnan(r["level"])) or r["level"] == lv):
                    prim.append(r)
                    break
        if len(prim) == len(PRIMARY):
            for r, ph in zip(prim, holm([r["p"] for r in prim])):
                r["role"] = "PRIMARY"; r["p_holm"] = ph
        else:
            print("WARNING K=%d: only %d of %d primary contrasts computable" % (K, len(prim), len(PRIMARY)))

    keys = list(crows[0].keys()) + ["p_holm"]
    with open(ROOT / "v155_contrasts.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in crows:
            w.writerow(dict((k, r.get(k, "")) for k in keys))

    # ---- print -------------------------------------------------------------------
    def fmt(r):
        s = "%+6.2f [%+.2f, %+.2f] p=%.3g MDE=%.2f" % (r["diff"], r["lo"], r["hi"], r["p"], r["mde"])
        if r.get("p_holm", "") != "":
            s += " p_holm=%.3g" % r["p_holm"]
        if r.get("equivalent"):
            s += " TOST(+/-1)=%s p=%.3g" % (r["equivalent"], r["p_tost"])
        return s

    def lvl(r):
        return "" if np.isnan(r["level"]) else "%g" % r["level"]

    def table(K, sub, arm, slist, label):
        print(label)
        print("  level  " + "".join("%-15s" % M for M in MODELS))
        for s in slist:
            line = "  %-6g " % s
            for M in MODELS:
                v = V(K, sub, s, M, arm)
                line += ("%5.2f +/- %4.2f  " % (v.mean(), v.std(ddof=1))) if v is not None and len(v) > 1 else "%-15s" % "-"
            print(line)

    print("v15.5 runs: %d   seeds: %s" % (len(D.rows), D.seeds))
    for K in KS:
        sub = "pooled" if K == 5 else "all"
        print("\n" + "=" * 96 + "\nK=%d (%s)\n" % (K, sub) + "=" * 96)
        table(K, sub, "matched", SIGMAS, "Accuracy %, MATCHED arm (trained and tested at sigma), mean +/- seed SD:")
        table(K, sub, "deployed", EVALS, "Accuracy %, DEPLOYED arm (trained clean, tested at sigma):")
        table(K, sub, "sp", EVAL_SP, "Accuracy %, salt-and-pepper p, trained clean (EXPLORATORY):")
        print("PRIMARY (Holm within K, 4 tests):")
        for r in crows:
            if r["K"] == K and r["role"] == "PRIMARY":
                print("  %-9s %-9s sigma=%-4s %s" % (r["contrast"], r["arm"], lvl(r), fmt(r)))
        for arm in ("deployed", "matched"):
            print("Secondary, %s arm:" % arm)
            for c in ("F", "T", "R", "BE", "Q"):
                for r in crows:
                    if (r["K"], r["subset"], r["contrast"], r["arm"], r["role"]) == (K, sub, c, arm, "secondary") and r["level"] != 0:
                        print("  %-8s sigma=%-4s %s" % (c, lvl(r), fmt(r)))
        print("Secondary, noise costs (NC matched, DEP deployed, REC = matched - deployed):")
        for c in ["NC_" + M for M in MODELS] + ["DEP_" + M for M in MODELS] + ["REC_" + M for M in MODELS]:
            for r in crows:
                if (r["K"], r["subset"], r["contrast"], r["role"]) == (K, sub, c, "secondary"):
                    print("  %-8s sigma=%-4s %s" % (c, lvl(r), fmt(r)))
        print("Secondary, slopes over sigma (pp per 0.1 sigma):")
        for r in crows:
            if r["K"] == K and r["subset"] == sub and r["contrast"].startswith("slope_") and r["role"] == "secondary":
                print("  %-12s %-9s %s" % (r["contrast"], r["arm"], fmt(r)))
        print("EXPLORATORY: salt-and-pepper (trained clean), and T without seed 46:")
        for r in crows:
            if r["K"] == K and r["subset"] == sub and r["role"] == "exploratory" and \
                    (r["contrast"] in ("F", "T", "T_no46") or r["contrast"].startswith("DEP_")):
                print("  %-8s %-11s level=%-4s %s" % (r["contrast"], r["arm"], lvl(r), fmt(r)))
        if K == 5:
            print("Per subset (primaries' contrasts):")
            for tsub in ("A", "B"):
                for name, arm, lv in PRIMARY:
                    for r in crows:
                        if (r["K"], r["subset"], r["contrast"], r["arm"]) == (K, tsub, name, arm) and \
                                ((lv is None and np.isnan(r["level"])) or r["level"] == lv):
                            print("  %s %-9s %-9s %s" % (tsub, name, arm, fmt(r)))
    print("\nWrote v155_runs.csv, v155_summary.csv, v155_contrasts.csv, v155_transfer.csv")


if __name__ == "__main__":
    main()
