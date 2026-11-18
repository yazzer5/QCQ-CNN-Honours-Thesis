#!/usr/bin/env python3
"""analyse_v157.py <repo157> [--force]  -> <repo157>/v157_out/ (curves.csv, summary.csv, primary.csv, figures/)
Refuses without .grid_ok unless --force (local testing only)."""
import csv
import glob
import json
import os
import sys

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODELS = ("M3", "M4", "M5")
COL = {"M3": "#2f6fdb", "M4": "#c2410c", "M5": "#7c3aed"}
LBL = {"M3": "M3 trained circuit", "M4": "M4 no front-end", "M5": "M5 frozen circuit"}
ARMS = (("0", "clean-trained"), ("0.3", "noise-trained"))
EARLY = (1, 2, 3, 5, 7, 10, 15, 20, 25, 30)
THRESH = 0.35


def holm(ps):
    o = np.argsort(ps)
    adj, run = np.empty(len(ps)), 0.0
    for i, j in enumerate(o):
        run = max(run, (len(ps) - i) * ps[j])
        adj[j] = min(1.0, run)
    return adj


def paired(d):
    d = np.asarray(d, float)
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    q = stats.t.ppf(.975, n - 1)
    p = 2 * stats.t.sf(abs(d.mean() / se), n - 1) if se > 0 else (0.0 if d.mean() else 1.0)
    return dict(n=n, mean=d.mean(), lo=d.mean() - q * se, hi=d.mean() + q * se, p=p,
                mde=(q + stats.t.ppf(.8, n - 1)) * se, n_neg=int((d < 0).sum()))


def value_at(rec, c, key):
    """value at checkpoint c, carrying the last value forward if the run stopped earlier"""
    v = None
    for r in rec:
        if r["epoch"] <= c:
            v = r[key]
    return v


def main():
    repo = sys.argv[1]
    if not os.path.exists(os.path.join(repo, ".grid_ok")) and "--force" not in sys.argv:
        sys.exit("REFUSED: no .grid_ok (run gate_v157.py grid first)")
    out = os.path.join(repo, "v157_out")
    fig = os.path.join(out, "figures")
    os.makedirs(fig, exist_ok=True)
    runs = {}
    for arm, _ in ARMS:
        for f in glob.glob(os.path.join(repo, "results_v157_k5_A_s%s" % arm, "track_M*_seed*.json")):
            t = json.load(open(f))
            runs[(arm, t["meta"]["model"], t["meta"]["seed"])] = t["records"]
    seeds = sorted(set(k[2] for k in runs))

    def series(rec, key, met):
        return [(r["epoch"], r[key][met]) for r in rec]

    with open(os.path.join(out, "v157_curves.csv"), "w") as fh:
        w = csv.writer(fh)
        w.writerow(["arm", "model", "seed", "epoch", "layer", "sigma", "metric", "value"])
        for (arm, M, S), rec in sorted(runs.items()):
            for r in rec:
                for key, m in r.items():
                    if "|" in key:
                        for k, v in m.items():
                            w.writerow([arm, M, S, r["epoch"], key.split("|")[0], key.split("|")[1], k, v])

    summ = {}
    for (arm, M, S), rec in runs.items():
        rad = series(rec, "L2|0.3", "radius")
        final = rad[-1][1]
        early = np.mean([value_at([dict(epoch=e, v=v) for e, v in rad], c, "v") for c in EARLY])
        tight = next(e for e, v in rad if v <= 1.1 * final)
        thr = next((e for e, v in rad if v <= THRESH), rec[-1]["epoch"])
        eb = np.mean([value_at([dict(epoch=e, v=v) for e, v in series(rec, "L2|0.3", "bleed")], c, "v") for c in EARLY])
        ee = np.mean([value_at([dict(epoch=e, v=v) for e, v in series(rec, "L2|0.3", "eta")], c, "v") for c in EARLY])
        er = np.mean([value_at([dict(epoch=e, v=v) for e, v in series(rec, "L3|0.3", "bleed")], c, "v") for c in EARLY])
        summ[(arm, M, S)] = dict(early_radius=early, final_radius=final, epochs_to_tight=tight, epochs_to_thresh=thr,
                                 early_bleed_z=eb, early_eta_z=ee, early_error=er, n_ckpt=len(rec), last_epoch=rec[-1]["epoch"])
    with open(os.path.join(out, "v157_summary.csv"), "w") as fh:
        keys = list(next(iter(summ.values())).keys())
        w = csv.writer(fh)
        w.writerow(["arm", "model", "seed"] + keys)
        for k in sorted(summ):
            w.writerow(list(k) + [summ[k][x] for x in keys])

    def contrast(arm, a, b, met):
        ss = [s for s in seeds if (arm, a, s) in summ and (arm, b, s) in summ]
        return paired([summ[(arm, a, s)][met] - summ[(arm, b, s)][met] for s in ss])

    prim = []
    for pid, arm, cn, a, b in (("P-Fc", "0", "F", "M3", "M4"), ("P-Tc", "0", "T", "M3", "M5"),
                               ("P-Fn", "0.3", "F", "M3", "M4"), ("P-Tn", "0.3", "T", "M3", "M5")):
        r = contrast(arm, a, b, "early_radius")
        r.update(id=pid, arm=dict(ARMS)[arm], contrast=cn)
        prim.append(r)
    for r, pa in zip(prim, holm(np.array([r["p"] for r in prim]))):
        r["p_holm"] = pa
    print("PRIMARY: early-training bubble radius at z (mean over epochs 1-30, test sigma 0.3); Holm over 4")
    for r in prim:
        print("  %-5s %-14s %s: %+.4f [%+.4f, %+.4f]  p=%.2g  p_holm=%.2g  MDE=%.4f  (M3 smaller in %d/%d)"
              % (r["id"], r["arm"], r["contrast"], r["mean"], r["lo"], r["hi"], r["p"], r["p_holm"], r["mde"], r["n_neg"], r["n"]))
    print("SECONDARY (uncorrected)")
    sec = []
    for arm, an in ARMS:
        for met in ("epochs_to_thresh", "epochs_to_tight", "final_radius", "early_bleed_z", "early_eta_z", "early_error"):
            for cn, a, b in (("F", "M3", "M4"), ("T", "M3", "M5")):
                r = contrast(arm, a, b, met)
                r.update(arm=an, metric=met, contrast=cn,
                         mean_a=np.mean([summ[(arm, a, s)][met] for s in seeds if (arm, a, s) in summ]),
                         mean_b=np.mean([summ[(arm, b, s)][met] for s in seeds if (arm, b, s) in summ]))
                sec.append(r)
                print("  %-14s %-17s %s: %s=%.3f %s=%.3f diff %+.3f [%+.3f, %+.3f] p=%.2g (M3 lower in %d/%d)"
                      % (an, met, cn, a, r["mean_a"], b, r["mean_b"], r["mean"], r["lo"], r["hi"], r["p"], r["n_neg"], r["n"]))
    with open(os.path.join(out, "v157_primary_secondary.csv"), "w") as fh:
        keys = ["id", "arm", "metric", "contrast", "n", "mean", "lo", "hi", "p", "p_holm", "mde", "n_neg"]
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in prim:
            w.writerow(dict(r, metric="early_radius"))
        for r in sec:
            w.writerow(r)

    grid = sorted(set(e for rec in runs.values() for e in (r["epoch"] for r in rec)))
    grid = [e for e in grid if e <= 120]

    def mean_curve(arm, M, key, met):
        mu, ci = [], []
        for c in grid:
            v = [value_at([dict(epoch=e, v=x) for e, x in series(runs[(arm, M, s)], key, met)], c, "v")
                 for s in seeds if (arm, M, s) in runs]
            v = np.array([x for x in v if x is not None], float)
            mu.append(v.mean())
            ci.append(stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0)
        return np.array(mu), np.array(ci)

    # E1 headline: bubble radius at z vs epoch
    f, ax = plt.subplots(1, 2, figsize=(14, 5))
    for i, (arm, an) in enumerate(ARMS):
        for M in MODELS:
            mu, ci = mean_curve(arm, M, "L2|0.3", "radius")
            ax[i].plot(grid, mu, color=COL[M], marker="o", ms=3, label=LBL[M])
            ax[i].fill_between(grid, mu - ci, mu + ci, color=COL[M], alpha=.15)
        ax[i].axhline(THRESH, color="k", ls=":", lw=.8)
        ax[i].axvspan(1, 30, color="#eee", zorder=-1)
        ax[i].set_xscale("symlog", linthresh=10)
        ax[i].set_xlabel("epochs completed")
        ax[i].set_ylabel("bubble radius at z (noisy cloud around the true point / class distance)")
        ax[i].set_title("%s, test σ = 0.3 (grey = early window, dotted = M4's final v15.6 radius)" % an, fontsize=9)
        ax[i].grid(alpha=.3)
    ax[0].legend()
    f.suptitle("E1. Who forms the tight bubble faster? (K5 A, mean ± 95% CI over 10 seeds)")
    f.tight_layout()
    f.savefig(os.path.join(fig, "E1_radius_vs_epoch.png"), dpi=200)
    plt.close(f)

    # E2 components: shift, spread, bleed, error vs epoch
    comp = (("L2|0.3", "eta", "shift η at z"), ("L2|0.3", "rho_W", "size ρ_W at z"),
            ("L2|0.3", "bleed", "bleed at z"), ("L3|0.3", "bleed", "test error (σ = 0.3)"))
    f, ax = plt.subplots(2, 4, figsize=(20, 8.5))
    for i, (arm, an) in enumerate(ARMS):
        for j, (key, met, lab) in enumerate(comp):
            for M in MODELS:
                mu, ci = mean_curve(arm, M, key, met)
                ax[i, j].plot(grid, mu, color=COL[M], marker="o", ms=2.5, label=LBL[M])
                ax[i, j].fill_between(grid, mu - ci, mu + ci, color=COL[M], alpha=.15)
            ax[i, j].set_xscale("symlog", linthresh=10)
            ax[i, j].set_title("%s — %s" % (an, lab), fontsize=9)
            ax[i, j].grid(alpha=.3)
    ax[0, 0].legend(fontsize=7)
    f.suptitle("E2. The bubble's parts during training (K5 A, test σ = 0.3)")
    f.tight_layout()
    f.savefig(os.path.join(fig, "E2_components_vs_epoch.png"), dpi=200)
    plt.close(f)

    # E3 per-seed curves
    f, ax = plt.subplots(2, 3, figsize=(17, 8.5), sharey="row")
    for i, (arm, an) in enumerate(ARMS):
        for j, M in enumerate(MODELS):
            for s in seeds:
                if (arm, M, s) in runs:
                    e, v = zip(*series(runs[(arm, M, s)], "L2|0.3", "radius"))
                    ax[i, j].plot(e, v, color="#d00" if s == 46 else COL[M], lw=1.8 if s == 46 else .8, alpha=.8)
            ax[i, j].set_xscale("symlog", linthresh=10)
            ax[i, j].set_title("%s — %s (red = seed 46)" % (an, LBL[M]), fontsize=9)
            ax[i, j].grid(alpha=.3)
    f.suptitle("E3. Every seed's bubble radius at z vs epoch")
    f.tight_layout()
    f.savefig(os.path.join(fig, "E3_per_seed.png"), dpi=200)
    plt.close(f)

    # E4 standard set: early radius strip/bar + paired diffs; effect vs MDE
    f, ax = plt.subplots(2, 3, figsize=(17, 9))
    for i, (arm, an) in enumerate(ARMS):
        for k, M in enumerate(MODELS):
            v = np.array([summ[(arm, M, s)]["early_radius"] for s in seeds if (arm, M, s) in summ])
            ax[i, 0].bar(k, v.mean(), yerr=stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)),
                         color=COL[M], alpha=.3, capsize=4)
            ax[i, 0].scatter(np.full(len(v), k) + np.linspace(-.12, .12, len(v)), v, color=COL[M], s=16)
        ax[i, 0].set_xticks(range(3))
        ax[i, 0].set_xticklabels(MODELS)
        ax[i, 0].set_title("%s: early radius (mean ± 95%% CI, dots = seeds)" % an, fontsize=9)
        for j, (cn, a, b) in enumerate((("F", "M3", "M4"), ("T", "M3", "M5"))):
            ss = [s for s in seeds if (arm, a, s) in summ and (arm, b, s) in summ]
            d = [summ[(arm, a, s)]["early_radius"] - summ[(arm, b, s)]["early_radius"] for s in ss]
            ax[i, j + 1].bar(range(len(ss)), d, color=["#555" if x < 0 else "#bbb" for x in d])
            ax[i, j + 1].axhline(0, color="k", lw=.8)
            ax[i, j + 1].set_xticks(range(len(ss)))
            ax[i, j + 1].set_xticklabels(ss, rotation=90, fontsize=7)
            ax[i, j + 1].set_title("%s: per-seed %s = %s − %s (below 0 = M3 tighter)" % (an, cn, a, b), fontsize=9)
    f.suptitle("E4. Primary: early-training bubble radius (epochs 1-30)")
    f.tight_layout()
    f.savefig(os.path.join(fig, "E4_primary_standard.png"), dpi=200)
    plt.close(f)

    # E5 epochs to threshold / to tight
    f, ax = plt.subplots(1, 2, figsize=(13, 5))
    for j, (met, lab) in enumerate((("epochs_to_thresh", "epochs until radius ≤ %.2f" % THRESH),
                                    ("epochs_to_tight", "epochs until within 10% of own final radius"))):
        for i, (arm, an) in enumerate(ARMS):
            for k, M in enumerate(MODELS):
                v = np.array([summ[(arm, M, s)][met] for s in seeds if (arm, M, s) in summ], float)
                x = i * 4 + k
                ax[j].scatter(np.full(len(v), x) + np.linspace(-.15, .15, len(v)), v, color=COL[M], s=16)
                ax[j].bar(x, np.median(v), color=COL[M], alpha=.3)
        ax[j].set_xticks([0, 1, 2, 4, 5, 6])
        ax[j].set_xticklabels(["M3", "M4", "M5"] * 2)
        ax[j].set_title("%s (bars = median; left 3 clean-trained, right 3 noise-trained)" % lab, fontsize=9)
        ax[j].grid(alpha=.3)
    f.suptitle("E5. How many epochs until the bubble is tight?")
    f.tight_layout()
    f.savefig(os.path.join(fig, "E5_epochs_to_tight.png"), dpi=200)
    plt.close(f)
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
