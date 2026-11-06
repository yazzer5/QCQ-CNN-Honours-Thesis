#!/usr/bin/env python3
"""
v15.5 standard plot set (PNG, 200 dpi) -> figures_v155/. Run after collect_v155.py:

    python3 scripts/plot_v155.py

  noise_curves.png        MATCHED arm: accuracy vs sigma, M1-M5, 95% CI
  deploy_curves.png       DEPLOYED arm: trained clean, tested at sigma (to 1.0), 95% CI   (headline)
  contrasts_vs_sigma.png  F, T, R, BE, Q vs sigma, both arms, 95% CI, +/-1 pp band
  recovery.png            REC_m = matched - deployed vs sigma, per model
  bars_k5.png, bars_k10.png       mean accuracy per sigma and model, both arms, seed-SD error bars
  strip_k5.png, strip_k10.png     every seed, both arms (per subset at K=5)
  paired_k5.png, paired_k10.png   per-seed paired differences F, T, R, BE, Q, both arms
  effect_vs_mde.png       |effect| vs MDE for every pooled contrast; primaries marked
  seed_sd_vs_sigma.png    seed SD vs sigma per model, both arms
Model colours are fixed per model in every figure; markers and direct labels carry identity too.
"""
import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collect_v155 as C  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures_v155"
OUT.mkdir(exist_ok=True)

SURF, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e8e7e2", "#c3c2b7"
COL = {"M1": "#2a78d6", "M2": "#eb6834", "M3": "#1baf7a", "M4": "#eda100", "M5": "#e87ba4"}
MK = {"M1": "o", "M2": "s", "M3": "D", "M4": "^", "M5": "v"}
NAME = {"M1": "M1 RandomLayers + QNN", "M2": "M2 Circuit 4 + QNN", "M3": "M3 Circuit 4 + classical",
        "M4": "M4 no front-end", "M5": "M5 Circuit 4 frozen"}
ARMCOL = {"matched": "#2a78d6", "deployed": "#eb6834"}
ARMLAB = {"matched": "matched (trained at sigma)", "deployed": "deployed (trained clean)"}
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
                     "legend.frameon": False, "lines.linewidth": 2})

D = C.Data()
V = D.vec
KS, SIGMAS, EVALS, MODELS = C.KS, C.SIGMAS, C.EVALS, C.MODELS
CONS = [("F", "F = M3 - M4\nquantum front-end vs none"), ("T", "T = M3 - M5\ntrained vs frozen"),
        ("R", "R = M5 - M4\nfrozen front-end vs none"), ("BE", "BE = M3 - M2\nclassical vs quantum head"),
        ("Q", "Q = M2 - M1\nCircuit 4 vs RandomLayers")]


def ci(v):
    if v is None or len(v) < 2:
        return np.nan, np.nan
    return v.mean(), stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))


def sub_of(K):
    return "pooled" if K == 5 else "all"


def ktitle(K):
    return "K=5 (subsets A, B pooled)" if K == 5 else "K=10"


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  " + name)


def curves(arm, slist, fname, title):
    fig, axes = plt.subplots(1, len(KS), figsize=(5.6 * len(KS), 4.2), squeeze=False)
    for ax, K in zip(axes[0], KS):
        ends = []
        for M in MODELS:
            mu, hw = zip(*[ci(V(K, sub_of(K), s, M, arm)) for s in slist])
            ax.errorbar(range(len(slist)), mu, yerr=hw, color=COL[M], marker=MK[M], ms=5, capsize=2, lw=2,
                        markeredgecolor=SURF, markeredgewidth=1, label=NAME[M])
            ends.append((mu[-1], M))
        ends.sort()
        ys = [e[0] for e in ends]
        lo, hi = ax.get_ylim()
        gap = (hi - lo) * 0.045
        for i in range(1, len(ys)):
            ys[i] = max(ys[i], ys[i - 1] + gap)
        for (y0, M), y in zip(ends, ys):
            ax.annotate(M, (len(slist) - 1, y0), xytext=(len(slist) - 0.75, y), color=INK2, fontsize=8,
                        va="center", annotation_clip=False)
        ax.set_ylim(top=min(ax.get_ylim()[1], 101))
        ax.set_xticks(range(len(slist)))
        ax.set_xticklabels(["%g" % s for s in slist])
        ax.set_xlabel("pixel-noise sigma (images in [0,1], clipped)")
        ax.set_ylabel("test accuracy (%)")
        ax.set_title(ktitle(K), loc="left")
        if 0.3 in slist:
            ax.axvline(slist.index(0.3), color=BASE, lw=1, ls=":", zorder=0)
    axes[0][0].legend(loc="lower left", fontsize=7.5)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, fname)


def read_contrasts():
    with open(ROOT / "v155_contrasts.csv") as f:
        rs = list(csv.DictReader(f))
    for r in rs:
        for k in ("K", "n"):
            r[k] = int(r[k])
        for k in ("level", "diff", "lo", "hi", "p", "mde"):
            r[k] = float(r[k]) if r[k] not in ("", "nan") else np.nan
    return rs


print("Writing figures_v155/:")
curves("matched", SIGMAS, "noise_curves.png",
       "Trained and tested on noisy images (matched arm), mean and 95% CI over seeds")
curves("deployed", EVALS, "deploy_curves.png",
       "Trained on clean images, tested on noisy images (deployed arm), mean and 95% CI")
CR = read_contrasts()

# ---- contrasts vs sigma (small multiples, both arms) -----------------------------------
fig, axes = plt.subplots(len(KS), len(CONS), figsize=(3.2 * len(CONS), 3.0 * len(KS)), squeeze=False)
X = dict((s, i) for i, s in enumerate(EVALS))
for i, K in enumerate(KS):
    for j, (c, lab) in enumerate(CONS):
        ax = axes[i][j]
        for arm, ls in (("matched", "-"), ("deployed", "--")):
            pts = sorted([r for r in CR if r["K"] == K and r["subset"] == sub_of(K) and r["contrast"] == c
                          and r["arm"] == arm and r["role"] in ("secondary", "PRIMARY") and not np.isnan(r["level"])],
                         key=lambda r: r["level"])
            if not pts:
                continue
            xs = [X[r["level"]] + (0.08 if arm == "deployed" else 0) for r in pts]
            ax.errorbar(xs, [r["diff"] for r in pts], yerr=[[r["diff"] - r["lo"] for r in pts], [r["hi"] - r["diff"] for r in pts]],
                        color=ARMCOL[arm], marker="o", ms=4, capsize=2, lw=1.6, ls=ls, label=ARMLAB[arm])
        ax.axhspan(-1, 1, color=GRID, alpha=0.6, zorder=0, lw=0)
        ax.axhline(0, color=BASE, lw=1, zorder=1)
        ax.set_xticks(range(len(EVALS)))
        ax.set_xticklabels(["%g" % s for s in EVALS], fontsize=7)
        if i == 0:
            ax.set_title(lab, fontsize=8.5, loc="left")
        if j == 0:
            ax.set_ylabel("%s\ndifference (pp)" % ktitle(K))
        if i == len(KS) - 1:
            ax.set_xlabel("sigma")
axes[0][0].legend(fontsize=7, loc="lower left")
fig.suptitle("Contrasts vs pixel-noise sigma, 95% CI; grey band = +/-1 pp equivalence margin", x=0.01, ha="left", fontsize=11)
fig.tight_layout()
save(fig, "contrasts_vs_sigma.png")

# ---- recovery ------------------------------------------------------------------------
fig, axes = plt.subplots(1, len(KS), figsize=(5.4 * len(KS), 3.8), squeeze=False)
XS = dict((s, i) for i, s in enumerate(SIGMAS))
for ax, K in zip(axes[0], KS):
    for k, M in enumerate(MODELS):
        pts = sorted([r for r in CR if r["K"] == K and r["subset"] == sub_of(K) and r["contrast"] == "REC_" + M],
                     key=lambda r: r["level"])
        if pts:
            off = (k - 2) * 0.06
            ax.errorbar([XS[r["level"]] + off for r in pts], [r["diff"] for r in pts],
                        yerr=[[r["diff"] - r["lo"] for r in pts], [r["hi"] - r["diff"] for r in pts]],
                        color=COL[M], marker=MK[M], ms=5, capsize=2, label=NAME[M], markeredgecolor=SURF)
    ax.axhline(0, color=BASE, lw=1)
    ax.set_xticks(range(len(SIGMAS))); ax.set_xticklabels(["%g" % s for s in SIGMAS])
    ax.set_xlabel("sigma"); ax.set_ylabel("matched - deployed (pp)"); ax.set_title(ktitle(K), loc="left")
axes[0][0].legend(fontsize=7.5)
fig.suptitle("What noise-aware training recovers (REC = trained at sigma - trained clean, both tested at sigma)",
             x=0.01, ha="left", fontsize=11)
fig.tight_layout()
save(fig, "recovery.png")

# ---- bars, strips, paired ------------------------------------------------------------
for K in KS:
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 6.8), squeeze=False)
    wbar = 0.16
    for ax, arm in zip(axes[:, 0], ("matched", "deployed")):
        for k, M in enumerate(MODELS):
            vs = [V(K, sub_of(K), s, M, arm) for s in SIGMAS]
            ax.bar(np.arange(len(SIGMAS)) + (k - 2) * wbar, [v.mean() for v in vs], wbar * 0.88,
                   yerr=[v.std(ddof=1) for v in vs], color=COL[M], label=NAME[M],
                   error_kw=dict(ecolor=INK2, lw=0.8, capsize=1.5))
        lo = min(V(K, sub_of(K), s, M, arm).min() for s in SIGMAS for M in MODELS)
        ax.set_ylim(max(0, lo - 5), 100)
        ax.set_xticks(range(len(SIGMAS))); ax.set_xticklabels(["sigma=%g" % s for s in SIGMAS])
        ax.set_ylabel("test accuracy (%)")
        ax.set_title("%s arm: %s" % (arm, ARMLAB[arm]), loc="left")
    axes[1][0].legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("%s: mean accuracy, error bars = seed SD" % ktitle(K), x=0.01, ha="left", fontsize=11)
    fig.tight_layout(); save(fig, "bars_k%d.png" % K)

    tags = C.TAGS[K]
    fig, axes = plt.subplots(2, len(SIGMAS), figsize=(2.4 * len(SIGMAS), 6.6), sharey="row", squeeze=False)
    for row, arm in enumerate(("matched", "deployed")):
        for ax, s in zip(axes[row], SIGMAS):
            for k, M in enumerate(MODELS):
                for t_i, T in enumerate(tags):
                    if arm == "matched":
                        ys = [D.matched.get((K, T, s, M, S)) for S in D.seeds]
                    else:
                        ys = [D.xe.get((K, T, 0.0, M, S, s)) for S in D.seeds]
                    ys = [y for y in ys if y is not None]
                    xs = k + (t_i - (len(tags) - 1) / 2.0) * 0.25 + np.linspace(-0.06, 0.06, len(ys))
                    ax.scatter(xs, ys, s=12, color=COL[M], marker=MK[M] if t_i == 0 else "x", linewidths=0.8,
                               edgecolors=SURF if t_i == 0 else None)
            ax.set_xticks(range(len(MODELS))); ax.set_xticklabels(MODELS, fontsize=7.5)
            ax.set_title("%s, sigma=%g" % (arm, s), loc="left", fontsize=8.5)
        axes[row][0].set_ylabel("test accuracy (%)")
    fig.suptitle("%s: every seed%s" % (ktitle(K), " (filled = subset A, x = subset B)" if K == 5 else ""),
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(); save(fig, "strip_k%d.png" % K)

    fig, axes = plt.subplots(2, len(CONS), figsize=(3.2 * len(CONS), 6.4), squeeze=False)
    for row, arm in enumerate(("deployed", "matched")):
        for ax, (c, lab) in zip(axes[row], CONS):
            a, b = C.CON[c]
            for xi, s in enumerate(SIGMAS):
                d = V(K, sub_of(K), s, a, arm) - V(K, sub_of(K), s, b, arm)
                ax.scatter(xi + np.linspace(-0.12, 0.12, len(d)), d, s=12, color=ARMCOL[arm], edgecolors=SURF, linewidths=0.6)
                ax.hlines(d.mean(), xi - 0.22, xi + 0.22, color=INK, lw=1.6)
            ax.axhspan(-1, 1, color=GRID, alpha=0.6, zorder=0, lw=0); ax.axhline(0, color=BASE, lw=1)
            ax.set_xticks(range(len(SIGMAS))); ax.set_xticklabels(["%g" % s for s in SIGMAS])
            ax.set_title(("%s\n" % lab.split("\n")[0]) + arm, loc="left", fontsize=8.5)
            if row == 1:
                ax.set_xlabel("sigma")
        axes[row][0].set_ylabel("per-seed difference (pp)")
    fig.suptitle("%s: per-seed paired differences (bar = mean; grey = +/-1 pp)" % ktitle(K), x=0.01, ha="left", fontsize=11)
    fig.tight_layout(); save(fig, "paired_k%d.png" % K)

# ---- effect vs MDE ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.6, 4.8))
pts = [r for r in CR if r["arm"] in ("matched", "deployed") and r["subset"] in ("pooled", "all") and np.isfinite(r["mde"])
       and r["role"] in ("secondary", "PRIMARY")]
for arm in ("matched", "deployed"):
    sec = [r for r in pts if r["role"] != "PRIMARY" and r["arm"] == arm]
    ax.scatter([r["mde"] for r in sec], [abs(r["diff"]) for r in sec], s=12, color=ARMCOL[arm], alpha=0.45,
               label="secondary, " + arm)
pri = [r for r in pts if r["role"] == "PRIMARY"]
ax.scatter([r["mde"] for r in pri], [abs(r["diff"]) for r in pri], s=46, color=INK, edgecolors=SURF, label="primary", zorder=3)
for r in pri:
    ax.annotate("%s %s%s K=%d" % (r["contrast"], r["arm"][:3], "" if np.isnan(r["level"]) else "@%g" % r["level"], r["K"]),
                (r["mde"], abs(r["diff"])), xytext=(4, 3), textcoords="offset points", fontsize=7, color=INK2)
m = max([r["mde"] for r in pts] + [abs(r["diff"]) for r in pts]) * 1.08
ax.plot([0, m], [0, m], color=BASE, lw=1, ls="--")
ax.set_xscale("symlog", linthresh=1); ax.set_yscale("symlog", linthresh=1)
ax.set_xlim(0, m); ax.set_ylim(0, m)
ax.set_xlabel("MDE (pp, 80% power, alpha=.05)"); ax.set_ylabel("|effect| (pp)")
ax.set_title("Effect size vs minimum detectable effect\n(above the line = resolvable at this n; symlog axes)", loc="left")
ax.legend(fontsize=7.5)
fig.tight_layout(); save(fig, "effect_vs_mde.png")

# ---- seed SD -------------------------------------------------------------------------
fig, axes = plt.subplots(2, len(KS), figsize=(5.0 * len(KS), 6.4), squeeze=False)
for row, arm in enumerate(("matched", "deployed")):
    for ax, K in zip(axes[row], KS):
        for M in MODELS:
            ax.plot(range(len(SIGMAS)), [V(K, sub_of(K), s, M, arm).std(ddof=1) for s in SIGMAS], color=COL[M],
                    marker=MK[M], ms=5, label=NAME[M], markeredgecolor=SURF)
        ax.set_xticks(range(len(SIGMAS))); ax.set_xticklabels(["%g" % s for s in SIGMAS])
        ax.set_xlabel("sigma"); ax.set_ylabel("seed SD (pp)"); ax.set_title("%s, %s arm" % (ktitle(K), arm), loc="left")
axes[0][0].legend(fontsize=7.5)
fig.suptitle("Seed-to-seed spread vs pixel-noise sigma", x=0.01, ha="left", fontsize=11)
fig.tight_layout(); save(fig, "seed_sd_vs_sigma.png")
print("Done.")
