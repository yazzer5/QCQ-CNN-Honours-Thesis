#!/usr/bin/env python3
"""
v15 standard plot set (PNG, 200 dpi) -> figures_v15/. Run after collect_v15.py:

    python3 scripts/plot_v15.py

  noise_curves.png      MATCHED arm: accuracy vs alpha, M1-M5, 95% CI       (headline)
  deploy_curves.png     DEPLOYED arm: trained clean, tested at alpha (to 16), 95% CI
  contrasts_vs_alpha.png F, T, R, BE, NC_M3 (+DEP_M3) vs alpha, 95% CI, +/-1 pp band
  recovery.png          REC_m = matched - deployed vs alpha, per model
  bars_k5.png, bars_k10.png       mean accuracy per alpha and model, seed-SD error bars
  strip_k5.png, strip_k10.png     every seed (per subset at K=5)
  paired_k5.png, paired_k10.png   per-seed paired differences F, T, NC_M3 at every alpha
  effect_vs_mde.png     |effect| vs MDE for every matched-arm contrast; primaries marked
  seed_sd_vs_alpha.png  seed SD vs alpha per model
Model colours are fixed per model in every figure (validated categorical palette;
markers and direct labels carry identity too).
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
import collect_v15 as C  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures_v15"
OUT.mkdir(exist_ok=True)

SURF, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e8e7e2", "#c3c2b7"
COL = {"M1": "#2a78d6", "M2": "#eb6834", "M3": "#1baf7a", "M4": "#eda100", "M5": "#e87ba4"}
MK = {"M1": "o", "M2": "s", "M3": "D", "M4": "^", "M5": "v"}
NAME = {"M1": "M1 RandomLayers + QNN", "M2": "M2 Circuit 4 + QNN", "M3": "M3 Circuit 4 + classical",
        "M4": "M4 no front-end", "M5": "M5 Circuit 4 frozen"}
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
                     "legend.frameon": False, "lines.linewidth": 2})

matched, deployed, rows = C.load()
SEEDS = sorted({k[4] for k in matched})
KS, ALPHAS, EVALS, MODELS = C.KS, C.ALPHAS, C.EVALS, C.MODELS


def vec(K, sub, A, M, arm="matched"):
    tags = C.TAGS[K] if sub == "pooled" else [sub]
    out = []
    for S in SEEDS:
        vals = []
        for T in tags:
            if arm == "matched":
                v = matched.get((K, T, 0.0 if M in C.CTRL_MODELS else A, M, S))
            else:
                v = deployed.get((K, T, 0.0, M, S, A))
            if v is None:
                return None
            vals.append(v)
        out.append(np.mean(vals))
    return np.array(out)


def ci(v):
    if v is None or len(v) < 2:
        return np.nan, np.nan
    return v.mean(), stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))


def sub_of(K):
    return "pooled" if K == 5 else "all"


def ktitle(K):
    return "K=5 (subsets A, B pooled)" if K == 5 else "K=10"


def xpos(alist):
    return {a: i for i, a in enumerate(alist)}


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("  " + name)


def curves(arm, alist, fname, title):
    fig, axes = plt.subplots(1, len(KS), figsize=(5.6 * len(KS), 4.2), squeeze=False)
    X = xpos(alist)
    for ax, K in zip(axes[0], KS):
        ends = []
        for M in MODELS:
            mu, hw = zip(*[ci(vec(K, sub_of(K), a, M, arm)) for a in alist])
            xs = [X[a] for a in alist]
            ax.errorbar(xs, mu, yerr=hw, color=COL[M], marker=MK[M], ms=5, capsize=2, lw=2,
                        markeredgecolor=SURF, markeredgewidth=1, label=NAME[M])
            ends.append((mu[-1], M))
        # direct labels at the right end, de-collided
        ends.sort()
        ys = [e[0] for e in ends]
        lo, hi = ax.get_ylim()
        gap = (hi - lo) * 0.045
        for i in range(1, len(ys)):
            ys[i] = max(ys[i], ys[i - 1] + gap)
        for (y0, M), y in zip(ends, ys):
            ax.annotate(M, (len(alist) - 1, y0), xytext=(len(alist) - 0.75, y), color=INK2, fontsize=8,
                        va="center", annotation_clip=False)
        ax.set_ylim(top=min(ax.get_ylim()[1], 101))
        ax.set_xticks(range(len(alist)))
        ax.set_xticklabels(["%g" % a for a in alist])
        ax.set_xlabel("noise scale alpha (1 = ibm_brisbane calibration)")
        ax.set_ylabel("test accuracy (%)")
        ax.set_title(ktitle(K), loc="left")
        ax.axvline(X[1.0], color=BASE, lw=1, ls=":", zorder=0) if 1.0 in X else None
    axes[0][0].legend(loc="lower left", fontsize=7.5)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, fname)


def read_contrasts():
    with open(ROOT / "v15_contrasts.csv") as f:
        rs = list(csv.DictReader(f))
    for r in rs:
        for k in ("K", "n"):
            r[k] = int(r[k])
        for k in ("alpha", "diff", "lo", "hi", "p", "mde"):
            r[k] = float(r[k]) if r[k] not in ("", "nan") else np.nan
    return rs


print("Writing figures_v15/:")
curves("matched", ALPHAS, "noise_curves.png",
       "Trained and tested under front-end noise (matched arm), mean and 95% CI over seeds")
curves("deployed", EVALS, "deploy_curves.png",
       "Trained without noise, tested under front-end noise (deployed arm), mean and 95% CI")

CR = read_contrasts()

# ---- contrasts vs alpha (small multiples) ------------------------------------------
panels = [("F", "F = M3 - M4\nquantum front-end vs none"), ("T", "T = M3 - M5\ntrained vs frozen"),
          ("R", "R = M5 - M4\nfrozen front-end vs none"), ("BE", "BE = M3 - M2\nclassical vs quantum head"),
          ("NC_M3", "NC_M3 = M3(alpha) - M3(0)\nnoise cost to M3")]
fig, axes = plt.subplots(len(KS), len(panels), figsize=(3.2 * len(panels), 3.0 * len(KS)), squeeze=False)
for i, K in enumerate(KS):
    for j, (c, lab) in enumerate(panels):
        ax = axes[i][j]
        for arm, col, alist, cname, ls in (("matched", "#2a78d6", ALPHAS, c, "-"),
                                           ("deployed", "#eb6834", EVALS, "DEP_M3" if c == "NC_M3" else c, "--")):
            pts = [r for r in CR if r["K"] == K and r["subset"] == sub_of(K) and r["contrast"] == cname
                   and r["arm"] == arm and not np.isnan(r["alpha"])]
            pts.sort(key=lambda r: r["alpha"])
            if not pts:
                continue
            X = xpos(EVALS)
            xs = [X[r["alpha"]] + (0.08 if arm == "deployed" else 0) for r in pts]
            ax.errorbar(xs, [r["diff"] for r in pts], yerr=[[r["diff"] - r["lo"] for r in pts], [r["hi"] - r["diff"] for r in pts]],
                        color=col, marker="o", ms=4, capsize=2, lw=1.6, ls=ls,
                        label="matched (trained at alpha)" if arm == "matched" else "deployed (trained at 0)")
        ax.axhspan(-1, 1, color=GRID, alpha=0.6, zorder=0, lw=0)
        ax.axhline(0, color=BASE, lw=1, zorder=1)
        ax.set_xticks(range(len(EVALS)))
        ax.set_xticklabels(["%g" % a for a in EVALS], fontsize=7)
        if i == 0:
            ax.set_title(lab, fontsize=8.5, loc="left")
        if j == 0:
            ax.set_ylabel("%s\ndifference (pp)" % ktitle(K))
        if i == len(KS) - 1:
            ax.set_xlabel("alpha")
axes[0][0].legend(fontsize=7, loc="lower left")
fig.suptitle("Contrasts vs noise scale, 95% CI; grey band = +/-1 pp equivalence margin", x=0.01, ha="left", fontsize=11)
fig.tight_layout()
save(fig, "contrasts_vs_alpha.png")

# ---- recovery ------------------------------------------------------------------------
fig, axes = plt.subplots(1, len(KS), figsize=(5.4 * len(KS), 3.8), squeeze=False)
for ax, K in zip(axes[0], KS):
    X = xpos(ALPHAS)
    for M in C.NOISE_MODELS:
        pts = sorted([r for r in CR if r["K"] == K and r["subset"] == sub_of(K) and r["contrast"] == "REC_" + M],
                     key=lambda r: r["alpha"])
        if pts:
            off = (C.NOISE_MODELS.index(M) - 1.5) * 0.06
            ax.errorbar([X[r["alpha"]] + off for r in pts], [r["diff"] for r in pts],
                        yerr=[[r["diff"] - r["lo"] for r in pts], [r["hi"] - r["diff"] for r in pts]],
                        color=COL[M], marker=MK[M], ms=5, capsize=2, label=NAME[M], markeredgecolor=SURF)
    ax.axhline(0, color=BASE, lw=1)
    ax.set_xticks(range(len(ALPHAS))); ax.set_xticklabels(["%g" % a for a in ALPHAS])
    ax.set_xlabel("alpha"); ax.set_ylabel("matched - deployed (pp)"); ax.set_title(ktitle(K), loc="left")
axes[0][0].legend(fontsize=7.5)
fig.suptitle("What noise-aware training recovers (REC = trained at alpha - trained clean, both tested at alpha)",
             x=0.01, ha="left", fontsize=11)
fig.tight_layout()
save(fig, "recovery.png")

# ---- bars, strips, paired ------------------------------------------------------------
for K in KS:
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    wbar = 0.16
    for k, M in enumerate(MODELS):
        mu = [vec(K, sub_of(K), a, M).mean() for a in ALPHAS]
        sd = [vec(K, sub_of(K), a, M).std(ddof=1) for a in ALPHAS]
        ax.bar(np.arange(len(ALPHAS)) + (k - 2) * wbar, mu, wbar * 0.88, yerr=sd, color=COL[M], label=NAME[M],
               error_kw=dict(ecolor=INK2, lw=0.8, capsize=1.5))
    lo = min(vec(K, sub_of(K), a, M).min() for a in ALPHAS for M in MODELS)
    ax.set_ylim(max(0, lo - 5), 100)
    ax.set_xticks(range(len(ALPHAS))); ax.set_xticklabels(["alpha=%g" % a for a in ALPHAS])
    ax.set_ylabel("test accuracy (%)"); ax.set_title("%s: mean accuracy (matched arm), error bars = seed SD" % ktitle(K), loc="left")
    ax.legend(fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(); save(fig, "bars_k%d.png" % K)

    tags = C.TAGS[K]
    fig, axes = plt.subplots(1, len(ALPHAS), figsize=(2.4 * len(ALPHAS), 3.6), sharey=True, squeeze=False)
    for ax, A in zip(axes[0], ALPHAS):
        for k, M in enumerate(MODELS):
            for t_i, T in enumerate(tags):
                ys = [matched.get((K, T, 0.0 if M in C.CTRL_MODELS else A, M, S)) for S in SEEDS]
                ys = [y for y in ys if y is not None]
                xs = k + (t_i - (len(tags) - 1) / 2) * 0.25 + np.linspace(-0.06, 0.06, len(ys))
                ax.scatter(xs, ys, s=12, color=COL[M], marker=MK[M] if t_i == 0 else "x", linewidths=0.8,
                           edgecolors=SURF if t_i == 0 else None)
        ax.set_xticks(range(len(MODELS))); ax.set_xticklabels(MODELS, fontsize=7.5)
        ax.set_title("alpha=%g" % A, loc="left")
    axes[0][0].set_ylabel("test accuracy (%)")
    fig.suptitle("%s: every seed%s" % (ktitle(K), " (filled = subset A, x = subset B)" if K == 5 else ""),
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(); save(fig, "strip_k%d.png" % K)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), squeeze=False)
    for ax, (lab, getd) in zip(axes[0], (
            ("F = M3 - M4", lambda A: vec(K, sub_of(K), A, "M3") - vec(K, sub_of(K), A, "M4")),
            ("T = M3 - M5", lambda A: vec(K, sub_of(K), A, "M3") - vec(K, sub_of(K), A, "M5")),
            ("NC_M3 = M3(alpha) - M3(0)", lambda A: vec(K, sub_of(K), A, "M3") - vec(K, sub_of(K), 0.0, "M3")))):
        for xi, A in enumerate(ALPHAS):
            d = getd(A)
            ax.scatter(xi + np.linspace(-0.12, 0.12, len(d)), d, s=12, color="#2a78d6", edgecolors=SURF, linewidths=0.6)
            ax.hlines(d.mean(), xi - 0.22, xi + 0.22, color=INK, lw=1.6)
        ax.axhspan(-1, 1, color=GRID, alpha=0.6, zorder=0, lw=0); ax.axhline(0, color=BASE, lw=1)
        ax.set_xticks(range(len(ALPHAS))); ax.set_xticklabels(["%g" % a for a in ALPHAS])
        ax.set_xlabel("alpha"); ax.set_title(lab, loc="left")
    axes[0][0].set_ylabel("per-seed difference (pp)")
    fig.suptitle("%s: per-seed paired differences (bar = mean; grey = +/-1 pp)" % ktitle(K), x=0.01, ha="left", fontsize=11)
    fig.tight_layout(); save(fig, "paired_k%d.png" % K)

# ---- effect vs MDE ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(5.4, 4.6))
pts = [r for r in CR if r["arm"] == "matched" and r["subset"] in ("pooled", "all") and np.isfinite(r["mde"])]
sec = [r for r in pts if r["role"] != "PRIMARY"]
pri = [r for r in pts if r["role"] == "PRIMARY"]
ax.scatter([r["mde"] for r in sec], [abs(r["diff"]) for r in sec], s=14, color=MUTED, label="secondary")
ax.scatter([r["mde"] for r in pri], [abs(r["diff"]) for r in pri], s=40, color="#2a78d6", edgecolors=SURF,
           label="primary", zorder=3)
for r in pri:
    ax.annotate("%s%s K=%d" % (r["contrast"], "" if np.isnan(r["alpha"]) else "@%g" % r["alpha"], r["K"]),
                (r["mde"], abs(r["diff"])), xytext=(4, 3), textcoords="offset points", fontsize=7, color=INK2)
m = max([r["mde"] for r in pts] + [abs(r["diff"]) for r in pts]) * 1.08
ax.plot([0, m], [0, m], color=BASE, lw=1, ls="--")
ax.set_xlim(0, m); ax.set_ylim(0, m)
ax.set_xlabel("MDE (pp, 80% power, alpha=.05)"); ax.set_ylabel("|effect| (pp)")
ax.set_title("Effect size vs minimum detectable effect\n(above the line = resolvable at this n)", loc="left")
ax.legend(fontsize=8)
fig.tight_layout(); save(fig, "effect_vs_mde.png")

# ---- seed SD -------------------------------------------------------------------------
fig, axes = plt.subplots(1, len(KS), figsize=(5.0 * len(KS), 3.6), squeeze=False)
for ax, K in zip(axes[0], KS):
    for M in MODELS:
        ax.plot(range(len(ALPHAS)), [vec(K, sub_of(K), a, M).std(ddof=1) for a in ALPHAS], color=COL[M],
                marker=MK[M], ms=5, label=NAME[M], markeredgecolor=SURF)
    ax.set_xticks(range(len(ALPHAS))); ax.set_xticklabels(["%g" % a for a in ALPHAS])
    ax.set_xlabel("alpha"); ax.set_ylabel("seed SD (pp)"); ax.set_title(ktitle(K), loc="left")
axes[0][0].legend(fontsize=7.5)
fig.suptitle("Seed-to-seed spread vs noise scale (matched arm)", x=0.01, ha="left", fontsize=11)
fig.tight_layout(); save(fig, "seed_sd_vs_alpha.png")
print("Done.")
