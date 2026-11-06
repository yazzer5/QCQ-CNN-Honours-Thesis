#!/usr/bin/env python3
"""
v14 standard plot set (run AFTER collect_v14.py, from the v14 repo root).
    python3 scripts/plot_v14.py
Writes to figures_v14/ at 200 dpi:
    learning_curves.png   HEADLINE: accuracy vs N per class (log x), M1-M5, 95% CI, per K
    contrasts_vs_n.png    F = M3-M4, T = M3-M5, R = M5-M4, BE = M3-M2 vs N, 95% CI, +/-1 pp band
    bars_k{K}.png         mean accuracy with 95% CI by N and model (bar chart)
    strip_k{K}.png        per-seed accuracy by N and model
    paired_k{K}.png       per-seed paired differences for F, T, R at every N
    effect_vs_mde.png     |effect| vs MDE for every pooled/all contrast
    seed_sd_vs_n.png      seed-to-seed SD vs N per model
"""
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

OUT = Path("figures_v14"); OUT.mkdir(exist_ok=True)
COL = {"M1": "#2a78d6", "M2": "#eb6834", "M3": "#1baf7a", "M4": "#4a3aa7", "M5": "#c2417e"}
CCOL = {"F": "#1baf7a", "T": "#c2417e", "R": "#4a3aa7", "BE": "#eb6834", "Q": "#2a78d6"}
INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False})
NS = [50, 100, 250, 500]
MODELS = ["M1", "M2", "M3", "M4", "M5"]
LABEL = {"F": "F: M3 - M4 (trained quantum front-end vs none)", "T": "T: M3 - M5 (training the front-end)",
         "R": "R: M5 - M4 (frozen random front-end vs none)", "BE": "BE: M3 - M2 (back-end cost)"}


def read(p):
    with open(p) as fh:
        return list(csv.DictReader(fh))


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


runs, con = read("v14_runs.csv"), read("v14_contrasts.csv")
KS = sorted({int(r["K"]) for r in runs})
HEAD = {5: "pooled", 10: "all"}
TAGS = {5: ["A", "B"], 10: ["all"]}
by = defaultdict(dict)
for r in runs:
    by[(int(r["K"]), r["subset"], int(r["N"]), r["model"])][int(r["seed"])] = float(r["test_acc"])
for (K, t, N, m) in list(by):
    if t == "A" and (K, "B", N, m) in by:
        A, B = by[(K, "A", N, m)], by[(K, "B", N, m)]
        by[(K, "pooled", N, m)] = {s: (A[s] + B[s]) / 2 for s in set(A) & set(B)}
C = {(int(r["K"]), r["subset"], r["N"], r["contrast"]): r for r in con}
rng = np.random.default_rng(0)


def ci(v):
    v = np.asarray(v)
    return stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan


def seeds(r):
    return np.array([float(x.split(":")[1]) for x in r["per_seed"].split(";")]) if r and r["per_seed"] else np.array([])


# learning curves ------------------------------------------------------------------
fig, axes = plt.subplots(1, len(KS), figsize=(5.2 * len(KS), 4.2), squeeze=False)
for ax, K in zip(axes[0], KS):
    for i, m in enumerate(MODELS):
        mu = [np.mean(list(by[(K, HEAD[K], N, m)].values())) if by.get((K, HEAD[K], N, m)) else np.nan for N in NS]
        e = [ci(list(by[(K, HEAD[K], N, m)].values())) if by.get((K, HEAD[K], N, m)) else np.nan for N in NS]
        ax.errorbar(np.array(NS) * (1 + (i - 2) * 0.025), mu, yerr=e, marker="o", ms=4, lw=1.4, capsize=3,
                    color=COL[m], label=m)
    ax.set_xscale("log"); ax.set_xticks(NS, [str(n) for n in NS]); ax.minorticks_off()
    ax.set_xlabel("training images per class (log scale)"); ax.set_ylabel("test accuracy (%)")
    ax.set_title(f"K={K} ({HEAD[K]}), width 15, mean and 95% CI over 10 seeds", fontsize=10)
axes[0][0].legend(ncol=5, loc="lower right", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "learning_curves.png", dpi=200); plt.close(fig)

# contrasts vs N -------------------------------------------------------------------
fig, axes = plt.subplots(1, len(KS), figsize=(5.6 * len(KS), 4.2), squeeze=False, sharey=False)
for ax, K in zip(axes[0], KS):
    ax.axhspan(-1, 1, color=GRID, alpha=.6, lw=0, label="±1 pp equivalence margin"); ax.axhline(0, color=MUTED, lw=.8)
    for j, c in enumerate(["F", "T", "R", "BE"]):
        pts = [(N, C.get((K, HEAD[K], str(N), c))) for N in NS]
        pts = [(N, r) for N, r in pts if r and f(r["n"]) >= 2]
        if not pts:
            continue
        x = np.array([N for N, _ in pts]) * (1 + (j - 1.5) * 0.03)
        d = np.array([f(r["diff"]) for _, r in pts]); lo = np.array([f(r["lo"]) for _, r in pts]); hi = np.array([f(r["hi"]) for _, r in pts])
        ax.errorbar(x, d, yerr=[d - lo, hi - d], marker="o", ms=4, capsize=3, lw=1.2, color=CCOL[c], label=LABEL[c])
    ax.set_xscale("log"); ax.set_xticks(NS, [str(n) for n in NS]); ax.minorticks_off()
    ax.set_xlabel("training images per class (log scale)"); ax.set_ylabel("difference (pp), 95% CI")
    ax.set_title(f"K={K} ({HEAD[K]}): paired contrasts vs data size", fontsize=10)
axes[0][0].legend(fontsize=7, loc="best")
fig.tight_layout(); fig.savefig(OUT / "contrasts_vs_n.png", dpi=200); plt.close(fig)

# bars ---------------------------------------------------------------------------------
for K in KS:
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.16
    for i, m in enumerate(MODELS):
        mu = [np.mean(list(by[(K, HEAD[K], N, m)].values())) if by.get((K, HEAD[K], N, m)) else np.nan for N in NS]
        e = [ci(list(by[(K, HEAD[K], N, m)].values())) if by.get((K, HEAD[K], N, m)) else np.nan for N in NS]
        ax.bar(np.arange(len(NS)) + (i - 2) * w, mu, w, yerr=e, capsize=2, color=COL[m], label=m)
    lo = np.nanmin([np.mean(list(v.values())) for k, v in by.items() if k[0] == K and k[1] == HEAD[K]])
    ax.set_ylim(max(0, lo - 8), 100); ax.set_xticks(range(len(NS)), [f"N={n}" for n in NS])
    ax.set_ylabel("test accuracy (%)"); ax.set_title(f"K={K} ({HEAD[K]}): mean accuracy, 95% CI", fontsize=10)
    ax.legend(ncol=5, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / f"bars_k{K}.png", dpi=200); plt.close(fig)

# strip ---------------------------------------------------------------------------------
for K in KS:
    fig, axes = plt.subplots(1, len(TAGS[K]), figsize=(8 * len(TAGS[K]), 4), squeeze=False)
    for ax, t in zip(axes[0], TAGS[K]):
        cols = [(N, m) for N in NS for m in MODELS]
        for i, (N, m) in enumerate(cols):
            v = np.array(list(by.get((K, t, N, m), {}).values()))
            if len(v):
                ax.scatter(i + rng.uniform(-.15, .15, len(v)), v, s=12, color=COL[m], alpha=.8)
                ax.hlines(v.mean(), i - .3, i + .3, color=INK, lw=1.1)
        ax.set_xticks(range(len(cols)), [f"{m}" for N, m in cols], fontsize=7)
        for k, N in enumerate(NS):
            ax.text(k * len(MODELS) + 2, 1.01, f"N={N}", transform=ax.get_xaxis_transform(), ha="center", fontsize=8, color=MUTED)
            if k:
                ax.axvline(k * len(MODELS) - .5, color=GRID, lw=1)
        ax.set_title(f"K={K} subset {t}", fontsize=10, pad=16); ax.set_ylabel("test accuracy (%)")
    fig.tight_layout(); fig.savefig(OUT / f"strip_k{K}.png", dpi=200); plt.close(fig)

# paired --------------------------------------------------------------------------------
for K in KS:
    items = [(N, c) for c in ("F", "T", "R") for N in NS]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.axhspan(-1, 1, color=GRID, alpha=.6, lw=0); ax.axhline(0, color=MUTED, lw=.8)
    for i, (N, c) in enumerate(items):
        r = C.get((K, HEAD[K], str(N), c)); s = seeds(r)
        if not len(s):
            continue
        ax.scatter(i + rng.uniform(-.12, .12, len(s)), s, s=14, color=CCOL[c], alpha=.8)
        if f(r["n"]) >= 2:
            ax.errorbar(i + .3, f(r["diff"]), yerr=[[f(r["diff"]) - f(r["lo"])], [f(r["hi"]) - f(r["diff"])]], fmt="D", ms=4, color=INK, capsize=3)
    ax.set_xticks(range(len(items)), [f"{c}\nN={N}" for N, c in items], fontsize=8)
    ax.set_ylabel("per-seed difference (pp)")
    ax.set_title(f"K={K} ({HEAD[K]}): F = M3-M4, T = M3-M5, R = M5-M4 per seed, with mean and 95% CI", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / f"paired_k{K}.png", dpi=200); plt.close(fig)

# effect vs MDE -----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 5.5)); mx = 0
for r in con:
    if f(r["n"]) < 2 or r["subset"] not in ("pooled", "all"):
        continue
    e, mde = abs(f(r["diff"])), f(r["mde"]); mx = max(mx, e, mde)
    base = r["contrast"].replace("slope(", "").replace("AULC(", "").rstrip(")")
    ax.scatter(mde, e, marker="o" if r["family"] == "primary" else "s", s=28, color=CCOL.get(base, MUTED), alpha=.85)
    if r["family"] == "primary":
        ax.annotate(f"K{r['K']} {r['contrast']} N={r['N']}", (mde, e), fontsize=6, color=MUTED, xytext=(3, 2), textcoords="offset points")
ax.plot([0, mx * 1.05], [0, mx * 1.05], color=INK, lw=.8, ls="--")
ax.set_xlabel("MDE (pp, 80% power)"); ax.set_ylabel("|effect| (pp)")
ax.set_title("Effect vs MDE (circles primary, squares secondary)\nbelow the line: not resolvable at n = 10", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "effect_vs_mde.png", dpi=200); plt.close(fig)

# seed SD vs N --------------------------------------------------------------------------
fig, axes = plt.subplots(1, len(KS), figsize=(5 * len(KS), 3.8), squeeze=False)
for ax, K in zip(axes[0], KS):
    for m in MODELS:
        sd = [np.std(list(by[(K, HEAD[K], N, m)].values()), ddof=1) if len(by.get((K, HEAD[K], N, m), {})) > 1 else np.nan for N in NS]
        ax.plot(NS, sd, marker="o", ms=4, color=COL[m], label=m)
    ax.set_xscale("log"); ax.set_xticks(NS, [str(n) for n in NS]); ax.minorticks_off()
    ax.set_xlabel("training images per class"); ax.set_ylabel("seed SD (pp)"); ax.set_title(f"K={K}: seed-to-seed spread", fontsize=10)
axes[0][0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "seed_sd_vs_n.png", dpi=200); plt.close(fig)
print(f"Wrote {len(list(OUT.glob('*.png')))} figures to {OUT}/")
