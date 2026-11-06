#!/usr/bin/env python3
"""
v12 standard plot set (run AFTER collect_v12.py, from the repo root).
    python scripts/plot_v12.py
Writes to figures_v12/ at 200 dpi:
    acc_vs_k.png           headline: mean accuracy vs number of classes, per model
    contrasts_vs_k.png     each contrast vs K, 95% CI, with +/- MDE band
    effect_vs_mde.png      |effect| vs MDE for every (K, contrast) -- detectable or not
    strip_k{K}.png         per-seed accuracy, per model, per subset
    paired_k{K}.png        per-seed paired differences for each contrast
"""
import csv
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("figures_v12"); OUT.mkdir(exist_ok=True)
MODELS = ["M1", "M2", "M3", "M4"]
COL = {"M1": "#2a78d6", "M2": "#eb6834", "M3": "#1baf7a", "M4": "#4a3aa7"}
MK = {"M1": "o", "M2": "s", "M3": "^", "M4": "D"}
LABEL = {"M1": "M1 fixed Q-front + Q-head", "M2": "M2 trained Q-front + Q-head",
         "M3": "M3 trained Q-front + classical head", "M4": "M4 classical CNN"}
CONS = ["M2-M1", "M3-M2", "M3-M4", "M4-M2"]
CCOL = dict(zip(CONS, ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]))
CTITLE = {"M2-M1": "M2 - M1: training the front-end",
          "M3-M2": "M3 - M2: dropping the quantum back-end",
          "M3-M4": "M3 - M4: quantum vs classical front-end",
          "M4-M2": "M4 - M2: classical CNN vs ours"}
INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False})
HEAD = ("pooled", "all")   # the one row per K used for headline plots


def read(p):
    with open(p) as fh:
        return list(csv.DictReader(fh))


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


runs, summ, con = read("v12_runs.csv"), read("v12_summary.csv"), read("v12_contrasts.csv")
Ks = sorted({int(r["K"]) for r in summ})
rng = np.random.default_rng(0)

# 1. accuracy vs K ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.8))
for m in MODELS:
    pts = sorted((int(s["K"]), f(s["mean"]), f(s["sd"]), int(s["n"])) for s in summ
                 if s["model"] == m and s["subset"] in HEAD)
    if not pts:
        continue
    k, mu, sd, n = map(np.array, zip(*pts))
    ci = 1.96 * np.nan_to_num(sd) / np.sqrt(np.maximum(n, 1))
    ax.errorbar(k, mu, yerr=ci, color=COL[m], marker=MK[m], ms=8, lw=2, capsize=3,
                label=LABEL[m], markeredgecolor="white", markeredgewidth=1.5)
    for s in summ:   # individual subsets, faint
        if s["model"] == m and s["subset"] in ("A", "B"):
            ax.plot(int(s["K"]) + (-0.1 if s["subset"] == "A" else 0.1), f(s["mean"]),
                    MK[m], color=COL[m], alpha=0.35, ms=5)
ax.plot(Ks, [100 / k for k in Ks], ls=":", color=MUTED, lw=1.5, label="chance (100/K)")
ax.set_xticks(Ks)
ax.set_xlabel("Number of MNIST classes (K)")
ax.set_ylabel("Test accuracy (%)")
ax.set_title("Accuracy vs number of classes  (mean, 95% CI; faint = each subset)",
             color=INK, loc="left", fontsize=10)
ax.legend(fontsize=8, loc="lower left")
fig.tight_layout(); fig.savefig(OUT / "acc_vs_k.png", dpi=200); plt.close(fig)

# 2. contrasts vs K with MDE band -----------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
for ax, c in zip(axes.flat, CONS):
    pts = sorted((int(r["K"]), f(r["diff"]), f(r["lo"]), f(r["hi"]), f(r["mde"]))
                 for r in con if r["contrast"] == c and r["subset"] in HEAD and int(r["n"]) > 1)
    ax.axhline(0, color=MUTED, lw=1)
    if pts:
        k, d, lo, hi, mde = map(np.array, zip(*pts))
        ax.fill_between(k, -mde, mde, color=GRID, lw=0, label="within +/- MDE (can't detect)")
        ax.errorbar(k, d, yerr=[d - lo, hi - d], color=CCOL[c], marker="o", ms=8, lw=2,
                    capsize=3, markeredgecolor="white", markeredgewidth=1.5,
                    label="mean difference, 95% CI")
    ax.set_title(CTITLE[c], fontsize=10, color=INK, loc="left")
    ax.set_ylabel("Difference (pp)")
    ax.set_xticks(Ks)
for ax in axes[1]:
    ax.set_xlabel("Number of classes (K)")
axes[0, 0].legend(fontsize=8, loc="best")
fig.tight_layout(); fig.savefig(OUT / "contrasts_vs_k.png", dpi=200); plt.close(fig)

# 3. effect vs MDE ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 5))
mx = 1.0
for c in CONS:
    pts = [(abs(f(r["diff"])), f(r["mde"]), int(r["K"])) for r in con
           if r["contrast"] == c and r["subset"] in HEAD and np.isfinite(f(r["mde"]))]
    if not pts:
        continue
    e, m, k = zip(*pts)
    mx = max(mx, max(e), max(m))
    ax.scatter(m, e, s=64, color=CCOL[c], edgecolor="white", lw=1.5, label=c, zorder=3)
    for ei, mi, ki in pts:
        ax.annotate(f"K={ki}", (mi, ei), xytext=(5, 3), textcoords="offset points",
                    fontsize=7, color=MUTED)
ax.plot([0, mx * 1.1], [0, mx * 1.1], ls="--", color=MUTED, lw=1)
ax.text(mx * 0.55, mx * 0.3, "below the line: effect is smaller\nthan this design can detect",
        fontsize=8, color=MUTED)
ax.set_xlabel("MDE (pp, 80% power, alpha = 0.05)")
ax.set_ylabel("|observed difference| (pp)")
ax.set_title("Effect size vs minimum detectable effect", color=INK, loc="left", fontsize=10)
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "effect_vs_mde.png", dpi=200); plt.close(fig)

# 4 & 5. per-K strip and paired-difference plots ---------------------------
by = defaultdict(list)
for r in runs:
    by[(int(r["K"]), r["subset"], r["model"])].append(f(r["test_acc"]))
for k in Ks:
    tags = sorted({t for (kk, t, _) in by if kk == k})
    fig, axes = plt.subplots(1, len(tags), figsize=(4.2 * len(tags), 4), sharey=True, squeeze=False)
    for ax, t in zip(axes[0], tags):
        for i, m in enumerate(MODELS):
            v = np.array(by.get((k, t, m), []))
            if not len(v):
                continue
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=36, color=COL[m],
                       edgecolor="white", lw=1, zorder=3)
            ax.hlines(v.mean(), i - 0.25, i + 0.25, color=INK, lw=2, zorder=4)
        ax.set_xticks(range(4)); ax.set_xticklabels(MODELS)
        ax.set_title(f"K={k}, subset {t}", color=INK, loc="left", fontsize=10)
    axes[0][0].set_ylabel("Test accuracy (%)")
    fig.suptitle("Per-seed test accuracy (black bar = mean)", x=0.02, ha="left", color=INK, fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / f"strip_k{k}.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4))
    for i, c in enumerate(CONS):
        for r in con:
            if int(r["K"]) == k and r["contrast"] == c and r["subset"] in HEAD and r["per_seed"]:
                d = np.array([float(x) for x in r["per_seed"].split(";")])
                ax.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, s=36, color=CCOL[c],
                           edgecolor="white", lw=1, zorder=3)
                if int(r["n"]) > 1:
                    ax.errorbar(i + 0.32, f(r["diff"]),
                                yerr=[[f(r["diff"]) - f(r["lo"])], [f(r["hi"]) - f(r["diff"])]],
                                color=INK, marker="o", ms=6, capsize=3, lw=1.5)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(range(4)); ax.set_xticklabels(CONS)
    ax.set_ylabel("Paired difference (pp)")
    ax.set_title(f"K={k}: per-seed paired differences, mean + 95% CI", color=INK, loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / f"paired_k{k}.png", dpi=200); plt.close(fig)

print("Wrote figures_v12/:", ", ".join(sorted(p.name for p in OUT.glob("*.png"))))
