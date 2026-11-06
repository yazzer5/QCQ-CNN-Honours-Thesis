#!/usr/bin/env python3
"""
v16b standard plot set. Run from the repo root after collect_v16b.py:
    python3 scripts/plot_v16b.py            -> figures_v16b/*.png  (200 dpi)

  bars.png            mean accuracy +- seed SD: M2 (w15) / M4 / BFR, per K and N
  strip.png           every seed, per K / subset / N
  paired_D4.png       PRIMARY: per-seed BFR - M4 with mean, 95% CI and the +-1 pp TOST band (P2)
  paired_D2.png       PRIMARY: per-seed BFR - M2 with mean, 95% CI and the P1 thresholds
  effect_vs_mde.png   |effect| vs MDE for every pooled contrast
  seed_sd.png         seed SD per model and cell, with the P5 line 1.5 x SD(M4)
  gate_scale.png      learned g and s at the best checkpoint, every run (secondary)
Colours: fixed per model (M2 blue, M4 aqua, BFR orange).
"""
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
OUT = "figures_v16b"
os.makedirs(OUT, exist_ok=True)
COL = {"M2": "#2a78d6", "M4": "#1baf7a", "BFR": "#eb6834"}
LAB = {"M2": "M2 (ZZ QNN head, w15)", "M4": "M4 (classical head)", "BFR": "BFR (bounded, gated quantum head)"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "savefig.dpi": 200})


def rd(f):
    return list(csv.DictReader(open(f)))


runs, con, gs = rd("v16b_runs.csv"), rd("v16b_contrasts.csv"), rd("v16b_gs.csv")
NS = sorted({int(r["N"]) for r in runs})
KS = sorted({int(r["K"]) for r in runs})
MODELS = ["M2", "M4", "BFR"]
KLAB = {5: "K = 5 (pooled A, B)", 10: "K = 10"}
P1_THR = {(10, 500): 1.2, (5, 50): 6.0, (10, 50): 6.0}
acc = {}
for r in runs:
    acc[(r["model"], int(r["K"]), r["subset"], int(r["N"]), int(r["seed"]))] = float(r["test_acc"])
SEEDS = sorted({int(r["seed"]) for r in runs})
TAGS = {5: ["A", "B"], 10: ["all"]}
rng = np.random.default_rng(0)


def vec(M, K, N, TAG=None):
    tags = [TAG] if TAG else TAGS[K]
    return np.array([np.mean([acc[(M, K, T, N, S)] for T in tags]) for S in SEEDS])


def prim(name):
    return {(int(c["K"]), int(c["N"])): c for c in con if c["contrast"] == name and c["family"] == "primary"}


# 1. bars
fig, axs = plt.subplots(1, len(KS), figsize=(7.2, 3.0))
for ax, K in zip(np.atleast_1d(axs), KS):
    w = 0.26
    for i, M in enumerate(MODELS):
        x = np.arange(len(NS)) + (i - 1) * (w + 0.02)
        ax.bar(x, [vec(M, K, N).mean() for N in NS], w, color=COL[M], label=LAB[M],
               yerr=[vec(M, K, N).std(ddof=1) for N in NS], error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
    ax.set_xticks(range(len(NS))); ax.set_xticklabels([f"N = {n}/class" for n in NS])
    lo = min(vec(M, K, N).min() for M in MODELS for N in NS)
    ax.set_ylim(max(0, lo - 8), 100); ax.set_title(KLAB[K], color=INK); ax.set_ylabel("test accuracy (%)")
np.atleast_1d(axs)[0].legend(loc="lower right", fontsize=6.5)
fig.tight_layout(); fig.savefig(f"{OUT}/bars.png"); plt.close(fig)

# 2. strip
cells = [(K, T) for K in KS for T in TAGS[K]]
fig, axs = plt.subplots(1, len(cells), figsize=(9, 3.0))
for ax, (K, T) in zip(axs, cells):
    for j, N in enumerate(NS):
        for i, M in enumerate(MODELS):
            v = vec(M, K, N, T)
            x = j * 4 + i + rng.uniform(-0.15, 0.15, len(v))
            ax.scatter(x, v, s=14, color=COL[M], edgecolor="white", linewidth=0.5, label=LAB[M] if j == 0 else None)
            ax.plot([j * 4 + i - 0.3, j * 4 + i + 0.3], [v.mean()] * 2, color=INK, lw=1.2)
    ax.set_xticks([1, 5]); ax.set_xticklabels([f"N={n}" for n in NS]); ax.set_title(f"K={K} {T}", color=INK)
axs[0].set_ylabel("test accuracy (%), every seed"); axs[0].legend(fontsize=6, loc="lower right")
fig.tight_layout(); fig.savefig(f"{OUT}/strip.png"); plt.close(fig)


# 3-4. paired primaries
def paired_plot(name, other, fname, title, band=None, thr=None):
    pc = prim(name)
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ticks, tl = [], []
    for pos, (K, N) in enumerate([(K, N) for K in KS for N in NS]):
        d = vec("BFR", K, N) - vec(other, K, N)
        c = pc[(K, N)]
        m, lo, hi = float(c["mean"]), float(c["lo"]), float(c["hi"])
        ax.scatter(pos + rng.uniform(-0.12, 0.12, len(d)), d, s=14, color=COL["BFR"], edgecolor="white", linewidth=0.5)
        ax.errorbar(pos + 0.32, m, yerr=[[m - lo], [hi - m]], fmt="o", color=INK, ms=4, capsize=3, lw=1.2)
        if thr and (K, N) in thr:
            ax.plot([pos - 0.3, pos + 0.45], [thr[(K, N)]] * 2, color=INK2, lw=1, ls="--")
            ax.annotate(f"P1 {thr[(K, N)]:+g}", (pos + 0.45, thr[(K, N)]), xytext=(2, 2), textcoords="offset points",
                        fontsize=6.5, color=INK2)
        ticks.append(pos); tl.append(f"K={K}\nN={N}")
    if band:
        ax.axhspan(-band, band, color=GRID, alpha=0.7, lw=0, label=f"TOST margin +-{band:g} pp")
        ax.legend(loc="upper left", fontsize=7)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(ticks); ax.set_xticklabels(tl)
    ax.set_ylabel(f"BFR - {other} (pp), per seed"); ax.set_title(title, color=INK)
    fig.tight_layout(); fig.savefig(f"{OUT}/{fname}"); plt.close(fig)


paired_plot("D4=BFR-M4", "M4", "paired_D4.png", "Primary: BFR vs M4 (mean, 95% CI)", band=1.0)
paired_plot("D2=BFR-M2", "M2", "paired_D2.png", "Primary: BFR vs M2-w15 (mean, 95% CI)", thr=P1_THR)

# 5. effect vs MDE
fig, ax = plt.subplots(figsize=(4.6, 3.6))
for name, mk, c in (("D4=BFR-M4", "o", COL["M4"]), ("D2=BFR-M2", "s", COL["M2"])):
    pts = [(float(r["mde"]), abs(float(r["mean"]))) for r in con if r["contrast"] == name and r["family"] == "primary"]
    ax.scatter(*zip(*pts), marker=mk, color=c, s=30, edgecolor="white", linewidth=0.5, label=name)
lim = max(1, max(max(float(r["mde"]), abs(float(r["mean"]))) for r in con if r["family"] == "primary")) * 1.1
ax.plot([0, lim], [0, lim], color=INK2, lw=0.8, ls="--"); ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("MDE (pp, 80% power)"); ax.set_ylabel("|effect| (pp)"); ax.legend(fontsize=7)
ax.set_title("Above the line: resolvable at n = 10", color=INK)
fig.tight_layout(); fig.savefig(f"{OUT}/effect_vs_mde.png"); plt.close(fig)

# 6. seed SD (P5)
fig, ax = plt.subplots(figsize=(6.0, 3.0))
cellsKN = [(K, N) for K in KS for N in NS]
x = np.arange(len(cellsKN))
for i, M in enumerate(MODELS):
    ax.bar(x + (i - 1) * 0.27, [vec(M, K, N).std(ddof=1) for K, N in cellsKN], 0.25, color=COL[M], label=LAB[M])
for xi, (K, N) in zip(x, cellsKN):
    lim5 = 1.5 * vec("M4", K, N).std(ddof=1)
    ax.plot([xi - 0.4, xi + 0.4], [lim5] * 2, color=INK, lw=1, ls="--")
ax.plot([], [], color=INK, lw=1, ls="--", label="P5 limit: 1.5 x SD(M4)")
ax.set_xticks(x); ax.set_xticklabels([f"K={K} N={N}" for K, N in cellsKN]); ax.set_ylabel("seed SD (pp)")
ax.set_title("Seed stability (P5)", color=INK); ax.legend(fontsize=6.5)
fig.tight_layout(); fig.savefig(f"{OUT}/seed_sd.png"); plt.close(fig)

# 7. gate and scale
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0))
grp = defaultdict(list)
for r in gs:
    grp[(int(r["K"]), int(r["N"]))].append(r)
for pos, (K, N) in enumerate(cellsKN):
    rs = grp[(K, N)]
    for ax, key in ((axs[0], "g_best"), (axs[1], "s_best")):
        v = np.array([float(r[key]) for r in rs])
        ax.scatter(pos + rng.uniform(-0.15, 0.15, len(v)), v, s=14, color=COL["BFR"], edgecolor="white", linewidth=0.5)
axs[0].axhline(0, color=INK2, lw=0.8); axs[0].set_ylabel("gate g (init 0)")
axs[1].axhline(np.pi / 2, color=INK2, lw=0.8, ls="--"); axs[1].set_ylabel("scale s (init pi/2, dashed)")
for ax in axs:
    ax.set_xticks(range(len(cellsKN))); ax.set_xticklabels([f"K{K}\nN{N}" for K, N in cellsKN], fontsize=7)
fig.suptitle("BFR: learned gate and encoding scale at the best checkpoint (every run)", color=INK, fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/gate_scale.png"); plt.close(fig)
print("Wrote " + ", ".join(sorted(os.listdir(OUT))) + f" to {OUT}/")
