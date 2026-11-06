#!/usr/bin/env python3
"""
v16b-ext standard plot set (K=10, N=50). Run from the repo root after collect_v16bx.py -> figures_v16bx/*.png (200 dpi)

  bars.png           mean accuracy +- seed SD: M2, M4, M4-trim, BFR, BFR-ZZ
  strip.png          every seed, lines joining the same seed across models (pairing)
  paired.png         PRIMARY: per-seed C1, C2, C3 (and D4 for reference), mean, 95% CI, E1/E2 thresholds
  effect_vs_mde.png  |effect| vs MDE for the 3 primary contrasts
  decomp.png         fit part vs gap part of each contrast (secondary)
  gate_scale.png     |g| and s at the best checkpoint, BFR vs BFR-ZZ
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
OUT = "figures_v16bx"
os.makedirs(OUT, exist_ok=True)
COL = {"M2": "#2a78d6", "M4": "#1baf7a", "M4trim": "#7fd3b5", "BFR": "#eb6834", "BFRZZ": "#9b4fd1"}
LAB = {"M2": "M2 (ZZ QNN head)", "M4": "M4", "M4trim": "M4-trim", "BFR": "BFR", "BFRZZ": "BFR-ZZ"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 200})
rd = lambda f: list(csv.DictReader(open(f)))  # noqa: E731
runs, con, dec, gs = rd("v16bx_runs.csv"), rd("v16bx_contrasts.csv"), rd("v16bx_decomp.csv"), rd("v16bx_gs.csv")
MODELS = ["M2", "M4", "M4trim", "BFR", "BFRZZ"]
SEEDS = sorted({int(r["seed"]) for r in runs})
A = {(r["model"], int(r["seed"])): float(r["test_acc"]) for r in runs}
vec = lambda M: np.array([A[(M, S)] for S in SEEDS])  # noqa: E731
rng = np.random.default_rng(0)

fig, ax = plt.subplots(figsize=(5.6, 3.2))
for i, M in enumerate(MODELS):
    ax.bar(i, vec(M).mean(), 0.7, color=COL[M], yerr=vec(M).std(ddof=1), error_kw=dict(ecolor=INK2, lw=0.8, capsize=3))
ax.set_xticks(range(len(MODELS))); ax.set_xticklabels([LAB[M] for M in MODELS])
ax.set_ylim(max(0, min(vec(M).min() for M in MODELS) - 8), 80); ax.set_ylabel("test accuracy (%)")
ax.set_title("K = 10, N = 50/class: mean +- seed SD (n = %d)" % len(SEEDS), color=INK)
fig.tight_layout(); fig.savefig(f"{OUT}/bars.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(5.6, 3.4))
for S in SEEDS:
    ax.plot(range(len(MODELS)), [A[(M, S)] for M in MODELS], color=GRID, lw=0.8, zorder=1)
for i, M in enumerate(MODELS):
    ax.scatter(np.full(len(SEEDS), i), vec(M), s=16, color=COL[M], edgecolor="white", linewidth=0.5, zorder=2)
    ax.plot([i - 0.25, i + 0.25], [vec(M).mean()] * 2, color=INK, lw=1.3, zorder=3)
ax.set_xticks(range(len(MODELS))); ax.set_xticklabels([LAB[M] for M in MODELS]); ax.set_ylabel("test accuracy (%), every seed")
ax.set_title("Per seed (grey lines join the same seed)", color=INK)
fig.tight_layout(); fig.savefig(f"{OUT}/strip.png"); plt.close(fig)

items = [("C1 = BFR - M4trim", "BFR", "M4trim", None), ("C2 = M4trim - M4", "M4trim", "M4", -2.3),
         ("C3 = BFRZZ - BFR", "BFRZZ", "BFR", -5.0), ("D4 = BFR - M4 (reference)", "BFR", "M4", None)]
cmap = {c["contrast"]: c for c in con}
fig, ax = plt.subplots(figsize=(6.4, 3.4))
for pos, (name, a, b, thr) in enumerate(items):
    d = vec(a) - vec(b); c = cmap[name]
    m, lo, hi = float(c["mean"]), float(c["lo"]), float(c["hi"])
    ax.scatter(pos + rng.uniform(-0.12, 0.12, len(d)), d, s=14, color=COL[a], edgecolor="white", linewidth=0.5)
    ax.errorbar(pos + 0.3, m, yerr=[[m - lo], [hi - m]], fmt="o", color=INK, ms=4, capsize=3, lw=1.2)
    if thr is not None:
        ax.plot([pos - 0.3, pos + 0.45], [thr] * 2, color=INK2, lw=1, ls="--")
        ax.annotate(("E1 " if thr == -2.3 else "E2 ") + f"{thr:g}", (pos + 0.45, thr), xytext=(2, -9), textcoords="offset points",
                    fontsize=6.5, color=INK2)
    if c.get("p_holm") not in (None, "") and float(c["p_holm"]) < 0.05:
        ax.annotate("Holm p<.05", (pos + 0.3, hi), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=6.5, color=INK2)
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xticks(range(len(items))); ax.set_xticklabels([n.split(" (")[0].replace(" = ", "\n") for n, *_ in items], fontsize=7.5)
ax.set_ylabel("difference (pp), per seed"); ax.set_title("Primary contrasts at K = 10, N = 50 (mean, 95% CI)", color=INK)
fig.tight_layout(); fig.savefig(f"{OUT}/paired.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(4.4, 3.4))
prim = [c for c in con if not c["contrast"].startswith("D4")]
for c, mk in zip(prim, ("o", "s", "^")):
    ax.scatter(float(c["mde"]), abs(float(c["mean"])), marker=mk, s=36, color=COL[c["a"]], edgecolor="white", label=c["contrast"])
lim = max(1, max(max(float(c["mde"]), abs(float(c["mean"]))) for c in prim)) * 1.15
ax.plot([0, lim], [0, lim], color=INK2, lw=0.8, ls="--"); ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("MDE (pp, 80% power)"); ax.set_ylabel("|effect| (pp)"); ax.legend(fontsize=6.5)
ax.set_title("Above the line: resolvable at n = 10", color=INK)
fig.tight_layout(); fig.savefig(f"{OUT}/effect_vs_mde.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(6.0, 3.2))
x = np.arange(len(dec))
for off, key, col, lab in ((-0.18, "fit", "#2a78d6", "fit part (train acc @ best)"), (0.18, "gap", "#eb6834", "gap part (-delta gap)")):
    m = np.array([float(d[key]) for d in dec])
    e = np.array([[m[i] - float(d[key + "_lo"]), float(d[key + "_hi"]) - m[i]] for i, d in enumerate(dec)]).T
    ax.bar(x + off, m, 0.34, color=col, label=lab, yerr=e, error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
ax.axhline(0, color=INK2, lw=0.8); ax.set_xticks(x); ax.set_xticklabels([d["contrast"].replace(" = ", "\n") for d in dec], fontsize=7)
ax.set_ylabel("pp"); ax.legend(fontsize=7); ax.set_title("Where each contrast comes from (secondary)", color=INK)
fig.tight_layout(); fig.savefig(f"{OUT}/decomp.png"); plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(6.0, 2.8))
for i, M in enumerate(("BFR", "BFRZZ")):
    g = [x for x in gs if x["model"] == M]
    axs[0].scatter(i + rng.uniform(-0.12, 0.12, len(g)), [abs(float(x["g"])) for x in g], s=16, color=COL[M], edgecolor="white")
    axs[1].scatter(i + rng.uniform(-0.12, 0.12, len(g)), [float(x["s"]) for x in g], s=16, color=COL[M], edgecolor="white")
axs[0].set_ylabel("|g| at best checkpoint"); axs[1].set_ylabel("s (init pi/2, dashed)"); axs[1].axhline(np.pi / 2, color=INK2, ls="--", lw=0.8)
for ax in axs:
    ax.set_xticks([0, 1]); ax.set_xticklabels(["BFR", "BFR-ZZ"])
fig.tight_layout(); fig.savefig(f"{OUT}/gate_scale.png"); plt.close(fig)
print("Wrote " + ", ".join(sorted(os.listdir(OUT))) + f" to {OUT}/")
