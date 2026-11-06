#!/usr/bin/env python3
"""
v16a standard plot set. Run from the repo root after collect_v16a.py:
    python3 scripts/plot_v16a.py            -> figures_v16a/*.png

  bars.png            mean accuracy +- seed SD, M2(w15) / M2c / M3, per K and N
  strip.png           every seed, per K / subset / N
  paired_D.png        PRIMARY: per-seed D = M2c - M2 with mean, 95% CI and the +-1 pp TOST band
  backend_cost.png    BE = M3 - M2 vs BEc = M3 - M2c across N, 95% CI (P3)
  agreement.png       per-image error agreement (P2): same-seed M2c-M2 vs cross-seed M2-M2
  effect_vs_mde.png   |effect| vs MDE for every contrast
Colours: validated categorical slots 1-3 (blue, orange, aqua), fixed per model.
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
OUT = "figures_v16a"
os.makedirs(OUT, exist_ok=True)
COL = {"M2": "#2a78d6", "M2c": "#eb6834", "M3": "#1baf7a"}
LAB = {"M2": "M2 (QNN head, w15)", "M2c": "M2c (closed-form head)", "M3": "M3 (classical head)"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "figure.dpi": 150})


def rd(f):
    return list(csv.DictReader(open(f)))


runs, summ, con, ag = rd("v16a_runs.csv"), rd("v16a_summary.csv"), rd("v16a_contrasts.csv"), rd("v16a_agreement.csv")
NS = sorted({int(r["N"]) for r in runs})
KS = sorted({int(r["K"]) for r in runs})
MODELS = ["M2", "M2c", "M3"]
KLAB = {5: "K = 5 (pooled A, B)", 10: "K = 10"}

# per-seed pooled accuracy
acc = defaultdict(list)
for r in runs:
    acc[(r["model"], int(r["K"]), r["subset"], int(r["N"]), int(r["seed"]))] = float(r["test_acc"])
SEEDS = sorted({int(r["seed"]) for r in runs})
TAGS = {5: ["A", "B"], 10: ["all"]}


def vec(M, K, N, TAG=None):
    tags = [TAG] if TAG else TAGS[K]
    return np.array([np.mean([acc[(M, K, T, N, S)] for T in tags]) for S in SEEDS])


# 1. bars
fig, axs = plt.subplots(1, len(KS), figsize=(7.2, 3.0), sharey=False)
for ax, K in zip(np.atleast_1d(axs), KS):
    w = 0.26
    for i, M in enumerate(MODELS):
        means = [vec(M, K, N).mean() for N in NS]
        sds = [vec(M, K, N).std(ddof=1) for N in NS]
        x = np.arange(len(NS)) + (i - 1) * (w + 0.02)
        ax.bar(x, means, w, color=COL[M], label=LAB[M], yerr=sds, error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
    ax.set_xticks(range(len(NS))); ax.set_xticklabels([f"N = {n}/class" for n in NS])
    lo = min(vec(M, K, N).min() for M in MODELS for N in NS)
    ax.set_ylim(max(0, lo - 8), 100); ax.set_title(KLAB[K], color=INK); ax.set_ylabel("test accuracy (%)")
np.atleast_1d(axs)[0].legend(loc="lower right", fontsize=7)
fig.tight_layout(); fig.savefig(f"{OUT}/bars.png"); plt.close(fig)

# 2. strip
cells = [(K, T) for K in KS for T in TAGS[K]]
fig, axs = plt.subplots(1, len(cells), figsize=(9, 3.0))
rng = np.random.default_rng(0)
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

# 3. paired D (primary)
prim = {(int(c["K"]), int(c["N"])): c for c in con if c["contrast"] == "D" and c["family"] == "primary"}
fig, ax = plt.subplots(figsize=(6.0, 3.2))
pos = 0
ticks, tl = [], []
for K in KS:
    for N in NS:
        d = vec("M2c", K, N) - vec("M2", K, N)
        c = prim[(K, N)]
        ax.scatter(pos + rng.uniform(-0.12, 0.12, len(d)), d, s=14, color=COL["M2c"], edgecolor="white", linewidth=0.5)
        ax.errorbar(pos + 0.32, float(c["mean"]), yerr=[[float(c["mean"]) - float(c["lo"])], [float(c["hi"]) - float(c["mean"])]],
                    fmt="o", color=INK, ms=4, capsize=3, lw=1.2)
        eq = c.get("p_tost_holm") not in (None, "") and float(c["p_tost_holm"]) < 0.05
        ax.annotate("equiv." if eq else "", (pos + 0.32, float(c["hi"])), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7, color=INK2)
        ticks.append(pos); tl.append(f"K={K}\nN={N}"); pos += 1
ax.axhspan(-1, 1, color="#e4e3df", alpha=0.6, lw=0, label="TOST margin +-1 pp")
ax.axhline(0, color=INK2, lw=0.8)
ax.set_xticks(ticks); ax.set_xticklabels(tl)
ax.set_ylabel("M2c - M2 (pp), per seed"); ax.set_title("Primary: closed-form head vs QNN head (mean, 95% CI)", color=INK)
ax.legend(loc="upper left", fontsize=7)
fig.tight_layout(); fig.savefig(f"{OUT}/paired_D.png"); plt.close(fig)

# 4. back-end cost
fig, axs = plt.subplots(1, len(KS), figsize=(7.2, 3.0), sharey=True)
for ax, K in zip(np.atleast_1d(axs), KS):
    for name, M, off in (("BE=M3-M2", "M2", -0.04), ("BEc=M3-M2c", "M2c", 0.04)):
        rows = sorted((c for c in con if c["contrast"] == name and int(c["K"]) == K and c["subset"] in ("pooled", "all")),
                      key=lambda c: int(c["N"]))
        x = np.log2([int(c["N"]) for c in rows]) + off
        m = np.array([float(c["mean"]) for c in rows])
        e = np.array([[m[i] - float(c["lo"]), float(c["hi"]) - m[i]] for i, c in enumerate(rows)]).T
        ax.errorbar(x, m, yerr=e, color=COL[M], marker="o", ms=5, lw=2, capsize=3, label=name.replace("=", " = "))
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(np.log2(NS)); ax.set_xticklabels([str(n) for n in NS]); ax.set_xlabel("N per class")
    ax.set_title(KLAB[K], color=INK)
np.atleast_1d(axs)[0].set_ylabel("cost against M3 (pp)"); np.atleast_1d(axs)[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f"{OUT}/backend_cost.png"); plt.close(fig)

# 5. agreement
fig, ax = plt.subplots(figsize=(6.0, 3.0))
x = np.arange(len(ag))
for i, (k, lab, c) in enumerate((("J_M2c_M2_same_seed", "M2c vs M2, same seed", COL["M2c"]),
                                 ("J_M2_M2_cross_seed", "M2 vs M2, different seeds", COL["M2"]),
                                 ("J_M2c_M2_cross_seed", "M2c vs M2, different seeds", "#86b6ef"))):
    ax.bar(x + (i - 1) * 0.27, [float(a[k]) for a in ag], 0.25, color=c, label=lab)
ax.set_xticks(x); ax.set_xticklabels([f"K={a['K']} N={a['N']}" for a in ag])
ax.set_ylabel("error-set Jaccard"); ax.set_ylim(0, 1); ax.set_title("P2: do the two heads get the same images wrong?", color=INK)
ax.legend(fontsize=7, loc="upper right")
fig.tight_layout(); fig.savefig(f"{OUT}/agreement.png"); plt.close(fig)

# 6. effect vs MDE
fig, ax = plt.subplots(figsize=(4.6, 3.6))
mk = {"D": ("o", COL["M2c"]), "BE=M3-M2": ("s", COL["M2"]), "BEc=M3-M2c": ("^", COL["M3"])}
for name, (m, c) in mk.items():
    pts = [(float(r["mde"]), abs(float(r["mean"]))) for r in con if r["contrast"] == name and r["subset"] in ("pooled", "all")]
    if pts:
        ax.scatter(*zip(*pts), marker=m, color=c, s=30, edgecolor="white", linewidth=0.5, label=name)
lim = max(1, max(max(float(r["mde"]), abs(float(r["mean"]))) for r in con)) * 1.1
ax.plot([0, lim], [0, lim], color=INK2, lw=0.8, ls="--"); ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel("MDE (pp, 80% power)"); ax.set_ylabel("|effect| (pp)"); ax.legend(fontsize=7)
ax.set_title("Above the line: resolvable at n = 10", color=INK)
fig.tight_layout(); fig.savefig(f"{OUT}/effect_vs_mde.png"); plt.close(fig)
print("Wrote " + ", ".join(sorted(os.listdir(OUT))) + f" to {OUT}/")
