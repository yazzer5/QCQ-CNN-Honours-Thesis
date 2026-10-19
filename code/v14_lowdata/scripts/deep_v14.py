#!/usr/bin/env python3
"""v14 deep analysis figures (exploratory unless noted). Run from the v14 repo root:
    python3 scripts/deep_v14.py
Writes figures_v14_deep/ at 200 dpi:
  D1_fit_vs_gap.png     train acc at best checkpoint vs test acc, per model and N (why models differ)
  D2_be_decomposition.png  M3-M2 split into fitting difference and generalisation-gap difference
  D3_gap_vs_n.png       train-test gap vs N per model
  D4_seed46_m5.png      per-seed M5 z-score across the 9 low-N cells (seed 46 replication)
  D5_t_without46.png    M3-M5 with vs without seed 46
"""
import glob, json, re, os
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

OUT = "figures_v14_deep"; os.makedirs(OUT, exist_ok=True)
COL = {"M1": "#2a78d6", "M2": "#eb6834", "M3": "#1baf7a", "M4": "#4a3aa7", "M5": "#c2417e"}
INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "axes.axisbelow": True, "legend.frameon": False})
rows = []
for f in glob.glob("results_v14_k*_*_n*/result_*.json"):
    m = re.search(r"results_v14_k(\d+)_(\w+?)_n(\d+)/result_(M\d)_mnist_seed(\d+)", f)
    j = json.load(open(f))
    rows.append(dict(K=int(m[1]), t=m[2], N=int(m[3]), M=m[4], s=int(m[5]), te=100 * j["test_acc"], tr=100 * j["train_acc_at_best"]))
NS = [50, 100, 250]
MODELS = ["M1", "M2", "M3", "M4", "M5"]


def pooled(K, N, M, key):
    sel = lambda t: {r["s"]: r[key] for r in rows if r["K"] == K and r["t"] == t and r["N"] == N and r["M"] == M}
    if K == 10:
        return sel("all")
    A, B = sel("A"), sel("B")
    return {s: (A[s] + B[s]) / 2 for s in A if s in B}


def ci(v):
    v = np.asarray(v); return stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))


# D1
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
for ax, K in zip(axes, (5, 10)):
    for M in MODELS:
        for i, N in enumerate(NS):
            tr, te = pooled(K, N, M, "tr"), pooled(K, N, M, "te")
            ax.scatter(np.mean(list(tr.values())), np.mean(list(te.values())), s=25 + 25 * i, color=COL[M],
                       label=M if i == 0 else None, alpha=.9)
            ax.annotate(f"N={N}", (np.mean(list(tr.values())), np.mean(list(te.values()))), fontsize=6, color=MUTED,
                        xytext=(4, -2), textcoords="offset points")
    ax.set_xlabel("train accuracy at best checkpoint (%)"); ax.set_ylabel("test accuracy (%)")
    ax.set_title(f"K={K}: every model fits the training set; they differ in generalisation", fontsize=9)
axes[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/D1_fit_vs_gap.png", dpi=200); plt.close(fig)

# D2
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, K in zip(axes, (5, 10)):
    fitd, gapd, eF, eG = [], [], [], []
    for N in NS:
        tr2, tr3, te2, te3 = (pooled(K, N, m, k) for m, k in (("M2", "tr"), ("M3", "tr"), ("M2", "te"), ("M3", "te")))
        f_ = np.array([tr3[s] - tr2[s] for s in tr2]); g_ = np.array([(tr2[s] - te2[s]) - (tr3[s] - te3[s]) for s in tr2])
        fitd.append(f_.mean()); gapd.append(g_.mean()); eF.append(ci(f_)); eG.append(ci(g_))
    x = np.arange(len(NS))
    ax.bar(x - .18, fitd, .36, yerr=eF, capsize=3, color="#9ca3af", label="fitting difference (M3 train - M2 train)")
    ax.bar(x + .18, gapd, .36, yerr=eG, capsize=3, color=COL["M2"], label="generalisation-gap difference (M2 gap - M3 gap)")
    ax.set_xticks(x, [f"N={n}" for n in NS]); ax.set_ylabel("pp (95% CI)")
    ax.set_title(f"K={K}: back-end cost M3 - M2 = fitting + gap", fontsize=10)
axes[0].legend(fontsize=7, loc="upper right")
fig.tight_layout(); fig.savefig(f"{OUT}/D2_be_decomposition.png", dpi=200); plt.close(fig)

# D3
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, K in zip(axes, (5, 10)):
    for M in MODELS:
        g = [np.mean([pooled(K, N, M, "tr")[s] - pooled(K, N, M, "te")[s] for s in pooled(K, N, M, "tr")]) for N in NS]
        ax.plot(NS, g, marker="o", color=COL[M], label=M)
    ax.set_xscale("log"); ax.set_xticks(NS, [str(n) for n in NS]); ax.minorticks_off()
    ax.set_xlabel("training images per class"); ax.set_ylabel("train - test at best checkpoint (pp)")
    ax.set_title(f"K={K}: generalisation gap vs data size", fontsize=10)
axes[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/D3_gap_vs_n.png", dpi=200); plt.close(fig)

# D4
cells = [(K, t, N) for K, t in ((5, "A"), (5, "B"), (10, "all")) for N in NS]
seeds = list(range(42, 52))
Z = np.zeros((len(seeds), len(cells)))
for j, (K, t, N) in enumerate(cells):
    d = {r["s"]: r["te"] for r in rows if r["K"] == K and r["t"] == t and r["N"] == N and r["M"] == "M5"}
    v = np.array([d[s] for s in seeds]); Z[:, j] = (v - v.mean()) / v.std(ddof=1)
fig, ax = plt.subplots(figsize=(8, 4.2))
im = ax.imshow(Z, cmap="RdBu", vmin=-3, vmax=3, aspect="auto")
ax.set_yticks(range(len(seeds)), seeds); ax.set_xticks(range(len(cells)), [f"K{K} {t}\nN={N}" for K, t, N in cells], fontsize=7)
ax.grid(False); ax.set_ylabel("seed")
for i in range(len(seeds)):
    for j in range(len(cells)):
        ax.text(j, i, f"{Z[i, j]:.1f}", ha="center", va="center", fontsize=6, color="white" if abs(Z[i, j]) > 1.8 else INK)
fig.colorbar(im, ax=ax, label="M5 within-cell z-score")
ax.set_title("M5 (frozen front-end): seed 46 is the worst seed in all 9 low-data cells (exploratory)", fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/D4_seed46_m5.png", dpi=200); plt.close(fig)

# D5
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, K in zip(axes, (5, 10)):
    ax.axhspan(-1, 1, color=GRID, alpha=.6, lw=0); ax.axhline(0, color=MUTED, lw=.8)
    for k, (lab, drop, c) in enumerate((("all 10 seeds", None, COL["M5"]), ("without seed 46", 46, MUTED))):
        mu, e = [], []
        for N in NS:
            a, b = pooled(K, N, "M3", "te"), pooled(K, N, "M5", "te")
            d = np.array([a[s] - b[s] for s in a if s != drop]); mu.append(d.mean()); e.append(ci(d))
        ax.errorbar(np.array(NS) * (1 + (k - .5) * .05), mu, yerr=e, marker="o", capsize=3, color=c, label=lab)
    ax.set_xscale("log"); ax.set_xticks(NS, [str(n) for n in NS]); ax.minorticks_off()
    ax.set_xlabel("training images per class"); ax.set_title(f"K={K}: T = M3 - M5 (95% CI)", fontsize=10)
axes[0].set_ylabel("pp"); axes[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/D5_t_without46.png", dpi=200); plt.close(fig)
print("Wrote", len(os.listdir(OUT)), "figures to", OUT)

# --- D1 v2: dumbbell plot (replaces the scatter version above) ---
rows = []
for f in glob.glob("results_v14_k*_*_n*/result_*.json"):
    m = re.search(r"results_v14_k(\d+)_(\w+?)_n(\d+)/result_(M\d)_mnist_seed(\d+)", f)
    j = json.load(open(f))
    rows.append(dict(K=int(m[1]), t=m[2], N=int(m[3]), M=m[4], s=int(m[5]), te=100 * j["test_acc"], tr=100 * j["train_acc_at_best"]))
NS, MODELS = [50, 100, 250], ["M1", "M2", "M3", "M4", "M5"]
LABEL = {"M1": "M1 frozen Henderson + QNN head", "M2": "M2 trained quantum + QNN head", "M3": "M3 trained quantum + classical head",
         "M4": "M4 no front-end + classical head", "M5": "M5 frozen quantum + classical head"}
def pooled(K, N, M, key):
    sel = lambda t: {r["s"]: r[key] for r in rows if r["K"] == K and r["t"] == t and r["N"] == N and r["M"] == M}
    if K == 10: return sel("all")
    A, B = sel("A"), sel("B"); return {s: (A[s] + B[s]) / 2 for s in A if s in B}
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
for ax, K in zip(axes, (5, 10)):
    for g, N in enumerate(NS):
        if g: ax.axvline(g - .5, color=GRID, lw=1.2)
        for i, M in enumerate(MODELS):
            x = g + (i - 2) * 0.15
            tr = np.mean(list(pooled(K, N, M, "tr").values())); te = np.mean(list(pooled(K, N, M, "te").values()))
            ax.plot([x, x], [te, tr], color=COL[M], lw=2.2, alpha=.55, solid_capstyle="round")
            ax.scatter(x, tr, s=34, facecolor="white", edgecolor=COL[M], lw=1.6, zorder=3)
            ax.scatter(x, te, s=40, color=COL[M], zorder=3)
            ax.text(x + .035, (tr + te) / 2, f"{tr - te:.0f}", fontsize=7, color=COL[M], va="center")
    ax.set_xticks(range(len(NS)), [f"{n} images / class" for n in NS])
    ax.set_xlim(-.5, len(NS) - .5); ax.set_ylabel("accuracy (%)")
    ax.set_title(f"K={K}: train accuracy (hollow) vs test accuracy (filled)\nline length = generalisation gap (pp)", fontsize=10)
handles = [Line2D([], [], color=COL[m], marker="o", lw=2.2, label=LABEL[m]) for m in MODELS] + \
          [Line2D([], [], color=MUTED, marker="o", markerfacecolor="white", lw=0, label="train (at best checkpoint)"),
           Line2D([], [], color=MUTED, marker="o", lw=0, label="test")]
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(.5, -.01))
fig.tight_layout(rect=(0, .1, 1, 1))
os.makedirs("figures_v14_deep", exist_ok=True)
fig.savefig("figures_v14_deep/D1_fit_vs_gap.png", dpi=200); plt.close(fig)
print("ok")
