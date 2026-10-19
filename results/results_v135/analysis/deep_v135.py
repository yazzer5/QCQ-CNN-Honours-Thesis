#!/usr/bin/env python3
"""
v13.5 deep-analysis figures D1-D8 (run from the folder holding v135_runs.csv).
    python3 deep_v135.py
Writes figures_v135_deep/D*.png (200 dpi) and deep_numbers.json.
D4-D7 are exploratory (not pre-registered).
"""
import csv, json, collections
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("figures_v135_deep"); OUT.mkdir(exist_ok=True)
COL = {"M1": "#2a78d6", "M2": "#eb6834", "M3": "#1baf7a", "M4": "#4a3aa7", "M5": "#c2417e"}
WCOL = {3: "#c2417e", 6: "#2a78d6", 15: "#eb6834"}
INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False})
KS = [3, 4, 5, 10]
HEAD = {3: "pooled", 4: "pooled", 5: "pooled", 10: "all"}
WID = {3: [3], 4: [3], 5: [3, 6, 15], 10: [3, 6, 15]}
SEEDS = list(range(42, 52))
CELLS = [(3, "A", 3), (3, "B", 3), (4, "A", 3), (4, "B", 3), (5, "A", 3), (5, "A", 6), (5, "A", 15),
         (5, "B", 3), (5, "B", 6), (5, "B", 15), (10, "all", 3), (10, "all", 6), (10, "all", 15)]

rows = list(csv.DictReader(open("v135_runs.csv")))
acc, best = collections.defaultdict(dict), collections.defaultdict(dict)
for r in rows:
    k = (int(r["K"]), r["subset"], int(r["W"]), r["model"])
    acc[k][int(r["seed"])] = float(r["test_acc"])
    if r["best_epoch"] not in ("", "None"):
        best[k][int(r["seed"])] = int(float(r["best_epoch"]))
for d in (acc, best):
    for (K, t, W, m) in list(d):
        if t == "A" and (K, "B", W, m) in d:
            A, B = d[(K, "A", W, m)], d[(K, "B", W, m)]
            d[(K, "pooled", W, m)] = {s: (A[s] + B[s]) / 2 for s in set(A) & set(B)}


def ci(v):
    v = np.asarray(v, float); n = len(v)
    se = v.std(ddof=1) / np.sqrt(n); t = stats.t.ppf(.975, n - 1)
    return v.mean(), v.mean() - t * se, v.mean() + t * se, stats.ttest_1samp(v, 0).pvalue if v.std() > 0 else 1.0


def diff(K, t, W, a, b, seeds=SEEDS):
    return [acc[(K, t, W, a)][s] - acc[(K, t, W, b)][s] for s in seeds]


NUM = {}
xs = np.arange(len(KS))

# D1 -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.4))
for i, m in enumerate(["M1", "M2", "M3", "M4", "M5"]):
    mu, e = [], []
    for K in KS:
        v = list(acc[(K, HEAD[K], 3, m)].values()); a, lo, hi, _ = ci(v); mu.append(a); e.append(hi - a)
    ax.errorbar(xs + (i - 2) * .07, mu, yerr=e, marker="o", ms=5, lw=2 if m == "M5" else 1, color=COL[m], capsize=3, label=m)
    NUM[f"D1_{m}"] = dict(zip(KS, [round(x, 2) for x in mu]))
ax.set_xticks(xs, [f"K={K}" for K in KS]); ax.set_ylabel("test accuracy (%)")
ax.set_title("D1: Accuracy of all five models vs number of classes (width 3, 95% CI)", fontsize=10)
ax.legend(ncol=5, loc="lower left"); fig.tight_layout(); fig.savefig(OUT / "D1_accuracy_vs_K.png", dpi=200); plt.close(fig)


# D2 / D3 ---------------------------------------------------------------------
def effect_plot(fname, title, pairs, widths_of, ylab):
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.axhspan(-1, 1, color=GRID, alpha=.7, lw=0, label="±1 pp equivalence margin"); ax.axhline(0, color=MUTED, lw=.8)
    out = {}
    for j, W in enumerate([3, 6, 15]):
        pts = [(i, K) for i, K in enumerate(KS) if W in widths_of(K)]
        if not pts:
            continue
        x, m, lo, hi = [], [], [], []
        for i, K in pts:
            a, l, h, p = ci(diff(K, HEAD[K], W, *pairs)); x.append(i + (j - 1) * .13); m.append(a); lo.append(l); hi.append(h)
            out[f"K{K}_w{W}"] = dict(diff=round(a, 2), lo=round(l, 2), hi=round(h, 2), p=round(p, 4))
        m, lo, hi = map(np.array, (m, lo, hi))
        ax.errorbar(x, m, yerr=[m - lo, hi - m], fmt="o", capsize=3, color=WCOL[W], label=f"width {W}")
    ax.set_xticks(xs, [f"K={K}" for K in KS]); ax.set_ylabel(ylab); ax.set_title(title, fontsize=10)
    ax.legend(loc="upper left", fontsize=8); fig.tight_layout(); fig.savefig(OUT / fname, dpi=200); plt.close(fig)
    return out


NUM["D2"] = effect_plot("D2_training_effect_vs_K.png",
                        "D2: Effect of TRAINING the front-end (M3 − M5) vs number of classes",
                        ("M3", "M5"), lambda K: WID[K], "M3 − M5 (pp), 95% CI")
NUM["D3"] = effect_plot("D3_untrained_frontend_vs_none.png",
                        "D3: Untrained quantum front-end vs no front-end (M5 − M4)",
                        ("M5", "M4"), lambda K: [3], "M5 − M4 (pp), 95% CI")

# D4 -------------------------------------------------------------------------
M = np.array([[acc[c + ("M3",)][s] - acc[c + ("M5",)][s] for s in SEEDS] for c in CELLS])
fig, ax = plt.subplots(figsize=(8.5, 6))
im = ax.imshow(M, cmap="RdBu_r", vmin=-11.5, vmax=11.5, aspect="auto")
ax.set_xticks(range(10), [str(s) for s in SEEDS]); ax.set_xlabel("seed (= the frozen front-end's random initialisation)")
ax.set_yticks(range(len(CELLS)), [f"K={k} {t} w{w}" for k, t, w in CELLS], fontsize=9)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=7, color="white" if abs(M[i, j]) > 6 else INK)
ax.add_patch(plt.Rectangle((3.5, -.5), 1, len(CELLS), fill=False, ec=INK, lw=2))
cb = fig.colorbar(im, ax=ax, shrink=.8); cb.set_label("M3 − M5 (pp): red = training helped")
ax.set_title("D4: Per-seed effect of training the front-end, every cell (exploratory)\n"
             "seed 46 (boxed) is a bad random front-end that training rescues every time", fontsize=10)
for s in ax.spines.values(): s.set_visible(False)
ax.grid(False); fig.tight_layout(); fig.savefig(OUT / "D4_per_seed_heatmap.png", dpi=200); plt.close(fig)
z = collections.defaultdict(list)
for c in CELLS:
    v = np.array([acc[c + ("M5",)][s] for s in SEEDS]); zz = (v - v.mean()) / v.std(ddof=1)
    for s, q in zip(SEEDS, zz): z[s].append(q)
NUM["D4_M5_mean_z"] = {s: round(float(np.mean(z[s])), 2) for s in SEEDS}
NUM["D4_seed46_worst_cells"] = int(sum(np.argmin([acc[c + ("M5",)][s] for s in SEEDS]) == SEEDS.index(46) for c in CELLS))

# D5 -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.axhspan(-1, 1, color=GRID, alpha=.7, lw=0); ax.axhline(0, color=MUTED, lw=.8)
NUM["D5"] = {}
for j, (lab, seeds, c) in enumerate([("all 10 seeds", SEEDS, "#c2417e"), ("without seed 46", [s for s in SEEDS if s != 46], "#1baf7a")]):
    x, m, lo, hi = [], [], [], []
    for i, K in enumerate(KS):
        W = 15 if K in (5, 10) else 3
        a, l, h, p = ci(diff(K, HEAD[K], W, "M3", "M5", seeds)); x.append(i + (j - .5) * .2); m.append(a); lo.append(l); hi.append(h)
        NUM["D5"][f"{lab}_K{K}_w{W}"] = dict(diff=round(a, 2), lo=round(l, 2), hi=round(h, 2), p=round(p, 4))
    m, lo, hi = map(np.array, (m, lo, hi))
    ax.errorbar(x, m, yerr=[m - lo, hi - m], fmt="o", capsize=3, color=c, label=lab)
ax.set_xticks(xs, ["K=3 (w3)", "K=4 (w3)", "K=5 (w15)", "K=10 (w15)"]); ax.set_ylabel("M3 − M5 (pp), 95% CI")
ax.set_title("D5: Training effect with and without seed 46 (exploratory)", fontsize=10)
ax.legend(loc="upper left"); fig.tight_layout(); fig.savefig(OUT / "D5_without_seed46.png", dpi=200); plt.close(fig)

# D6 -------------------------------------------------------------------------
lab, s3, s5, s5x = [], [], [], []
for K in KS:
    for W in WID[K]:
        lab.append(f"K={K}\nw{W}")
        s3.append(np.std(list(acc[(K, HEAD[K], W, "M3")].values()), ddof=1))
        s5.append(np.std(list(acc[(K, HEAD[K], W, "M5")].values()), ddof=1))
        s5x.append(np.std([acc[(K, HEAD[K], W, "M5")][s] for s in SEEDS if s != 46], ddof=1))
fig, ax = plt.subplots(figsize=(8, 4.2)); x = np.arange(len(lab)); w = .27
ax.bar(x - w, s3, w, color=COL["M3"], label="M3 (trained front-end)")
ax.bar(x, s5, w, color=COL["M5"], label="M5 (frozen front-end)")
ax.bar(x + w, s5x, w, color=COL["M5"], alpha=.4, label="M5 without seed 46")
ax.set_xticks(x, lab, fontsize=8); ax.set_ylabel("seed-to-seed SD (pp)")
ax.set_title("D6: Seed-to-seed spread, trained vs frozen front-end", fontsize=10)
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "D6_seed_stability.png", dpi=200); plt.close(fig)
NUM["D6"] = {l.replace("\n", " "): dict(M3=round(a, 2), M5=round(b, 2), M5_no46=round(c, 2)) for l, a, b, c in zip(lab, s3, s5, s5x)}

# D7 -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.2)); NUM["D7"] = {}
for k, m in enumerate(["M3", "M5"]):
    med = [np.median(list(best[(K, HEAD[K], W, m)].values())) for K in KS for W in WID[K]]
    q1 = [np.percentile(list(best[(K, HEAD[K], W, m)].values()), 25) for K in KS for W in WID[K]]
    q3 = [np.percentile(list(best[(K, HEAD[K], W, m)].values()), 75) for K in KS for W in WID[K]]
    xx = x + (k - .5) * .3; med, q1, q3 = map(np.array, (med, q1, q3))
    ax.errorbar(xx, med, yerr=[med - q1, q3 - med], fmt="o", capsize=3, color=COL[m], label=f"{m} median (IQR)")
    NUM["D7"][m] = dict(zip([l.replace("\n", " ") for l in lab], [float(v) for v in med]))
ax.axhline(30, color=MUTED, ls="--", lw=.8); ax.text(len(lab) - .5, 31, "30-epoch warmup", ha="right", fontsize=8, color=MUTED)
ax.set_xticks(x, lab, fontsize=8); ax.set_ylabel("best epoch (checkpoint kept)")
ax.set_title("D7: When each model reaches its best validation accuracy", fontsize=10)
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT / "D7_learning_speed.png", dpi=200); plt.close(fig)

# D8 -------------------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(10, 4), sharey=False); NUM["D8"] = {}
for ax, (a, b, ttl) in zip(axs, [("M5", "M1", "M5 − M1: classical vs quantum head, both front-ends frozen\n(head AND circuit type differ: descriptive)"),
                                 ("M3", "M2", "M3 − M2: classical vs quantum head, both front-ends trained\n(v12 back-end cost, for reference)")]):
    ax.axhline(0, color=MUTED, lw=.8)
    mm, lo, hi = [], [], []
    for K in KS:
        r = ci(diff(K, HEAD[K], 3, a, b)); mm.append(r[0]); lo.append(r[1]); hi.append(r[2])
        NUM["D8"][f"{a}-{b}_K{K}"] = round(r[0], 2)
    mm, lo, hi = map(np.array, (mm, lo, hi))
    ax.errorbar(xs, mm, yerr=[mm - lo, hi - mm], fmt="o-", capsize=3, color=COL[a])
    ax.set_xticks(xs, [f"K={K}" for K in KS]); ax.set_title(ttl, fontsize=9); ax.set_ylabel("difference (pp), 95% CI")
fig.suptitle("D8: The head matters more than the front-end (width 3)", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "D8_head_effect.png", dpi=200); plt.close(fig)

json.dump(NUM, open(OUT / "deep_numbers.json", "w"), indent=1, default=str)
print(json.dumps(NUM, indent=1, default=str))
