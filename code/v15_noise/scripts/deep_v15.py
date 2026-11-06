#!/usr/bin/env python3
"""v15 EXPLORATORY figures (not pre-registered) -> figures_v15_deep/. Reads v15_runs.csv.
    python3 scripts/deep_v15.py        (from the v15 repo root, or next to v15_runs.csv)

D1  Training under the noise removes its cost: per model, accuracy change vs alpha,
    trained at alpha (matched) vs trained clean (deployed).
D2  Head amplification: M2 (QNN head) vs M3 (classical head), same Circuit-4 front-end,
    same noise, trained clean.
D3  Noise tolerance: per seed, the alpha at which a clean-trained model first loses 2 pp.
D4  Seed 46: every seed's deployed curve, M5 (frozen) and M3 (trained).
D5  Subset consistency: the three alpha=1 primaries on subsets A and B separately, and pooled.
D6  Training dynamics: best epoch vs training alpha, per model.
"""
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

HERE = Path(__file__).resolve()
ROOT = HERE.parents[1] if (HERE.parents[1] / "v15_runs.csv").exists() else HERE.parent
OUT = ROOT / "figures_v15_deep"
OUT.mkdir(exist_ok=True)
SURF, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e8e7e2", "#c3c2b7"
COL = {"M1": "#2a78d6", "M2": "#eb6834", "M3": "#1baf7a", "M4": "#eda100", "M5": "#e87ba4"}
MK = {"M1": "o", "M2": "s", "M3": "D", "M4": "^", "M5": "v"}
NAME = {"M1": "M1 RandomLayers + QNN head", "M2": "M2 Circuit 4 + QNN head", "M3": "M3 Circuit 4 + classical head",
        "M4": "M4 no front-end", "M5": "M5 Circuit 4 frozen + classical head"}
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
                     "axes.edgecolor": BASE, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
                     "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
                     "legend.frameon": False, "lines.linewidth": 2})
R = list(csv.DictReader(open(ROOT / "v15_runs.csv")))
IDX = {(int(r["K"]), r["subset"], float(r["alpha_train"]), r["model"], int(r["seed"])): r for r in R}
SEEDS = list(range(42, 52))
TRAIN_A = [0, 0.5, 1, 2, 4]
EV = [0, 0.5, 1, 2, 4, 8, 16]
TAGS = {5: ["A", "B"], 10: ["all"]}
QM = ["M1", "M2", "M3", "M5"]


def kt(K):
    return "K=5 (A, B pooled)" if K == 5 else "K=10"


def run(K, T, a, M, S):
    return IDX[(K, T, 0.0 if M == "M4" else float(a), M, S)]


def matched(K, a, M, S):
    return np.mean([float(run(K, T, a, M, S)["test_acc"]) for T in TAGS[K]])


def dep(K, M, S, e):
    return np.mean([float(run(K, T, 0, M, S)["acc_eval_%g" % e]) for T in TAGS[K]])


def ci(d):
    d = np.asarray(d, float)
    return d.mean(), stats.t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---- D1: matched vs deployed cost, per model -----------------------------------------
fig, axes = plt.subplots(2, 5, figsize=(16, 6.2), sharex=True)
x = np.arange(len(TRAIN_A))
for i, K in enumerate((5, 10)):
    for j, M in enumerate(QM + ["M4"]):
        ax = axes[i][j]
        m = [ci([matched(K, a, M, S) - matched(K, 0, M, S) for S in SEEDS]) for a in TRAIN_A]
        d = [ci([dep(K, M, S, a) - dep(K, M, S, 0) for S in SEEDS]) for a in TRAIN_A]
        ax.axhspan(-1, 1, color=GRID, alpha=0.7, lw=0, zorder=0)
        ax.axhline(0, color=BASE, lw=1)
        ax.errorbar(x - 0.06, [v[0] for v in m], yerr=[v[1] for v in m], color=COL[M], marker=MK[M], ms=5,
                    capsize=2, markeredgecolor=SURF, label="trained at alpha (matched)")
        ax.errorbar(x + 0.06, [v[0] for v in d], yerr=[v[1] for v in d], color=MUTED, marker=MK[M], ms=5,
                    capsize=2, ls="--", markeredgecolor=SURF, label="trained clean (deployed)")
        ax.set_xticks(x); ax.set_xticklabels(["%g" % a for a in TRAIN_A])
        ax.set_title(("%s\n%s" % (NAME[M], kt(K))) if i == 0 else kt(K), loc="left", fontsize=8.5)
        if j == 0: ax.set_ylabel("accuracy change vs alpha=0 (pp)")
        if i == 1: ax.set_xlabel("noise alpha")
        if M == "M4":
            ax.set_ylim(-2, 2)
            ax.text(0.5, 0.18, "no quantum front-end:\nnoise has nothing to act on\n(all predictions identical at every alpha)",
                    transform=ax.transAxes, ha="center", fontsize=7.5, color=INK2)
axes[0][0].legend(fontsize=7.5, loc="lower left")
fig.suptitle("D1. Training under the noise removes its cost: matched curves sit on 0 (grey = +/-1 pp), "
             "clean-trained curves fall; M4 is the no-quantum control", x=0.01, ha="left", fontsize=11)
fig.tight_layout(); save(fig, "D1_train_under_noise.png")

# ---- D2: head amplification -----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 3.9))
for ax, K in zip(axes, (5, 10)):
    xe = np.arange(len(EV))
    for M, off in (("M2", -0.06), ("M3", 0.06)):
        mu, hw = zip(*[ci([dep(K, M, S, e) - dep(K, M, S, 0) for S in SEEDS]) for e in EV])
        ax.errorbar(xe + off, mu, yerr=hw, color=COL[M], marker=MK[M], ms=5, capsize=2, markeredgecolor=SURF, label=NAME[M])
    ax.axhline(0, color=BASE, lw=1); ax.axvline(EV.index(1), color=BASE, lw=1, ls=":")
    ax.set_xticks(xe); ax.set_xticklabels(["%g" % e for e in EV])
    ax.set_xlabel("test noise alpha (both trained without noise)"); ax.set_ylabel("accuracy change vs noiseless (pp)")
    ax.set_title(kt(K), loc="left")
axes[0].legend(fontsize=8, loc="lower left")
fig.suptitle("D2. Same front-end, same noise: the QNN head loses far more than the classical head",
             x=0.01, ha="left", fontsize=11)
fig.tight_layout(); save(fig, "D2_head_amplification.png")


# ---- D3: tolerance threshold ------------------------------------------------------------
def thr(K, T, M, S, drop=2.0):
    r = run(K, T, 0, M, S)
    y = [float(r["acc_eval_%g" % e]) for e in EV]
    for i in range(1, len(EV)):
        if y[0] - y[i] >= drop:
            x0 = np.log2(EV[i - 1]) if EV[i - 1] > 0 else np.log2(0.25)
            x1 = np.log2(EV[i]); d0, d1 = y[0] - y[i - 1], y[0] - y[i]
            return 2 ** (x0 + (drop - d0) / (d1 - d0) * (x1 - x0))
    return 32.0


fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
for ax, K in zip(axes, (5, 10)):
    for k, M in enumerate(QM):
        vals = [(thr(K, T, M, S), T, S) for T in TAGS[K] for S in SEEDS]
        ys = np.array([v[0] for v in vals])
        xs = k + np.linspace(-0.18, 0.18, len(ys))
        ax.scatter(xs, ys, s=18, color=COL[M], marker=MK[M], edgecolors=SURF, linewidths=0.6, zorder=3)
        med = np.median(ys)
        ax.hlines(med, k - 0.3, k + 0.3, color=INK, lw=1.8, zorder=4)
        ax.annotate("%.1fx" % med, (k + 0.32, med), fontsize=8, color=INK2, va="center")
        for (v, T, S), xx in zip(vals, xs):
            if S == 46 and M == "M5":
                ax.annotate("46", (xx, v), xytext=(3, -8), textcoords="offset points", fontsize=7, color=INK2)
    ax.set_yscale("log", base=2)
    ax.set_yticks([0.5, 1, 2, 4, 8, 16]); ax.set_yticklabels(["0.5x", "1x", "2x", "4x", "8x", "16x"])
    ax.axhline(1, color=BASE, lw=1, ls=":")
    ax.set_xticks(range(len(QM))); ax.set_xticklabels(QM)
    ax.set_title(kt(K) + ("  (each point = one seed x subset)" if K == 5 else "  (each point = one seed)"), loc="left")
axes[0].set_ylabel("noise level where a clean-trained\nmodel first loses 2 pp (x device)")
fig.suptitle("D3. Noise tolerance: classical-head models absorb several times more noise than QNN-head models "
             "(bar = median; dotted = device level)", x=0.01, ha="left", fontsize=11)
fig.tight_layout(); save(fig, "D3_noise_tolerance.png")

# ---- D4: seed 46 ------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(11, 6.6), sharex=True)
for j, K in enumerate((5, 10)):
    for i, M in enumerate(("M5", "M3")):
        ax = axes[i][j]; xe = np.arange(len(EV))
        for S in SEEDS:
            y = [dep(K, M, S, e) for e in EV]
            if S == 46:
                ax.plot(xe, y, color=COL[M], lw=2.4, marker="o", ms=4, markeredgecolor=SURF, zorder=3)
                ax.annotate("seed 46", (xe[3], y[3]), xytext=(6, -2), textcoords="offset points", fontsize=8, color=INK2)
            else:
                ax.plot(xe, y, color=MUTED, lw=1, alpha=0.7)
        ax.axvline(EV.index(1), color=BASE, lw=1, ls=":")
        ax.set_xticks(xe); ax.set_xticklabels(["%g" % e for e in EV])
        ax.set_title("%s, %s" % ({"M5": "M5 frozen front-end", "M3": "M3 trained front-end"}[M], kt(K)), loc="left")
        if j == 0: ax.set_ylabel("test accuracy (%)")
        if i == 1: ax.set_xlabel("test noise alpha (trained without noise)")
fig.suptitle("D4. Every seed, trained clean: seed 46's frozen front-end is the only one that breaks near device noise",
             x=0.01, ha="left", fontsize=11)
fig.tight_layout(); save(fig, "D4_seed46.png")

# ---- D5: subset consistency of the alpha=1 primaries ------------------------------------
rows = []
for name, f in (("NC3@1  M3(1) - M3(0)", lambda K, T, S: float(run(K, T, 1, "M3", S)["test_acc"]) - float(run(K, T, 0, "M3", S)["test_acc"])),
                ("F@1  M3 - M4", lambda K, T, S: float(run(K, T, 1, "M3", S)["test_acc"]) - float(run(K, T, 1, "M4", S)["test_acc"])),
                ("T@1  M3 - M5", lambda K, T, S: float(run(K, T, 1, "M3", S)["test_acc"]) - float(run(K, T, 1, "M5", S)["test_acc"]))):
    for lab, K, tags in (("K=5 subset A", 5, ["A"]), ("K=5 subset B", 5, ["B"]), ("K=5 pooled", 5, ["A", "B"]), ("K=10", 10, ["all"])):
        d = [np.mean([f(K, T, S) for T in tags]) for S in SEEDS]
        rows.append((name, lab) + ci(d))
fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
labs = ["K=5 subset A", "K=5 subset B", "K=5 pooled", "K=10"]
for ax, name in zip(axes, [r[0] for r in rows[::4]]):
    rs = [r for r in rows if r[0] == name]
    for y, r in enumerate(rs):
        pooled = r[1] in ("K=5 pooled", "K=10")
        ax.errorbar(r[2], y, xerr=r[3], fmt="D" if pooled else "o", color="#2a78d6" if pooled else MUTED,
                    ms=6 if pooled else 5, capsize=2, markeredgecolor=SURF)
    ax.axvspan(-1, 1, color=GRID, alpha=0.7, lw=0, zorder=0); ax.axvline(0, color=BASE, lw=1)
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs); ax.invert_yaxis()
    ax.set_title(name, loc="left"); ax.set_xlabel("difference (pp), 95% CI")
fig.suptitle("D5. The three alpha=1 primaries agree in sign on both K=5 subsets (grey = +/-1 pp; blue = the pre-registered pooled test)",
             x=0.01, ha="left", fontsize=11)
fig.tight_layout(); save(fig, "D5_subset_consistency.png")

# ---- D6: training dynamics -------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, K in zip(axes, (5, 10)):
    for k, M in enumerate(QM):
        med = [np.median([int(run(K, T, a, M, S)["best_epoch"]) for T in TAGS[K] for S in SEEDS]) for a in TRAIN_A]
        q1 = [np.percentile([int(run(K, T, a, M, S)["best_epoch"]) for T in TAGS[K] for S in SEEDS], 25) for a in TRAIN_A]
        q3 = [np.percentile([int(run(K, T, a, M, S)["best_epoch"]) for T in TAGS[K] for S in SEEDS], 75) for a in TRAIN_A]
        off = (k - 1.5) * 0.07
        ax.errorbar(np.arange(len(TRAIN_A)) + off, med, yerr=[np.subtract(med, q1), np.subtract(q3, med)], color=COL[M],
                    marker=MK[M], ms=5, capsize=2, markeredgecolor=SURF, label=NAME[M])
    m4 = np.median([int(run(K, T, 0, "M4", S)["best_epoch"]) for T in TAGS[K] for S in SEEDS])
    ax.axhline(m4, color=COL["M4"], lw=1.5, ls="--", zorder=1, label="M4 no front-end (median, alpha=0)")
    ax.set_xticks(range(len(TRAIN_A))); ax.set_xticklabels(["%g" % a for a in TRAIN_A])
    ax.set_xlabel("training noise alpha"); ax.set_title(kt(K), loc="left")
axes[0].set_ylabel("best epoch (median, IQR)")
axes[1].legend(fontsize=7.5, loc="lower right")
fig.suptitle("D6. Noise does not change how training runs: the best epoch has no trend with alpha",
             x=0.01, ha="left", fontsize=11)
fig.tight_layout(); save(fig, "D6_training_dynamics.png")

for r in rows:
    print("D5 %-22s %-14s %+.2f +/- %.2f" % r)
print("wrote", sorted(os.listdir(OUT)))
