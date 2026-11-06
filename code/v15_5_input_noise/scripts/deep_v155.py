#!/usr/bin/env python3
"""v15.5 EXPLORATORY figures (not pre-registered) -> figures_v155_deep/.
Reads v155_runs.csv and v155_transfer.csv (collect_v155.py writes them) and, for D0,
noise_examples_v155.npz (verify_v155.py writes it).
    python3 scripts/deep_v155.py        (from the v15.5 repo root)

D0  What the noise looks like: 8 real test digits at every sigma, and salt-and-pepper.
D1  Robustness curves: accuracy lost vs sigma, trained clean (deployed) vs trained noisy (matched).
D2  Transfer matrix: trained at sigma_t, tested at sigma_e (M3, M4, and their difference F).
D3  Head amplification: deployed loss of M2 (QNN head) minus M3 (classical head), same front-end.
D4  Seed 46: every seed's deployed curve, M5 (frozen) and M3 (trained).
D5  T with and without seed 46, both arms.
D6  Subset consistency: the four primaries on subsets A, B, pooled and K=10.
D7  Salt-and-pepper (trained clean): accuracy vs p.
D8  Training dynamics: median best epoch vs training sigma.
Every figure title says EXPLORATORY. Each prints a one-line "what and why" for the results doc.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures_v155_deep"
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
R = list(csv.DictReader(open(ROOT / "v155_runs.csv")))
IDX = dict(((int(r["K"]), r["subset"], float(r["sigma_train"]), r["model"], int(r["seed"])), r) for r in R)
KS = sorted({int(r["K"]) for r in R})
SEEDS = sorted({int(r["seed"]) for r in R})
SIG = sorted({float(r["sigma_train"]) for r in R})
EV = sorted(float(k[len("acc_eval_g"):]) for k in R[0] if k.startswith("acc_eval_g"))
SP = sorted(float(k[len("acc_eval_sp"):]) for k in R[0] if k.startswith("acc_eval_sp"))
MODELS = ["M1", "M2", "M3", "M4", "M5"]
TAGS = {5: ["A", "B"], 10: ["all"]}
NOTES = []


def kt(K):
    return "K=5 (A, B pooled)" if K == 5 else "K=10"


def matched(K, s, M, S):
    return np.mean([float(IDX[(K, T, s, M, S)]["test_acc"]) for T in TAGS[K]])


def dep(K, M, S, e, train=0.0, kind="g"):
    return np.mean([float(IDX[(K, T, train, M, S)]["acc_eval_%s%g" % (kind, e)]) for T in TAGS[K]])


def ci(v):
    v = np.asarray(v, float)
    return v.mean(), stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0


def save(fig, name, what):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    NOTES.append((name, what))
    print("  %s -- %s" % (name, what))


print("Writing figures_v155_deep/ (EXPLORATORY):")

# ---- D0 examples -----------------------------------------------------------------------
ex = ROOT / "noise_examples_v155.npz"
if ex.exists():
    z = np.load(ex)
    g, sg, sp, ps = z["gauss"], z["sigmas"], z["sp"], z["ps"]
    n = min(6, g.shape[1])
    rows = len(sg) + len(ps)
    fig, axes = plt.subplots(n, rows, figsize=(1.05 * rows, 1.1 * n), squeeze=False)
    for i in range(n):
        for j in range(rows):
            ax = axes[i][j]
            img = g[j, i] if j < len(sg) else sp[j - len(sg), i]
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s_ in ax.spines.values():
                s_.set_visible(False)
            if i == 0:
                ax.set_title(("sigma=%g" % sg[j]) if j < len(sg) else ("s&p p=%g" % ps[j - len(sg)]), fontsize=7)
    fig.suptitle("D0 (illustration): the corruption at each level, same 6 test digits", x=0.01, ha="left", fontsize=10)
    save(fig, "D0_noise_examples.png",
         "What the corruption looks like at each sigma and salt-and-pepper p, so the accuracy numbers have a picture.")
else:
    print("  D0 skipped (noise_examples_v155.npz not found; verify_v155.py writes it)")

# ---- D1 robustness curves ----------------------------------------------------------------
fig, axes = plt.subplots(len(KS), 5, figsize=(15, 3.2 * len(KS)), squeeze=False, sharey="row")
for i, K in enumerate(KS):
    for j, M in enumerate(MODELS):
        ax = axes[i][j]
        base = [matched(K, 0.0, M, S) for S in SEEDS]
        mu, hw = zip(*[ci([dep(K, M, S, e) - b for S, b in zip(SEEDS, base)]) for e in EV])
        ax.errorbar(range(len(EV)), mu, yerr=hw, color=COL[M], marker=MK[M], ms=4, capsize=2, ls="--",
                    label="trained clean (deployed)")
        X = dict((e, k) for k, e in enumerate(EV))
        mu2, hw2 = zip(*[ci([matched(K, s, M, S) - b for S, b in zip(SEEDS, base)]) for s in SIG])
        ax.errorbar([X[s] for s in SIG], mu2, yerr=hw2, color=INK2, marker="o", ms=4, capsize=2, label="trained at sigma (matched)")
        ax.axhline(0, color=BASE, lw=1)
        ax.axhspan(-1, 1, color=GRID, alpha=0.6, zorder=0, lw=0)
        ax.set_xticks(range(len(EV))); ax.set_xticklabels(["%g" % e for e in EV], fontsize=7)
        ax.set_title("%s\n%s" % (NAME[M], kt(K)), fontsize=8, loc="left")
        if j == 0:
            ax.set_ylabel("accuracy change vs clean (pp)")
        if i == len(KS) - 1:
            ax.set_xlabel("test sigma")
axes[0][0].legend(fontsize=7, loc="lower left")
fig.suptitle("D1 (EXPLORATORY): accuracy lost to pixel noise, trained clean vs trained under the noise", x=0.01, ha="left", fontsize=11)
fig.tight_layout()
save(fig, "D1_robustness_curves.png",
     "How fast each model loses accuracy as the images get noisier, and how much training on noisy images gives back.")

# ---- D2 transfer matrix ------------------------------------------------------------------
TR = list(csv.DictReader(open(ROOT / "v155_transfer.csv")))
TX = dict(((int(r["K"]), r["model"], float(r["sigma_train"]), float(r["sigma_test"])), float(r["mean"])) for r in TR)
fig, axes = plt.subplots(len(KS), 3, figsize=(12, 3.6 * len(KS)), squeeze=False)
for i, K in enumerate(KS):
    for j, (lab, f) in enumerate((("M3 Circuit 4 + classical", lambda a, b: TX[(K, "M3", a, b)]),
                                  ("M4 no front-end", lambda a, b: TX[(K, "M4", a, b)]),
                                  ("F = M3 - M4", lambda a, b: TX[(K, "M3", a, b)] - TX[(K, "M4", a, b)]))):
        Z = np.array([[f(a, b) for b in EV] for a in SIG])
        ax = axes[i][j]
        if j < 2:
            im = ax.imshow(Z, cmap="viridis", aspect="auto", vmin=np.nanmin(Z), vmax=np.nanmax(Z))
        else:
            lim = max(1.0, np.nanmax(np.abs(Z)))
            im = ax.imshow(Z, cmap="RdBu", aspect="auto", vmin=-lim, vmax=lim)
        for a_ in range(Z.shape[0]):
            for b_ in range(Z.shape[1]):
                ax.text(b_, a_, "%.1f" % Z[a_, b_], ha="center", va="center", fontsize=6.5,
                        color="white" if (j < 2 and Z[a_, b_] < np.nanmean(Z)) else INK)
        ax.set_xticks(range(len(EV))); ax.set_xticklabels(["%g" % e for e in EV])
        ax.set_yticks(range(len(SIG))); ax.set_yticklabels(["%g" % s for s in SIG])
        ax.set_xlabel("test sigma"); ax.set_ylabel("training sigma"); ax.grid(False)
        ax.set_title("%s, %s" % (lab, kt(K)), loc="left", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.suptitle("D2 (EXPLORATORY): trained at one noise level, tested at another (mean accuracy %, and F in pp)",
             x=0.01, ha="left", fontsize=11)
fig.tight_layout()
save(fig, "D2_transfer_matrix.png",
     "Whether training at one noise level protects at other levels, and whether the front-end changes that (right panel: F).")

# ---- D3 head amplification ---------------------------------------------------------------
fig, axes = plt.subplots(1, len(KS), figsize=(5.4 * len(KS), 3.8), squeeze=False)
for ax, K in zip(axes[0], KS):
    for M, lab in (("M2", "M2 (QNN head)"), ("M3", "M3 (classical head)"), ("M1", "M1 (QNN head, RandomLayers)")):
        mu, hw = zip(*[ci([dep(K, M, S, e) - matched(K, 0.0, M, S) for S in SEEDS]) for e in EV])
        ax.errorbar(range(len(EV)), mu, yerr=hw, color=COL[M], marker=MK[M], ms=5, capsize=2, label=lab)
    ax.axhline(0, color=BASE, lw=1)
    ax.set_xticks(range(len(EV))); ax.set_xticklabels(["%g" % e for e in EV])
    ax.set_xlabel("test sigma (trained clean)"); ax.set_ylabel("accuracy lost (pp)"); ax.set_title(kt(K), loc="left")
axes[0][0].legend(fontsize=7.5, loc="lower left")
fig.suptitle("D3 (EXPLORATORY): does the quantum head amplify input noise? Same Circuit-4 front-end, M2 vs M3",
             x=0.01, ha="left", fontsize=11)
fig.tight_layout()
save(fig, "D3_head_amplification.png",
     "Checks whether v15's finding (the QNN head magnifies front-end noise) also holds when the noise is in the pixels.")

# ---- D4 seed 46 --------------------------------------------------------------------------
fig, axes = plt.subplots(len(KS), 2, figsize=(10, 3.6 * len(KS)), squeeze=False, sharey="row")
for i, K in enumerate(KS):
    for j, M in enumerate(("M5", "M3")):
        ax = axes[i][j]
        for S in SEEDS:
            ys = [dep(K, M, S, e) for e in EV]
            ax.plot(range(len(EV)), ys, color=COL[M] if S == 46 else BASE, lw=2.4 if S == 46 else 1.0,
                    marker="o" if S == 46 else None, ms=4, zorder=3 if S == 46 else 1,
                    label="seed 46" if S == 46 else None)
        ax.set_xticks(range(len(EV))); ax.set_xticklabels(["%g" % e for e in EV])
        ax.set_title("%s, %s (grey = other seeds)" % (NAME[M], kt(K)), loc="left", fontsize=8.5)
        ax.set_xlabel("test sigma (trained clean)")
        if j == 0:
            ax.set_ylabel("test accuracy (%)")
        ax.legend(fontsize=7.5)
fig.suptitle("D4 (EXPLORATORY): seed 46's frozen front-end under noisy input", x=0.01, ha="left", fontsize=11)
fig.tight_layout()
save(fig, "D4_seed46.png",
     "Seed 46 was the worst frozen front-end in v13.5, v14 and v15; this shows whether it is also the noise-fragile one here.")

# ---- D5 T with / without seed 46 --------------------------------------------------------
fig, axes = plt.subplots(1, len(KS), figsize=(5.4 * len(KS), 3.8), squeeze=False)
for ax, K in zip(axes[0], KS):
    for arm, ls, col in (("deployed", "--", "#eb6834"), ("matched", "-", "#2a78d6")):
        for drop, mk, a_ in ((False, "o", 1.0), (True, "s", 0.55)):
            ss = [S for S in SEEDS if not (drop and S == 46)]
            if arm == "matched":
                lv = SIG
                f = lambda s, S: matched(K, s, "M3", S) - matched(K, s, "M5", S)  # noqa: E731
            else:
                lv = EV
                f = lambda s, S: dep(K, "M3", S, s) - dep(K, "M5", S, s)  # noqa: E731
            X = dict((e, k) for k, e in enumerate(EV))
            mu, hw = zip(*[ci([f(s, S) for S in ss]) for s in lv])
            ax.errorbar([X[s] + (0.08 if drop else 0) for s in lv], mu, yerr=hw, color=col, ls=ls, marker=mk, ms=4,
                        capsize=2, alpha=a_, label="%s%s" % (arm, ", without seed 46" if drop else ""))
    ax.axhline(0, color=BASE, lw=1); ax.axhspan(-1, 1, color=GRID, alpha=0.6, zorder=0, lw=0)
    ax.set_xticks(range(len(EV))); ax.set_xticklabels(["%g" % e for e in EV])
    ax.set_xlabel("sigma"); ax.set_ylabel("T = M3 - M5 (pp)"); ax.set_title(kt(K), loc="left")
axes[0][0].legend(fontsize=7)
fig.suptitle("D5 (EXPLORATORY): training vs freezing the front-end, with and without seed 46", x=0.01, ha="left", fontsize=11)
fig.tight_layout()
save(fig, "D5_T_without46.png",
     "How much of the training benefit T is the rescue of one bad initialisation (seed 46), at each noise level.")

# ---- D6 subset consistency ----------------------------------------------------------------
CR = list(csv.DictReader(open(ROOT / "v155_contrasts.csv")))
want = [("F", "deployed", "0.3", "F dep@0.3"), ("slope_F", "deployed", "", "slope dep F"),
        ("F", "matched", "0.3", "F mat@0.3"), ("T", "deployed", "0.3", "T dep@0.3")]
cells = [(5, "A"), (5, "B"), (5, "pooled"), (10, "all")]
fig, axes = plt.subplots(1, len(want), figsize=(3.4 * len(want), 3.6), squeeze=False)
for ax, (c, arm, lv, lab) in zip(axes[0], want):
    for x, (K, sub) in enumerate(cells):
        r = [q for q in CR if int(q["K"]) == K and q["subset"] == sub and q["contrast"] == c and q["arm"] == arm and
             ((lv == "" and q["level"] in ("", "nan")) or (lv != "" and q["level"] not in ("", "nan") and float(q["level"]) == float(lv)))
             and q["role"] in ("secondary", "PRIMARY")]
        if not r:
            continue
        r = r[0]
        d, lo, hi = float(r["diff"]), float(r["lo"]), float(r["hi"])
        ax.errorbar([x], [d], yerr=[[d - lo], [hi - d]], color=INK if sub in ("pooled", "all") else MUTED, marker="o", capsize=3)
    ax.axhline(0, color=BASE, lw=1); ax.axhspan(-1, 1, color=GRID, alpha=0.6, zorder=0, lw=0)
    ax.set_xticks(range(len(cells))); ax.set_xticklabels(["K5 A", "K5 B", "K5 pooled", "K10"], fontsize=7.5)
    ax.set_title(lab + (" (pp per 0.1 sigma)" if c.startswith("slope") else " (pp)"), loc="left", fontsize=9)
fig.suptitle("D6 (EXPLORATORY): the four primaries on each subset and pooled, 95% CI", x=0.01, ha="left", fontsize=11)
fig.tight_layout()
save(fig, "D6_subset_consistency.png",
     "Whether the primary results point the same way on both 5-class subsets and at 10 classes.")

# ---- D7 salt-and-pepper ----------------------------------------------------------------
if SP:
    fig, axes = plt.subplots(1, len(KS), figsize=(5.4 * len(KS), 3.8), squeeze=False)
    for ax, K in zip(axes[0], KS):
        lv = [0.0] + SP
        for M in MODELS:
            mu, hw = zip(*[ci([matched(K, 0.0, M, S) if p == 0 else dep(K, M, S, p, kind="sp") for S in SEEDS]) for p in lv])
            ax.errorbar(range(len(lv)), mu, yerr=hw, color=COL[M], marker=MK[M], ms=5, capsize=2, label=NAME[M],
                        markeredgecolor=SURF)
        ax.set_xticks(range(len(lv))); ax.set_xticklabels(["%g" % p for p in lv])
        ax.set_xlabel("salt-and-pepper p (trained clean)"); ax.set_ylabel("test accuracy (%)"); ax.set_title(kt(K), loc="left")
    axes[0][0].legend(fontsize=7, loc="lower left")
    fig.suptitle("D7 (EXPLORATORY): a different corruption, no extra training: salt-and-pepper noise",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    save(fig, "D7_salt_and_pepper.png",
         "Checks whether the Gaussian-noise ranking of models holds for a different kind of corruption.")

# ---- D8 training dynamics ----------------------------------------------------------------
fig, axes = plt.subplots(1, len(KS), figsize=(5.4 * len(KS), 3.6), squeeze=False)
for ax, K in zip(axes[0], KS):
    for M in MODELS:
        med = [np.median([float(IDX[(K, T, s, M, S)]["best_epoch"]) for T in TAGS[K] for S in SEEDS]) for s in SIG]
        ax.plot(range(len(SIG)), med, color=COL[M], marker=MK[M], ms=5, label=NAME[M], markeredgecolor=SURF)
    ax.set_xticks(range(len(SIG))); ax.set_xticklabels(["%g" % s for s in SIG])
    ax.set_xlabel("training sigma"); ax.set_ylabel("median best epoch"); ax.set_title(kt(K), loc="left")
axes[0][0].legend(fontsize=7)
fig.suptitle("D8 (EXPLORATORY): does training on noisy images change how long training takes?", x=0.01, ha="left", fontsize=11)
fig.tight_layout()
save(fig, "D8_training_dynamics.png",
     "Whether noise augmentation slows convergence or pushes runs towards the 200-epoch cap.")

with open(OUT / "README_deep.md", "w") as f:
    f.write("# v15.5 exploratory figures (not pre-registered)\n\n| Figure | What and why |\n|---|---|\n")
    for n_, w_ in NOTES:
        f.write("| `%s` | %s |\n" % (n_, w_))
print("Done. (figures_v155_deep/README_deep.md lists what and why)")
