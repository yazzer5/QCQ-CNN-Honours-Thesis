#!/usr/bin/env python3
"""
v13.5 standard plot set (run AFTER collect_v135.py, from the v13.5 repo root).
    python3 scripts/plot_v135.py
Writes to figures_v135/ at 200 dpi:
    acc_vs_k.png        mean accuracy (95% CI) of M1-M5 vs K at width 3 (headline view)
    front_end_vs_k.png  HEADLINE: M3-M5 (training the front-end) and M5-M4
                        (untrained front-end vs none) vs K, 95% CI, +/-1 pp band
    strip_k{K}.png      per-seed accuracy per (model, width), per subset
    paired_k{K}.png     per-seed paired differences for the primary contrasts
    effect_vs_mde.png   |effect| vs MDE for every contrast
"""
import csv
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("figures_v135"); OUT.mkdir(exist_ok=True)
COL = {"M1": "#2a78d6", "M2": "#eb6834", "M3": "#1baf7a", "M4": "#4a3aa7", "M5": "#c2417e"}
INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False})
KS = [3, 4, 5, 10]
HEAD = {3: "pooled", 4: "pooled", 5: "pooled", 10: "all"}
TAGS = {3: ["A", "B"], 4: ["A", "B"], 5: ["A", "B"], 10: ["all"]}


def read(p):
    with open(p) as fh:
        return list(csv.DictReader(fh))


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def seeds(r):
    return np.array([float(x) for x in r["per_seed"].split(";")]) if r["per_seed"] else np.array([])


runs, con = read("v135_runs.csv"), read("v135_contrasts.csv")
C = {(r["family"], int(r["K"]), r["subset"], int(r["W"]), r["contrast"]): r for r in con}
by = defaultdict(dict)
for r in runs:
    by[(int(r["K"]), r["subset"], int(r["W"]), r["model"])][int(r["seed"])] = float(r["test_acc"])
for (K, t, W, m) in list(by):
    if t == "A" and (K, "B", W, m) in by:
        A, B = by[(K, "A", W, m)], by[(K, "B", W, m)]
        by[(K, "pooled", W, m)] = {s: (A[s] + B[s]) / 2 for s in set(A) & set(B)}
rng = np.random.default_rng(0)


def ci(v):
    v = np.asarray(v)
    if len(v) < 2:
        return np.nan
    from scipy import stats
    return stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))


# acc_vs_k -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 4.2))
xs = np.arange(len(KS))
for i, m in enumerate(["M1", "M2", "M3", "M4", "M5"]):
    mu = [np.mean(list(by[(K, HEAD[K], 3, m)].values())) if by.get((K, HEAD[K], 3, m)) else np.nan for K in KS]
    e = [ci(list(by[(K, HEAD[K], 3, m)].values())) if by.get((K, HEAD[K], 3, m)) else np.nan for K in KS]
    ax.errorbar(xs + (i - 2) * 0.07, mu, yerr=e, marker="o", ms=5, lw=1.6 if m == "M5" else 1.0,
                color=COL[m], capsize=3, label=m)
ax.set_xticks(xs, [f"K={K}" for K in KS]); ax.set_ylabel("test accuracy (%)")
ax.set_title("Mean accuracy by class count, width 3 (95% CI; K<10 pooled over subsets)", fontsize=10)
ax.legend(ncol=5, loc="lower left")
fig.tight_layout(); fig.savefig(OUT / "acc_vs_k.png", dpi=200); plt.close(fig)

# front_end_vs_k (headline) --------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for ax, (name, title) in zip(axes, [("M3-M5", "M3 - M5: effect of TRAINING the quantum front-end"),
                                    ("M5-M4", "M5 - M4: untrained quantum front-end vs none")]):
    ax.axhspan(-1, 1, color=GRID, alpha=.6, lw=0, label="±1 pp equivalence margin")
    ax.axhline(0, color=MUTED, lw=.8)
    ws = [3, 6, 15] if name == "M3-M5" else [3]
    for j, W in enumerate(ws):
        pts = [(i, C.get(("primary", K, HEAD[K], W, name))) for i, K in enumerate(KS)]
        pts = [(i, r) for i, r in pts if r and f(r["n"]) >= 2]
        if not pts:
            continue
        x = np.array([i for i, _ in pts]) + (j - (len(ws) - 1) / 2) * 0.12
        d = np.array([f(r["diff"]) for _, r in pts])
        lo = np.array([f(r["lo"]) for _, r in pts]); hi = np.array([f(r["hi"]) for _, r in pts])
        ax.errorbar(x, d, yerr=[d - lo, hi - d], fmt="o", capsize=3,
                    color=["#c2417e", "#2a78d6", "#eb6834"][j], label=f"width {W}")
    ax.set_xticks(range(len(KS)), [f"K={K}" for K in KS]); ax.set_title(title, fontsize=10)
axes[0].set_ylabel("difference (pp), 95% CI"); axes[0].legend(loc="best", fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "front_end_vs_k.png", dpi=200); plt.close(fig)

# strip per K -----------------------------------------------------------------
for K in KS:
    cols = [(m, W) for W in (3, 6, 15) for m in ("M1", "M2", "M3", "M4", "M5")
            if any(by.get((K, t, W, m)) for t in TAGS[K])]
    fig, axes = plt.subplots(1, len(TAGS[K]), figsize=(max(6, 0.75 * len(cols)) * len(TAGS[K]), 4), squeeze=False)
    for ax, t in zip(axes[0], TAGS[K]):
        for i, (m, W) in enumerate(cols):
            v = np.array(list(by.get((K, t, W, m), {}).values()))
            if len(v):
                ax.scatter(i + rng.uniform(-.15, .15, len(v)), v, s=14, color=COL[m], alpha=.8)
                ax.hlines(v.mean(), i - .3, i + .3, color=INK, lw=1.2)
        ax.set_xticks(range(len(cols)), [f"{m}\nw{W}" for m, W in cols], fontsize=8)
        ax.set_title(f"K={K} subset {t}", fontsize=10); ax.set_ylabel("test accuracy (%)")
    fig.tight_layout(); fig.savefig(OUT / f"strip_k{K}.png", dpi=200); plt.close(fig)

# paired per K -----------------------------------------------------------------
for K in KS:
    items = [(W, "M3-M5") for W in ((3, 6, 15) if K in (5, 10) else (3,))] + [(3, "M5-M4")]
    fig, ax = plt.subplots(figsize=(1.6 * len(items) + 2, 4))
    ax.axhspan(-1, 1, color=GRID, alpha=.6, lw=0); ax.axhline(0, color=MUTED, lw=.8)
    for i, (W, name) in enumerate(items):
        r = C.get(("primary", K, HEAD[K], W, name))
        if not r:
            continue
        s = seeds(r)
        ax.scatter(i + rng.uniform(-.12, .12, len(s)), s, s=16, color="#c2417e" if name == "M3-M5" else "#4a3aa7", alpha=.8)
        if f(r["n"]) >= 2:
            ax.errorbar(i + .3, f(r["diff"]), yerr=[[f(r["diff"]) - f(r["lo"])], [f(r["hi"]) - f(r["diff"])]],
                        fmt="D", color=INK, capsize=3)
    ax.set_xticks(range(len(items)), [f"{n}\nw{W}" for W, n in items], fontsize=9)
    ax.set_ylabel("per-seed difference (pp)")
    ax.set_title(f"K={K} ({HEAD[K]}): primary contrasts, per seed + mean 95% CI", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / f"paired_k{K}.png", dpi=200); plt.close(fig)

# effect vs MDE -----------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 5))
mx = 0
for r in con:
    if f(r["n"]) < 2 or r["subset"] not in ("pooled", "all"):
        continue
    e, mde = abs(f(r["diff"])), f(r["mde"]); mx = max(mx, e, mde)
    mk = "o" if r["family"] == "primary" else "s"
    ax.scatter(mde, e, marker=mk, s=30, color="#c2417e" if "M3-M5" in r["contrast"] else MUTED, alpha=.85)
    ax.annotate(f"K{r['K']} w{r['W']} {r['contrast']}", (mde, e), fontsize=6, color=MUTED,
                xytext=(3, 2), textcoords="offset points")
ax.plot([0, mx * 1.05], [0, mx * 1.05], color=INK, lw=.8, ls="--")
ax.set_xlabel("MDE (pp, 80% power)"); ax.set_ylabel("|effect| (pp)")
ax.set_title("Effect vs MDE (circles primary, squares secondary)\nbelow the line: not resolvable at this n", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "effect_vs_mde.png", dpi=200); plt.close(fig)
print(f"Wrote {len(list(OUT.glob('*.png')))} figures to {OUT}/")
