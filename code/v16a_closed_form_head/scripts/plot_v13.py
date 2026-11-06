#!/usr/bin/env python3
"""
v13 standard plot set (run AFTER collect_v13.py, from the v13 repo root).
    python3 scripts/plot_v13.py
Writes to figures_v13/ at 200 dpi:
    gap_vs_settings.png  HEADLINE: back-end cost (M3-M2) vs measurement settings
                         (1 = Z, 2 = Z+X, 9 = all Paulis), per K, 95% CI, MDE band
    acc_vs_width.png     mean accuracy of M2 and M3 vs readout width, per K
    primary_k{K}.png     per-seed paired differences for the 3 primary contrasts
    strip_k{K}.png       per-seed accuracy per (model, width), per subset
    effect_vs_mde.png    |effect| vs MDE for every primary/secondary contrast
    sd_vs_width.png      seed-to-seed SD of M2 and M3 vs width (stability)
    k10_completion.png   v12 contrasts at K=10 with 10 seeds (width 3)
"""
import csv
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("figures_v13"); OUT.mkdir(exist_ok=True)
WIDTHS = [3, 6, 15]
SETTINGS = {3: 1, 6: 2, 15: 9}
WLAB = {3: "w3  Z basis\n(1 setting)", 6: "w6  Z+X\n(2 settings)", 15: "w15  all Paulis\n(9 settings)"}
COL = {"M1": "#2a78d6", "M2": "#eb6834", "M3": "#1baf7a", "M4": "#4a3aa7"}
KCOL = {5: "#2a78d6", 10: "#eb6834"}
INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False})
HEAD = ("pooled", "all")
PRIMARY = ["M2w6-M2w3", "M2w15-M2w6", "M3w15-M2w15"]
PTITLE = {"M2w6-M2w3": "M2: w6 - w3\nall cosine features",
          "M2w15-M2w6": "M2: w15 - w6\nadd sine features",
          "M3w15-M2w15": "M3 - M2 at w15\ncost left at full readout"}
PCOL = dict(zip(PRIMARY, ["#2a78d6", "#eb6834", "#1baf7a"]))


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


runs, summ, con = read("v13_runs.csv"), read("v13_summary.csv"), read("v13_contrasts.csv")
rng = np.random.default_rng(0)
C = {(r["family"], int(r["K"]), r["subset"], r["contrast"]): r for r in con}
Ks = sorted({int(r["K"]) for r in summ})

# 1. HEADLINE: gap vs measurement settings ------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.axhline(0, color=MUTED, lw=1)
for K in Ks:
    t = "pooled" if K == 5 else "all"
    pts = [(SETTINGS[w], C[("secondary", K, t, f"gap_w{w}")]) for w in WIDTHS
           if ("secondary", K, t, f"gap_w{w}") in C and int(C[("secondary", K, t, f"gap_w{w}")]["n"]) > 1]
    if not pts:
        continue
    x = np.array([p[0] for p in pts]); d = np.array([f(p[1]["diff"]) for p in pts])
    lo = np.array([f(p[1]["lo"]) for p in pts]); hi = np.array([f(p[1]["hi"]) for p in pts])
    ax.errorbar(x * (0.96 if K == 5 else 1.04), d, yerr=[d - lo, hi - d], color=KCOL[K], marker="o",
                ms=8, lw=2, capsize=3, markeredgecolor="white", markeredgewidth=1.5, label=f"K = {K}")
ax.axhspan(-1, 1, color=GRID, lw=0, label="+/- 1 pp equivalence margin")
ax.set_xscale("log"); ax.set_xticks([1, 2, 9]); ax.set_xticklabels(["1\nZ (w3)", "2\nZ+X (w6)", "9\nall (w15)"])
ax.minorticks_off()
ax.set_xlabel("Measurement settings per input (hardware shot cost)")
ax.set_ylabel("M3 - M2 (pp): cost of the quantum back-end")
ax.set_title("Does widening the readout close the back-end gap?  (mean, 95% CI)", color=INK, loc="left", fontsize=10)
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout(); fig.savefig(OUT / "gap_vs_settings.png", dpi=200); plt.close(fig)

# 2. accuracy vs width ---------------------------------------------------------
fig, axes = plt.subplots(1, len(Ks), figsize=(5.2 * len(Ks), 4.3), squeeze=False)
for ax, K in zip(axes[0], Ks):
    for m in ("M2", "M3"):
        pts = sorted((int(s["W"]), f(s["mean"]), f(s["sd"]), int(s["n"])) for s in summ
                     if int(s["K"]) == K and s["model"] == m and s["subset"] in HEAD)
        if not pts:
            continue
        w, mu, sd, n = map(np.array, zip(*pts))
        x = [WIDTHS.index(v) for v in w]
        ax.errorbar(x, mu, yerr=1.96 * np.nan_to_num(sd) / np.sqrt(n), color=COL[m], marker="o", ms=8, lw=2,
                    capsize=3, markeredgecolor="white", markeredgewidth=1.5,
                    label={"M2": "M2 quantum head", "M3": "M3 classical head (width-matched)"}[m])
    ax.set_xticks(range(3)); ax.set_xticklabels([WLAB[w] for w in WIDTHS], fontsize=8)
    ax.set_title(f"K = {K}", color=INK, loc="left", fontsize=10)
    ax.set_ylabel("Test accuracy (%)")
axes[0][0].legend(fontsize=8, loc="lower right")
fig.suptitle("Accuracy vs readout width (mean, 95% CI)", x=0.02, ha="left", color=INK, fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "acc_vs_width.png", dpi=200); plt.close(fig)

# 3. primary paired differences per K -----------------------------------------
for K in Ks:
    t = "pooled" if K == 5 else "all"
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for i, c in enumerate(PRIMARY):
        r = C.get(("primary", K, t, c))
        if not r or int(r["n"]) < 2:
            continue
        d = seeds(r)
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, s=36, color=PCOL[c], edgecolor="white", lw=1, zorder=3)
        ax.errorbar(i + 0.32, f(r["diff"]), yerr=[[f(r["diff"]) - f(r["lo"])], [f(r["hi"]) - f(r["diff"])]],
                    color=INK, marker="o", ms=6, capsize=3, lw=1.5)
        ax.text(i + 0.36, f(r["hi"]), f"  p_holm={f(r['p_holm']):.2g}\n  MDE={f(r['mde']):.1f}", fontsize=7,
                color=MUTED, va="bottom")
    ax.axhline(0, color=MUTED, lw=1)
    ax.axhspan(-1, 1, color=GRID, lw=0)
    ax.set_xticks(range(3)); ax.set_xticklabels([PTITLE[c] for c in PRIMARY], fontsize=8)
    ax.set_ylabel("Paired difference (pp)")
    ax.set_title(f"K={K}: primary contrasts, per-seed differences, mean + 95% CI", color=INK, loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / f"primary_k{K}.png", dpi=200); plt.close(fig)

# 4. strip per K ---------------------------------------------------------------
by = defaultdict(list)
for r in runs:
    by[(int(r["K"]), r["subset"], int(r["W"]), r["model"])].append(f(r["test_acc"]))
for K in Ks:
    tags = sorted({t for (k, t, _, _) in by if k == K})
    cols = [(m, w) for w in WIDTHS for m in ("M1", "M2", "M3", "M4") if any(by.get((K, t, w, m)) for t in tags)]
    fig, axes = plt.subplots(1, len(tags), figsize=(max(5, 0.75 * len(cols)) * len(tags), 4.2), sharey=True, squeeze=False)
    for ax, t in zip(axes[0], tags):
        for i, (m, w) in enumerate(cols):
            v = np.array(by.get((K, t, w, m), []))
            if len(v):
                ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=30, color=COL[m], edgecolor="white", lw=1, zorder=3)
                ax.hlines(v.mean(), i - 0.25, i + 0.25, color=INK, lw=2, zorder=4)
        ax.set_xticks(range(len(cols))); ax.set_xticklabels([f"{m}\nw{w}" for m, w in cols], fontsize=8)
        ax.set_title(f"K={K}, subset {t}", color=INK, loc="left", fontsize=10)
    axes[0][0].set_ylabel("Test accuracy (%)")
    fig.suptitle("Per-seed test accuracy (black bar = mean)", x=0.02, ha="left", color=INK, fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / f"strip_k{K}.png", dpi=200); plt.close(fig)

# 5. effect vs MDE -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.8, 5.2)); mx = 1.0
for fam, mk in (("primary", "o"), ("secondary", "s"), ("k10_completion", "D")):
    for r in con:
        if r["family"] == fam and r["subset"] in HEAD and np.isfinite(f(r["mde"])):
            e, m = abs(f(r["diff"])), f(r["mde"]); mx = max(mx, e, m)
            ax.scatter(m, e, s=50, marker=mk, color=KCOL[int(r["K"])], edgecolor="white", lw=1.2, zorder=3)
            ax.annotate(r["contrast"], (m, e), xytext=(4, 2), textcoords="offset points", fontsize=6, color=MUTED)
ax.plot([0, mx * 1.1], [0, mx * 1.1], ls="--", color=MUTED, lw=1)
ax.text(mx * 0.55, mx * 0.25, "below the line: smaller than\nthis design can detect", fontsize=8, color=MUTED)
for K in Ks:
    ax.scatter([], [], color=KCOL[K], label=f"K = {K}")
for fam, mk in (("primary", "o"), ("secondary", "s"), ("K=10 completion", "D")):
    ax.scatter([], [], marker=mk, color=MUTED, label=fam)
ax.set_xlabel("MDE (pp, 80% power, alpha = 0.05)"); ax.set_ylabel("|observed difference| (pp)")
ax.set_title("Effect size vs minimum detectable effect", color=INK, loc="left", fontsize=10)
ax.legend(fontsize=7, loc="upper left")
fig.tight_layout(); fig.savefig(OUT / "effect_vs_mde.png", dpi=200); plt.close(fig)

# 6. SD vs width ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 4.2))
for K in Ks:
    for m, ls in (("M2", "-"), ("M3", "--")):
        pts = sorted((int(s["W"]), f(s["sd"])) for s in summ
                     if int(s["K"]) == K and s["model"] == m and s["subset"] in HEAD)
        if pts:
            w, sd = zip(*pts)
            ax.plot([WIDTHS.index(v) for v in w], sd, ls=ls, marker="o", color=KCOL[K], lw=2, label=f"K={K} {m}")
ax.set_xticks(range(3)); ax.set_xticklabels([WLAB[w] for w in WIDTHS], fontsize=8)
ax.set_ylabel("Seed-to-seed SD (pp)")
ax.set_title("Stability vs readout width (solid M2 quantum head, dashed M3)", color=INK, loc="left", fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(OUT / "sd_vs_width.png", dpi=200); plt.close(fig)

# 7. K=10 completion -----------------------------------------------------------
rows = [r for r in con if r["family"] == "k10_completion" and int(r["n"]) > 1]
if rows:
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for i, r in enumerate(rows):
        d = seeds(r)
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, s=36, color=list(COL.values())[i], edgecolor="white", lw=1, zorder=3)
        ax.errorbar(i + 0.32, f(r["diff"]), yerr=[[f(r["diff"]) - f(r["lo"])], [f(r["hi"]) - f(r["diff"])]],
                    color=INK, marker="o", ms=6, capsize=3, lw=1.5)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([r["contrast"] for r in rows])
    ax.set_ylabel("Paired difference (pp)")
    ax.set_title(f"K=10, width 3, n={rows[0]['n']} seeds: v12 contrasts completed", color=INK, loc="left", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "k10_completion.png", dpi=200); plt.close(fig)

print("Wrote figures_v13/:", ", ".join(sorted(p.name for p in OUT.glob("*.png"))))
