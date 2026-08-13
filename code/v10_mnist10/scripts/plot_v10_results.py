#!/usr/bin/env python3
"""Figures for the MNIST 10-class (v10) grid.

Reads results_v10_mnist10/*.json and writes PNGs to figures/.
Falls back to the hardcoded 2026-08-13 values if the directory is absent,
so the figures can be regenerated anywhere.

    python scripts/plot_v10_results.py [results_dir]
"""
import sys, json, glob
from pathlib import Path
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SEEDS = [42, 43, 44, 45, 46]
MODELS = ["M1", "M2", "M3", "M4"]
LABEL = {
    "M1": "M1\nfrozen quanv\n+ QNN head",
    "M2": "M2\ntrainable C4\n+ QNN head",
    "M3": "M3\ntrainable C4\n+ classical",
    "M4": "M4\nbilinear resize\n+ classical",
}
# QNN head vs classical head
COL = {"M1": "#B4472A", "M2": "#C25E3F", "M3": "#3E6E8E", "M4": "#5A8CA8"}

FALLBACK = {
    "M1": {42: .6470, 43: .7435, 44: .7115, 45: .7670, 46: .6605},
    "M2": {42: .7775, 43: .6275, 44: .7255, 45: .7030, 46: .7125},
    "M3": {42: .7865, 43: .8335, 44: .8155, 45: .8485, 46: .8140},
    "M4": {42: .7910, 43: .8355, 44: .8225, 45: .8415, 46: .8300},
}


def load(results_dir):
    if not results_dir or not Path(results_dir).is_dir():
        print("Using fallback values (2026-08-13 grid)")
        return FALLBACK
    acc = {m: {} for m in MODELS}
    for m in MODELS:
        for f in sorted(glob.glob(f"{results_dir}/result_{m}_*.json")):
            d = json.loads(Path(f).read_text())
            acc[m][d["seed"]] = d["test_acc"]
    if not all(len(acc[m]) == 5 for m in MODELS):
        raise SystemExit(f"Expected 5 seeds per model, got "
                         f"{ {m: len(acc[m]) for m in MODELS} }")
    return acc


def mde(diff):
    n = len(diff)
    sd = diff.std(ddof=1)
    return (stats.t.ppf(.975, n - 1) + stats.t.ppf(.80, n - 1)) * sd / np.sqrt(n)


def fig1_spread(acc, out):
    """The headline figure: seed spread per model."""
    fig, ax = plt.subplots(figsize=(9, 6))
    for i, m in enumerate(MODELS):
        v = np.array([acc[m][s] for s in SEEDS]) * 100
        ax.plot([i - .28, i + .28], [v.mean()] * 2, color=COL[m], lw=3, zorder=3)
        ax.vlines(i, v.min(), v.max(), color=COL[m], lw=1.2, alpha=.5, zorder=1)
        jit = np.linspace(-.13, .13, len(SEEDS))
        ax.scatter([i + j for j in jit], v, s=95, color=COL[m],
                   edgecolor="white", lw=1.4, zorder=4)
        for j, s in zip(jit, SEEDS):
            ax.annotate(str(s), (i + j, acc[m][s] * 100), fontsize=6.5,
                        color="#444", ha="center", va="center", zorder=5)
        ax.annotate(f"spread\n{v.max()-v.min():.1f} pp", (i, v.max() + 1.4),
                    ha="center", fontsize=9, color=COL[m], weight="bold")

    ax.axvspan(-.5, 1.5, color="#B4472A", alpha=.055, zorder=0)
    ax.axvspan(1.5, 3.5, color="#3E6E8E", alpha=.055, zorder=0)
    ax.text(.5, 61.2, "QNN head", ha="center", fontsize=10.5,
            color="#B4472A", weight="bold")
    ax.text(2.5, 61.2, "classical head", ha="center", fontsize=10.5,
            color="#3E6E8E", weight="bold")

    ax.set_xticks(range(4))
    ax.set_xticklabels([LABEL[m] for m in MODELS], fontsize=9)
    ax.set_xlim(-.5, 3.5)
    ax.set_ylim(60, 87)
    ax.set_ylabel("Test accuracy (%)", fontsize=11)
    ax.set_title("MNIST 10-class: every seed, every model\n"
                 "Each point is one run; the number inside is its seed",
                 fontsize=12.5, weight="bold", pad=14)
    ax.grid(axis="y", alpha=.25, ls=":")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.text(.5, .012,
             "The two QNN-head models scatter across 12-15 pp; the classical-head "
             "models across 5-6 pp.\nThat scatter is what makes the M2-M1 "
             "comparison unanswerable at 5 seeds.",
             ha="center", fontsize=9, color="#555")
    fig.tight_layout(rect=[0, .05, 1, 1])
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig2_paired(acc, out):
    """Per-seed paired differences for the three attribution contrasts."""
    pairs = [("M2", "M1", "front-end trainability"),
             ("M2", "M3", "quantum back-end"),
             ("M3", "M4", "quantum front-end")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.2), sharey=True)
    for ax, (a, b, what) in zip(axes, pairs):
        d = np.array([acc[a][s] - acc[b][s] for s in SEEDS]) * 100
        m_ = mde(d / 100) * 100
        t, p = stats.ttest_rel([acc[a][s] for s in SEEDS],
                               [acc[b][s] for s in SEEDS])
        ax.axhspan(-m_, m_, color="#999", alpha=.16, zorder=0)
        ax.axhline(0, color="#333", lw=1, zorder=1)
        cols = ["#2E7D32" if x > 0 else "#B4472A" for x in d]
        ax.bar(range(5), d, color=cols, width=.62, zorder=2)
        for i, x in enumerate(d):
            ax.annotate(f"{x:+.1f}", (i, x + (.9 if x > 0 else -1.9)),
                        ha="center", fontsize=9, weight="bold")
        ax.axhline(d.mean(), color="#111", ls="--", lw=1.6, zorder=3)
        ax.set_xticks(range(5))
        ax.set_xticklabels(SEEDS)
        ax.set_xlabel("seed", fontsize=10)
        ax.set_title(f"{a} - {b}\n{what}\nmean {d.mean():+.2f} pp, p = {p:.3f}",
                     fontsize=11, weight="bold")
        ax.grid(axis="y", alpha=.22, ls=":")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("difference in test accuracy (pp)", fontsize=11)
    axes[0].legend(handles=[Patch(color="#999", alpha=.3,
                                  label="within MDE — undetectable")],
                   loc="lower left", fontsize=8.5, framealpha=.9)
    fig.suptitle("Per-seed differences. Grey band = minimum detectable effect",
                 fontsize=13, weight="bold", y=1.0)
    fig.text(.5, .005,
             "M2-M1 flips sign across seeds: +13.1, -11.6, +1.4, -6.4, +5.2. "
             "The seed matters more than the architecture.",
             ha="center", fontsize=9.5, color="#555")
    fig.tight_layout(rect=[0, .04, 1, .97])
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig3_mde(acc, out):
    """Effect vs MDE, with the binary grids for context."""
    rows = []
    for a, b in [("M2", "M1"), ("M2", "M3"), ("M3", "M4")]:
        d = np.array([acc[a][s] - acc[b][s] for s in SEEDS]) * 100
        rows.append((f"{a}-{b}\n(10-class)", abs(d.mean()), mde(d / 100) * 100))
    # Binary grids, from FMNIST_LOG.md
    rows += [("M2-M1\n(FMNIST 0,6)", 0.62, 1.4),
             ("M2-M1\n(FMNIST 2,6)", 1.04, 2.3),
             ("M2-M1\n(FMNIST 2,4)", 0.30, 2.8)]

    fig, ax = plt.subplots(figsize=(11, 5.6))
    y = np.arange(len(rows))
    for i, (lab, eff, m_) in enumerate(rows):
        new = i < 3
        ax.barh(i, m_, color="#BBB", height=.62, zorder=1,
                label="MDE (detectable at 80% power)" if i == 0 else None)
        ax.barh(i, eff, color="#B4472A" if new else "#8899AA", height=.34,
                zorder=2, label="observed |effect|" if i == 0 else None)
        ax.annotate(f"MDE {m_:.1f}", (m_ + .25, i), va="center", fontsize=9,
                    color="#666")
        ax.annotate(f"{eff:.2f}", (eff + .25, i - .30), va="center",
                    fontsize=9, color="#B4472A" if new else "#667", weight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("percentage points", fontsize=11)
    ax.set_title("Every effect is smaller than what the design can detect\n"
                 "and the 10-class design detects far LESS than the binary ones",
                 fontsize=12.5, weight="bold", pad=12)
    ax.legend(fontsize=9.5, loc="lower right")
    ax.grid(axis="x", alpha=.25, ls=":")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.axhline(2.5, color="#333", lw=1, ls="--", alpha=.6)
    ax.annotate("10-class (v10)", (16.4, 0.15), fontsize=9.5,
                color="#B4472A", weight="bold")
    ax.annotate("binary (v7-v9)", (16.4, 3.15), fontsize=9.5,
                color="#667", weight="bold")
    fig.text(.5, .012,
             "The 10-class experiment was run to gain power. Its MDE is 16.1 pp "
             "against 1.4-2.8 pp for the binary grids.",
             ha="center", fontsize=9.5, color="#555")
    fig.tight_layout(rect=[0, .05, 1, 1])
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig4_head(acc, out):
    """The one solid result: head type, pooled."""
    q = np.array([acc[m][s] for m in ("M1", "M2") for s in SEEDS]) * 100
    c = np.array([acc[m][s] for m in ("M3", "M4") for s in SEEDS]) * 100
    t, p = stats.ttest_ind(q, c, equal_var=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.4),
                                   gridspec_kw={"width_ratios": [1, 1.25]})
    for i, (v, lab, col) in enumerate([(q, "QNN head\n(M1, M2)", "#B4472A"),
                                       (c, "classical head\n(M3, M4)", "#3E6E8E")]):
        ax1.bar(i, v.mean(), yerr=v.std(ddof=1), color=col, width=.55,
                capsize=8, error_kw={"lw": 1.6})
        ax1.scatter([i] * len(v) + np.random.RandomState(0).uniform(-.13, .13, len(v)),
                    v, s=42, color="white", edgecolor=col, lw=1.3, zorder=3)
        ax1.annotate(f"{v.mean():.1f}%", (i + .38, v.mean()),
                     ha="left", va="center", fontsize=11, weight="bold", color=col)
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["QNN head\n(M1, M2)", "classical head\n(M3, M4)"],
                        fontsize=10)
    ax1.set_ylabel("Test accuracy (%)", fontsize=11)
    ax1.set_ylim(55, 92); ax1.set_xlim(-.6, 1.75)
    ax1.set_title(f"Back-end: 11.4 pp gap\nWelch p = {p:.1e}",
                  fontsize=11.5, weight="bold")
    ax1.grid(axis="y", alpha=.25, ls=":")

    contrasts = [("quantum front-end\nM3 - M4", -0.45, 1.38),
                 ("front-end trainability\nM2 - M1", 0.33, 16.08),
                 ("quantum back-end\nM2 - M3", -11.04, 12.09)]
    for i, (lab, eff, m_) in enumerate(contrasts):
        ax2.barh(i, eff, color="#B4472A" if eff < -1 else "#AAA", height=.5)
        ax2.errorbar(0, i, xerr=m_, color="#333", capsize=6, lw=1.4, zorder=3)
        ax2.annotate(f"{eff:+.2f} pp", (eff + (-2.6 if eff < -1 else .6), i - .3),
                     fontsize=10, weight="bold")
    ax2.set_yticks(range(3))
    ax2.set_yticklabels([c[0] for c in contrasts], fontsize=10)
    ax2.axvline(0, color="#333", lw=1)
    ax2.set_xlabel("difference in test accuracy (pp)   "
                   "(bars = effect, whiskers = MDE)", fontsize=10)
    ax2.set_title("Only the back-end penalty exceeds noise",
                  fontsize=11.5, weight="bold")
    ax2.grid(axis="x", alpha=.25, ls=":")
    for ax in (ax1, ax2):
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.text(.5, .012,
             "The quantum front-end changes nothing measurable (M3-M4, tight MDE "
             "of 1.4 pp).\nThe quantum back-end costs 11 pp and destabilises "
             "training.",
             ha="center", fontsize=9.5, color="#555")
    fig.tight_layout(rect=[0, .06, 1, 1])
    fig.savefig(out, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    acc = load(sys.argv[1] if len(sys.argv) > 1 else "results_v10_mnist10")
    out = Path("figures"); out.mkdir(exist_ok=True)
    fig1_spread(acc, out / "v10_fig1_seed_spread.png")
    fig2_paired(acc, out / "v10_fig2_paired_differences.png")
    fig3_mde(acc, out / "v10_fig3_effect_vs_mde.png")
    fig4_head(acc, out / "v10_fig4_head_comparison.png")
    print("Wrote 4 figures to figures/")
