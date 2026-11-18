#!/usr/bin/env python3
"""
v15.8 analysis (PREREGISTRATION_v15_8.md). Run from the v15.8 repo root after the grid gate passes:

    python3 scripts/analyse_v158.py | tee analyse_v158.log

M6 from this repo; M3, M4, M5 from the stored v15.5 runs (same seeds, same noisy test sets; the
gate checked the pairing). Accuracies in percent, contrasts in percentage points (pp).

Arms: DEPLOYED = trained clean (sigma_train = 0), cross-evaluated at every sigma_test.
      MATCHED  = trained at sigma_train = 0.3, tested at 0.3.
Per K: K = 5 is the per-seed mean of subsets A and B (as in v15.5); K = 10 as is.

PRIMARY (sigma_test = 0.3, each K x arm, Holm over {G, H}):
    G = M3 - M6   trained circuit vs fixed cos(pi x)
    H = M6 - M4   fixed cos(pi x) vs no front-end
SECONDARY / EXPLORATORY (uncorrected, with MDE): F = M3 - M4, M6 - M5, every sigma_test,
per subset, clean accuracy, salt-and-pepper.
Paired t over seeds; 95% CI; MDE at 80% power; TOST +/-1 pp.
Writes v158_out/: runs CSV, contrasts CSV, summary, standard plot set (PNG 200 dpi), this script.
"""
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
OUT = ROOT / "v158_out"
MARGIN = 1.0
ALPHA, POWER = 0.05, 0.80


def cfg():
    names = ["GRID_KS", "GRID_SIGMAS", "EVAL_SIGMAS", "EVAL_SP", "GRID_SEEDS", "V155_ROOT"]
    cmd = "cd '%s/pbs' && source ./config158.sh && " % ROOT + " && ".join('echo "${%s}"' % n for n in names)
    o = subprocess.run(["bash", "-c", cmd], stdout=subprocess.PIPE, universal_newlines=True, check=True).stdout.split("\n")
    return ([int(x) for x in o[0].split()], o[1].split(), [float(x) for x in o[2].split()],
            [float(x) for x in o[3].split()], [int(x) for x in o[4].split()], Path(o[5]))


KS, SIGMAS, EVALS, EVAL_SP, SEEDS, V155 = cfg()
TAGS = {5: ["A", "B"], 10: ["all"]}
MODELS = ["M3", "M4", "M5", "M6"]
COL = {"M3": "#3b6fb6", "M4": "#8a8a8a", "M5": "#c9822b", "M6": "#2e9e6a"}
CON = {"G": ("M3", "M6"), "H": ("M6", "M4"), "F": ("M3", "M4"), "M6-M5": ("M6", "M5")}
PRIMARY = ["G", "H"]

if not (ROOT / "pbs" / ".grid_ok").exists():
    sys.exit("pbs/.grid_ok missing: the grid gate has not passed. Do not read results.")
OUT.mkdir(exist_ok=True)

# ---- load ---------------------------------------------------------------------------
# A[(M, K, TAG, sg_train, seed)] = {"g0.3": pct, ..., "sp0.1": pct}
A, rows = {}, []
for K in KS:
    for TAG in TAGS[K]:
        for SG in SIGMAS:
            for M in MODELS:
                for S in SEEDS:
                    base = (ROOT / ("results_v158_k%d_%s_s%s" % (K, TAG, SG))) if M == "M6" else \
                        (V155 / ("results_v155_k%d_%s_s%s" % (K, TAG, SG)))
                    f = base / ("v155_xeval_%s_seed%d.json" % (M, S))
                    if not f.exists():
                        sys.exit("missing %s (the gate should have caught this)" % f)
                    s = json.loads(f.read_text())
                    if s.get("selfcheck") != "MATCH":
                        sys.exit("%s not MATCH" % f)
                    d = {"g%g" % float(k): 100.0 * v for k, v in s["acc"].items()}
                    d.update({"sp%g" % float(k): 100.0 * v for k, v in s["acc_sp"].items()})
                    A[(M, K, TAG, SG, S)] = d
                    for lv, v in sorted(d.items()):
                        rows.append(dict(model=M, K=K, subset=TAG, sigma_train=SG, seed=S, test_level=lv, acc_pct="%.4f" % v,
                                         source="v15.8" if M == "M6" else "v15.5"))
with open(OUT / "v158_runs.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print("Loaded %d runs (M6 new; M3/M4/M5 stored v15.5), seeds %s." % (len(A), SEEDS))


def vec(M, K, SG, lv, tag=None):
    """Per-seed accuracy vector; K = 5 pooled = mean of A and B unless tag is given."""
    tags = [tag] if tag else TAGS[K]
    return np.array([np.mean([A[(M, K, t, SG, S)][lv] for t in tags]) for S in SEEDS])


def arm_sg(arm):
    return "0" if arm == "deployed" else "0.3"


def test(d):
    n = len(d)
    m, sd = float(np.mean(d)), float(np.std(d, ddof=1))
    se = sd / np.sqrt(n)
    tc = stats.t.ppf(1 - ALPHA / 2, n - 1)
    p = float(stats.ttest_1samp(d, 0).pvalue) if sd > 0 else (0.0 if m != 0 else 1.0)
    mde = (tc + stats.t.ppf(POWER, n - 1)) * se
    if sd > 0:
        p_lo = 1 - stats.t.cdf((m + MARGIN) / se, n - 1)
        p_hi = stats.t.cdf((m - MARGIN) / se, n - 1)
        tost = float(max(p_lo, p_hi))
    else:
        tost = 0.0 if abs(m) < MARGIN else 1.0
    return dict(n=n, mean=m, sd=sd, lo=m - tc * se, hi=m + tc * se, p=p, mde=mde, tost=tost,
                wins=int((d > 0).sum()), losses=int((d < 0).sum()))


def holm(ps):
    order = np.argsort(ps)
    adj, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def fmt(r, adj=None):
    s = "%+6.2f pp [%+6.2f, %+6.2f]  p=%.4f" % (r["mean"], r["lo"], r["hi"], r["p"])
    if adj is not None:
        s += "  Holm p=%.4f" % adj
    s += "  MDE=%.2f  %d/%d seeds >0" % (r["mde"], r["wins"], r["n"])
    if r["hi"] < MARGIN and r["lo"] > -MARGIN:
        s += "  TOST(+/-1) p=%.4f" % r["tost"]
    return s


con_rows, prim = [], {}
print("\n" + "=" * 92)
print("PRIMARY: sigma_test = 0.3, paired over seeds %s (n=%d); Holm over {G, H} within each K x arm" % (SEEDS, len(SEEDS)))
print("  G = M3 - M6 (trained circuit vs fixed cos)   H = M6 - M4 (fixed cos vs no front-end)")
print("=" * 92)
for K in KS:
    for arm in ("deployed", "matched"):
        SG = arm_sg(arm)
        res = {c: test(vec(CON[c][0], K, SG, "g0.3") - vec(CON[c][1], K, SG, "g0.3")) for c in PRIMARY}
        adj = holm([res[c]["p"] for c in PRIMARY])
        means = {M: vec(M, K, SG, "g0.3").mean() for M in MODELS}
        print("\nK=%d %s  (means: %s)" % (K, arm, "  ".join("%s %.2f" % (M, means[M]) for M in MODELS)))
        for c, a in zip(PRIMARY, adj):
            print("  %-2s %s" % (c, fmt(res[c], a)))
            res[c]["holm"] = a
        g, h = res["G"], res["H"]
        sig = lambda r: r["holm"] < ALPHA  # noqa: E731
        if sig(h) and h["mean"] > 0 and not sig(g):
            rd = "H > 0, G not detected -> the fixed cos encoding reproduces the robustness (classical)"
        elif sig(h) and h["mean"] > 0 and sig(g) and g["mean"] > 0:
            rd = "H > 0, G > 0 -> the encoding explains part; the trained map adds more"
        elif not sig(h) and sig(g) and g["mean"] > 0:
            rd = "H not detected, G > 0 -> needs the trained mixing"
        elif sig(g) and g["mean"] < 0:
            rd = "G < 0 -> the fixed delta = 0 encoding is MORE robust than the trained circuit"
        else:
            rd = "neither contrast detected at this n (report MDEs: G %.2f, H %.2f pp)" % (g["mde"], h["mde"])
        print("  reading (pre-registered table): " + rd)
        prim[(K, arm)] = res
        for c in PRIMARY:
            con_rows.append(dict(tier="primary", contrast=c, K=K, subset="pooled" if K == 5 else "all", arm=arm,
                                 sigma_test=0.3, **{k: ("%.5g" % v if isinstance(v, float) else v) for k, v in res[c].items()}))

print("\n" + "=" * 92)
print("PREDICTIONS (written before results)")
print("=" * 92)
p1 = [(K, prim[(K, "deployed")]["H"]) for K in KS]
print("P1 H > 0 deployed @0.3 in every K: %s" % ("; ".join(
    "K=%d %+.2f (Holm p=%.4f) %s" % (K, r["mean"], r["holm"], "held" if (r["mean"] > 0 and r["holm"] < ALPHA) else "not shown")
    for K, r in p1)))
print("P2 G <= 0 deployed @0.3: %s" % ("; ".join(
    "K=%d %+.2f [%+.2f, %+.2f] %s" % (K, r["mean"], r["lo"], r["hi"],
                                     "held (CI excludes G > 0)" if r["hi"] <= 0 else
                                     ("contradicted (G > 0 detected)" if (r["lo"] > 0) else "not decided"))
    for K, r in ((K, prim[(K, "deployed")]["G"]) for K in KS))))
print("P3 clean (sigma_test = 0), M6 - M4 within +/-1.5 pp: %s" % ("; ".join(
    "K=%d %+.2f [%+.2f, %+.2f] %s" % (K, r["mean"], r["lo"], r["hi"], "held" if (r["lo"] > -1.5 and r["hi"] < 1.5) else "not shown")
    for K, r in ((K, test(vec("M6", K, "0", "g0") - vec("M4", K, "0", "g0"))) for K in KS))))

print("\n" + "=" * 92)
print("SECONDARY (uncorrected): every contrast, every sigma_test, deployed arm; and matched @0.3")
print("=" * 92)
for K in KS:
    print("\nK=%d deployed (trained clean)" % K)
    print("  %-6s" % "sigma" + "".join("%-34s" % c for c in CON))
    for lv in EVALS:
        cells = []
        for c, (a, b) in CON.items():
            r = test(vec(a, K, "0", "g%g" % lv) - vec(b, K, "0", "g%g" % lv))
            cells.append("%+6.2f [%+6.2f,%+6.2f] MDE %4.1f" % (r["mean"], r["lo"], r["hi"], r["mde"]))
            con_rows.append(dict(tier="secondary", contrast=c, K=K, subset="pooled" if K == 5 else "all", arm="deployed",
                                 sigma_test=lv, **{k: ("%.5g" % v if isinstance(v, float) else v) for k, v in r.items()}))
        print("  %-6g" % lv + "".join("%-34s" % x for x in cells))
    print("K=%d matched @0.3: " % K + "  ".join(
        "%s %+.2f [%+.2f,%+.2f]" % (c, r["mean"], r["lo"], r["hi"])
        for c, r in ((c, test(vec(a, K, "0.3", "g0.3") - vec(b, K, "0.3", "g0.3"))) for c, (a, b) in CON.items())))

print("\nPer subset, deployed @0.3 (secondary):")
for K in KS:
    for TAG in TAGS[K]:
        print("  K=%d %-3s " % (K, TAG) + "  ".join(
            "%s %+.2f [%+.2f,%+.2f]" % (c, r["mean"], r["lo"], r["hi"]) for c, r in
            ((c, test(vec(a, K, "0", "g0.3", TAG) - vec(b, K, "0", "g0.3", TAG))) for c, (a, b) in CON.items())))

print("\nEXPLORATORY: salt-and-pepper, deployed arm:")
for K in KS:
    print("  K=%d " % K + "  ".join(
        "p=%g G %+.2f H %+.2f" % (p, test(vec("M3", K, "0", "sp%g" % p) - vec("M6", K, "0", "sp%g" % p))["mean"],
                                  test(vec("M6", K, "0", "sp%g" % p) - vec("M4", K, "0", "sp%g" % p))["mean"])
        for p in EVAL_SP))

with open(OUT / "v158_contrasts.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(con_rows[0]))
    w.writeheader(); w.writerows(con_rows)

# ---- standard plot set --------------------------------------------------------------
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
panels = [(K, arm) for K in KS for arm in ("deployed", "matched")]


def ptitle(K, arm):
    return "K=%d, %s (σ_test = 0.3)" % (K, "trained clean" if arm == "deployed" else "trained at σ = 0.3")


# 1. per-seed strip
fig, axs = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.2), sharey=False)
for ax, (K, arm) in zip(axs, panels):
    SG = arm_sg(arm)
    V = {M: vec(M, K, SG, "g0.3") for M in MODELS}
    for i, M in enumerate(MODELS):
        jit = np.linspace(-0.12, 0.12, len(SEEDS))
        ax.scatter(i + jit, V[M], s=18, color=COL[M], zorder=3)
        ax.hlines(V[M].mean(), i - 0.25, i + 0.25, color="k", lw=1.2, zorder=4)
    for si in range(len(SEEDS)):
        ax.plot(range(len(MODELS)), [V[M][si] for M in MODELS], color="0.8", lw=0.6, zorder=1)
    ax.set_xticks(range(len(MODELS))); ax.set_xticklabels(MODELS)
    ax.set_title(ptitle(K, arm), fontsize=8); ax.set_ylabel("test accuracy (%)")
fig.suptitle("v15.8: per-seed accuracy (lines join the same seed)", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig1_strip.png", dpi=200); plt.close(fig)

# 2. bar chart, mean +/- 95% CI
fig, axs = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.2))
for ax, (K, arm) in zip(axs, panels):
    SG = arm_sg(arm)
    ms, es = [], []
    for M in MODELS:
        v = vec(M, K, SG, "g0.3")
        ms.append(v.mean()); es.append(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)))
    ax.bar(range(len(MODELS)), ms, yerr=es, color=[COL[M] for M in MODELS], capsize=3)
    lo = min(m - e for m, e in zip(ms, es))
    ax.set_ylim(max(0, lo - 5), min(100, max(m + e for m, e in zip(ms, es)) + 2))
    ax.set_xticks(range(len(MODELS))); ax.set_xticklabels(MODELS)
    ax.set_title(ptitle(K, arm), fontsize=8); ax.set_ylabel("mean accuracy (%) ± 95% CI")
fig.tight_layout(); fig.savefig(OUT / "fig2_bar.png", dpi=200); plt.close(fig)

# 3. per-seed paired differences per contrast
fig, axs = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.2))
for ax, (K, arm) in zip(axs, panels):
    SG = arm_sg(arm)
    for i, (c, (a, b)) in enumerate(CON.items()):
        d = vec(a, K, SG, "g0.3") - vec(b, K, SG, "g0.3")
        r = test(d)
        ax.scatter(i + np.linspace(-0.1, 0.1, len(d)), d, s=16, color="k" if c in PRIMARY else "0.5", zorder=3)
        ax.errorbar(i + 0.25, r["mean"], yerr=[[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]], fmt="o", ms=4,
                    color="C3" if c in PRIMARY else "C7", capsize=3)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(CON))); ax.set_xticklabels(["%s\n%s−%s" % (c, *CON[c]) if c != "M6-M5" else "M6−M5" for c in CON], fontsize=7)
    ax.set_title(ptitle(K, arm), fontsize=8); ax.set_ylabel("paired difference (pp)")
fig.suptitle("Per-seed paired differences; red = primary (mean ± 95% CI)", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig3_paired.png", dpi=200); plt.close(fig)

# 4. effect size vs MDE
fig, ax = plt.subplots(figsize=(5.2, 3.6))
labels, eff, mdes = [], [], []
for K, arm in panels:
    SG = arm_sg(arm)
    for c, (a, b) in CON.items():
        r = test(vec(a, K, SG, "g0.3") - vec(b, K, SG, "g0.3"))
        labels.append("K%d %s %s" % (K, arm[:3], c)); eff.append(abs(r["mean"])); mdes.append(r["mde"])
y = np.arange(len(labels))
ax.barh(y + 0.2, eff, height=0.4, color="C0", label="|mean effect|")
ax.barh(y - 0.2, mdes, height=0.4, color="C1", alpha=0.7, label="MDE (80% power)")
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=6); ax.invert_yaxis()
ax.set_xlabel("pp at σ_test = 0.3"); ax.legend(fontsize=7, frameon=False)
ax.set_title("Effect size vs minimum detectable effect (n = %d seeds)" % len(SEEDS), fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "fig4_effect_vs_mde.png", dpi=200); plt.close(fig)

# 5. deployed accuracy vs sigma_test
fig, axs = plt.subplots(1, len(KS), figsize=(4.2 * len(KS), 3.3))
for ax, K in zip(np.atleast_1d(axs), KS):
    for M in MODELS:
        mu = [vec(M, K, "0", "g%g" % lv).mean() for lv in EVALS]
        ci = [stats.t.ppf(0.975, len(SEEDS) - 1) * vec(M, K, "0", "g%g" % lv).std(ddof=1) / np.sqrt(len(SEEDS)) for lv in EVALS]
        ax.errorbar(EVALS, mu, yerr=ci, marker="o", ms=3, capsize=2, color=COL[M], label=M)
    ax.axvline(0.3, color="0.7", ls=":", lw=0.8)
    ax.set_xlabel("σ_test (pixel noise)"); ax.set_ylabel("accuracy (%)")
    ax.set_title("K=%d, trained clean (deployed)" % K, fontsize=9); ax.legend(fontsize=7, frameon=False)
fig.tight_layout(); fig.savefig(OUT / "fig5_deployed_curves.png", dpi=200); plt.close(fig)

shutil.copy2(__file__, OUT / "analyse_v158.py")
print("\nWrote %s: v158_runs.csv, v158_contrasts.csv, fig1_strip, fig2_bar, fig3_paired, fig4_effect_vs_mde, "
      "fig5_deployed_curves (PNG 200 dpi), analyse_v158.py" % OUT)
