#!/usr/bin/env python3
"""analyse_v156.py <repo156> [--force]

Refuses unless <repo156>/.grid_ok exists (the gate passed), unless --force (for local testing only).
Writes into <repo156>/v156_out/:
  v156_metrics.csv     long table: cell, K, model, seed, arm, sigma_test, layer, metric, value
  v156_contrasts.csv   every paired contrast (F, T, R), both arms, all sigma, layers, metrics
  v156_primary.csv     the 4 pre-registered primaries per K (Holm within K)
  figures/             fig1 metric-vs-sigma grid, fig2 layer profile, fig3 z-plane regions,
                       standard set (strip, bars, paired, effect-vs-MDE), fig P6
"""
import csv
import glob
import json
import os
import sys

import numpy as np
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODELS = ("M3", "M4", "M5")
COL = {"M3": "#2f6fdb", "M4": "#c2410c", "M5": "#7c3aed"}
CELLS = (("k5_A", 5, "A"), ("k5_B", 5, "B"), ("k10_all", 10, "all"))
ARMS = (("0", "deployed"), ("0.3", "matched"))
LAYERS = ("Lpix", "L0", "L1", "L2", "L3")
SIGMAS = (0.0, 0.1, 0.2, 0.3, 0.5)
SEEDS = range(42, 52)
CONTRASTS = (("F", "M3", "M4"), ("T", "M3", "M5"), ("R", "M5", "M4"))
SEC_METRICS = ("eta", "rho_W", "rho_B", "Jtr_ret", "J_ret", "bleed")
PRIMARY = (("P-Feta", "F", "L0", "eta"), ("P-Teta", "T", "L0", "eta"),
           ("P-FJ", "F", "L2", "J_ret"), ("P-TJ", "T", "L2", "J_ret"))


def load(repo):
    tag = "v156" if glob.glob(os.path.join(repo, "results_v156_*")) else "v155"
    rows, npz = [], {}
    for cell, K, sub in CELLS:
        for arm, armname in ARMS:
            for M in MODELS:
                for S in SEEDS:
                    f = os.path.join(repo, "results_%s_%s_s%s" % (tag, cell, arm), "geom_%s_seed%d.npz" % (M, S))
                    if not os.path.exists(f):
                        continue
                    z = np.load(f)
                    npz[(cell, armname, M, S)] = f
                    for key, m in json.loads(str(z["metrics"])).items():
                        L, s = key.split("|")
                        for k, v in m.items():
                            rows.append(dict(cell=cell, K=K, subset=sub, model=M, seed=S, arm=armname,
                                             sigma_test=float(s), layer=L, metric=k, value=v))
    return rows, npz


def table(rows):
    """dict (Kgroup, arm, sigma, layer, metric, model) -> {seed: value}; K5 pooled over A and B per seed."""
    acc = {}
    for r in rows:
        kg = "K5" if r["K"] == 5 else "K10"
        key = (kg, r["arm"], r["sigma_test"], r["layer"], r["metric"], r["model"])
        acc.setdefault(key, {}).setdefault(r["seed"], []).append(r["value"])
    out = {}
    for key, d in acc.items():
        need = 2 if key[0] == "K5" else 1
        out[key] = dict((s, float(np.mean(v))) for s, v in d.items() if len(v) == need)
    return out


def paired(a, b):
    seeds = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in seeds])
    n = len(d)
    if n < 3 or not np.all(np.isfinite(d)):
        return None
    m, sd = d.mean(), d.std(ddof=1)
    df = n - 1
    se = sd / np.sqrt(n)
    t = m / se if se > 0 else np.inf * np.sign(m) if m else 0.0
    p = 2 * stats.t.sf(abs(t), df) if np.isfinite(t) else 0.0
    q = stats.t.ppf(0.975, df)
    mde = (q + stats.t.ppf(0.80, df)) * se
    return dict(n=n, mean=m, sd=sd, lo=m - q * se, hi=m + q * se, t=t, p=p, mde=mde,
                n_neg=int((d < 0).sum()), n_pos=int((d > 0).sum()))


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for i, j in enumerate(order):
        run = max(run, (len(ps) - i) * ps[j])
        adj[j] = min(1.0, run)
    return adj


def sin_delta_b(zfile):
    z = np.load(zfile)
    ws = [z[k].ravel() for k in z.files if k.startswith("fe__") and z[k].size == 6]
    if not ws:
        return np.nan
    w = ws[0].astype(float)
    d = w[1] + np.arctan2(np.sin(w[4]) * np.cos(w[2] / 2), np.cos(w[4]))
    return abs(np.sin(d))


def main():
    repo = sys.argv[1]
    if not os.path.exists(os.path.join(repo, ".grid_ok")) and "--force" not in sys.argv:
        sys.exit("REFUSED: no .grid_ok (run gate_v156.py grid first)")
    out = os.path.join(repo, "v156_out")
    fig = os.path.join(out, "figures")
    os.makedirs(fig, exist_ok=True)
    rows, npz = load(repo)
    with open(os.path.join(out, "v156_metrics.csv"), "w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    T = table(rows)

    # contrasts
    con = []
    for kg in ("K5", "K10"):
        for _, arm in ARMS:
            for s in SIGMAS[1:]:
                for L in LAYERS:
                    for met in SEC_METRICS:
                        if met == "J_ret" and L != "L2":
                            continue
                        for cn, a, b in CONTRASTS:
                            A, B = T.get((kg, arm, s, L, met, a)), T.get((kg, arm, s, L, met, b))
                            if not A or not B:
                                continue
                            r = paired(A, B)
                            if r:
                                r.update(K=kg, arm=arm, sigma=s, layer=L, metric=met, contrast=cn,
                                         mean_a=np.mean(list(A.values())), mean_b=np.mean(list(B.values())))
                                con.append(r)
    keys = ["K", "arm", "sigma", "layer", "metric", "contrast", "n", "mean_a", "mean_b", "mean", "sd", "lo", "hi",
            "t", "p", "mde", "n_neg", "n_pos"]
    with open(os.path.join(out, "v156_contrasts.csv"), "w") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in con:
            w.writerow(dict((k, r[k]) for k in keys))

    # primaries
    prim = []
    for kg in ("K5", "K10"):
        sel = []
        for pid, cn, L, met in PRIMARY:
            r = [c for c in con if c["K"] == kg and c["arm"] == "deployed" and c["sigma"] == 0.3
                 and c["layer"] == L and c["metric"] == met and c["contrast"] == cn]
            if r:
                sel.append(dict(r[0], id=pid))
        if sel:
            adj = holm(np.array([r["p"] for r in sel]))
            for r, pa in zip(sel, adj):
                r["p_holm"] = pa
                prim.append(r)
    pk = ["K", "id", "contrast", "layer", "metric", "n", "mean_a", "mean_b", "mean", "lo", "hi", "p", "p_holm", "mde",
          "n_neg", "n_pos"]
    with open(os.path.join(out, "v156_primary.csv"), "w") as f:
        w = csv.DictWriter(f, fieldnames=pk)
        w.writeheader()
        for r in prim:
            w.writerow(dict((k, r[k]) for k in pk))
    print("PRIMARY (deployed, sigma_test = 0.3; Holm within K)")
    for r in prim:
        print("  %-4s %-7s %s on %-5s at %-3s: %+.4f [%+.4f, %+.4f]  p=%.2g  p_holm=%.2g  MDE=%.4f  (%d/%d seeds < 0)"
              % (r["K"], r["id"], r["contrast"], r["metric"], r["layer"], r["mean"], r["lo"], r["hi"], r["p"],
                 r["p_holm"], r["mde"], r["n_neg"], r["n"]))

    def series(kg, arm, L, met, M):
        mu, ci = [], []
        for s in SIGMAS:
            d = T.get((kg, arm, s, L, met, M), {})
            v = np.array(list(d.values()))
            mu.append(v.mean() if len(v) else np.nan)
            ci.append(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0)
        return np.array(mu), np.array(ci)

    # fig1: metric vs sigma, every layer
    rowsm = (("eta", "noise shift  η"), ("rho_W", "noise spread / data spread  ρ_W"),
             ("FISHER", "Fisher retention  J_σ / J_0"), ("bleed", "bleed (wrong region)"))
    for kg in ("K5", "K10"):
        f1, ax = plt.subplots(len(rowsm), len(LAYERS), figsize=(18, 13), squeeze=False)
        for i, (met, lab) in enumerate(rowsm):
            for j, L in enumerate(LAYERS):
                mm = ("J_ret" if L == "L2" else "Jtr_ret") if met == "FISHER" else met
                for M in MODELS:
                    for arm, ls in (("deployed", "-"), ("matched", "--")):
                        mu, ci = series(kg, arm, L, mm, M)
                        ax[i, j].errorbar(SIGMAS, mu, yerr=ci, color=COL[M], ls=ls, marker="o", ms=3, capsize=2,
                                          label="%s %s" % (M, arm))
                if i == 0:
                    ax[i, j].set_title(L)
                if j == 0:
                    ax[i, j].set_ylabel(lab)
                if i == len(rowsm) - 1:
                    ax[i, j].set_xlabel("test σ")
                ax[i, j].grid(alpha=.3)
        ax[0, 0].legend(fontsize=7)
        f1.suptitle("v15.6 %s: noise geometry by layer (mean ± 95%% CI over seeds; solid = clean-trained, "
                    "dashed = σ=0.3-trained)" % kg)
        f1.tight_layout()
        f1.savefig(os.path.join(fig, "fig1_metric_vs_sigma_%s.png" % kg), dpi=200)
        plt.close(f1)

        # fig2: layer profile at sigma = 0.3
        f2, ax = plt.subplots(1, 4, figsize=(18, 4.2))
        for i, (met, lab) in enumerate(rowsm):
            for M in MODELS:
                for arm, ls in (("deployed", "-"), ("matched", "--")):
                    mu = []
                    for L in LAYERS:
                        mm = ("J_ret" if L == "L2" else "Jtr_ret") if met == "FISHER" else met
                        d = T.get((kg, arm, 0.3, L, mm, M), {})
                        mu.append(np.mean(list(d.values())) if d else np.nan)
                    ax[i].plot(LAYERS, mu, color=COL[M], ls=ls, marker="o", label="%s %s" % (M, arm))
            ax[i].set_title(lab)
            ax[i].grid(alpha=.3)
        ax[0].legend(fontsize=7)
        f2.suptitle("v15.6 %s: where along the network the noise grows or shrinks (test σ = 0.3)" % kg)
        f2.tight_layout()
        f2.savefig(os.path.join(fig, "fig2_layer_profile_%s.png" % kg), dpi=200)
        plt.close(f2)

    # fig3: z-plane decision regions with clean and noisy points (seed 42, test sigma 0.3)
    for cell, K, _ in CELLS:
        for _, arm in ARMS:
            f3, ax = plt.subplots(1, 3, figsize=(18, 6))
            for i, M in enumerate(MODELS):
                fz = npz.get((cell, arm, M, 42))
                if not fz:
                    continue
                z = np.load(fz)
                zc, y, zs, ys = z["z_clean"], z["y_full"], z["z_noisy"][3], z["y_sub"]
                pts = np.concatenate([zc, zs.reshape(-1, 2)])
                lo, hi = np.percentile(pts, 0.5, 0), np.percentile(pts, 99.5, 0)
                pad = 0.1 * (hi - lo)
                gx, gy = np.meshgrid(np.linspace(lo[0] - pad[0], hi[0] + pad[0], 400),
                                     np.linspace(lo[1] - pad[1], hi[1] + pad[1], 400))
                reg = (np.c_[gx.ravel(), gy.ravel()] @ z["W_eff"].T + z["b_eff"]).argmax(1).reshape(gx.shape)
                cm = plt.get_cmap("tab10")
                ax[i].contourf(gx, gy, reg, levels=np.arange(K + 1) - 0.5, colors=[cm(k) for k in range(K)], alpha=.12)
                ax[i].contour(gx, gy, reg, levels=np.arange(K + 1) - 0.5, colors="k", linewidths=.6)
                ax[i].scatter(zs[..., 0].ravel(), zs[..., 1].ravel(), c=[cm(k) for k in np.tile(ys, zs.shape[0])],
                              s=2, alpha=.25, marker="x")
                ax[i].scatter(zc[:, 0], zc[:, 1], c=[cm(k) for k in y], s=5, edgecolors="none")
                ax[i].set_xlim(gx.min(), gx.max())
                ax[i].set_ylim(gy.min(), gy.max())
                ax[i].set_title("%s  (dots: clean test; crosses: σ=0.3 noisy draws)" % M)
                ax[i].set_xlabel("z₀")
                ax[i].set_ylabel("z₁")
            f3.suptitle("v15.6 z-plane, %s, %s, seed 42: class regions (fc3∘fc_mid argmax) and the noisy clouds"
                        % (cell, arm))
            f3.tight_layout()
            f3.savefig(os.path.join(fig, "fig3_zplane_%s_%s.png" % (cell, arm)), dpi=200)
            plt.close(f3)

    # standard set for the primaries (deployed, sigma 0.3)
    for kg in ("K5", "K10"):
        fs, ax = plt.subplots(2, 4, figsize=(20, 9))
        for j, (pid, cn, L, met) in enumerate(PRIMARY):
            vals = dict((M, T.get((kg, "deployed", 0.3, L, met, M), {})) for M in MODELS)
            for i, M in enumerate(MODELS):
                v = np.array([vals[M][s] for s in sorted(vals[M])])
                ax[0, j].scatter(np.full(len(v), i) + np.linspace(-.12, .12, len(v)), v, color=COL[M], s=18)
                if len(v):
                    ax[0, j].bar(i, v.mean(), yerr=stats.t.ppf(.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v)),
                                 color=COL[M], alpha=.25, capsize=4)
            ax[0, j].set_xticks(range(3))
            ax[0, j].set_xticklabels(MODELS)
            ax[0, j].set_title("%s: %s at %s" % (pid, met, L))
            a_, b_ = dict(CONTRASTS_MAP)[cn]
            sd = sorted(set(vals[a_]) & set(vals[b_]))
            dd = [vals[a_][s] - vals[b_][s] for s in sd]
            ax[1, j].bar(range(len(sd)), dd, color=["#555" if x < 0 else "#bbb" for x in dd])
            ax[1, j].axhline(0, color="k", lw=.8)
            ax[1, j].set_xticks(range(len(sd)))
            ax[1, j].set_xticklabels(sd, rotation=90, fontsize=7)
            ax[1, j].set_title("per-seed %s = %s − %s" % (cn, a_, b_))
        fs.suptitle("v15.6 %s primaries (deployed, test σ = 0.3): strip + bar (mean ± 95%% CI), paired differences" % kg)
        fs.tight_layout()
        fs.savefig(os.path.join(fig, "std_primaries_%s.png" % kg), dpi=200)
        plt.close(fs)
    fe, ax = plt.subplots(figsize=(7, 6))
    sec = [c for c in con if c["arm"] == "deployed" and c["sigma"] == 0.3]
    ax.scatter([c["mde"] for c in sec], [abs(c["mean"]) for c in sec], s=8, color="#999", label="secondary")
    ax.scatter([c["mde"] for c in prim], [abs(c["mean"]) for c in prim], s=40, color="#d00", label="primary")
    mx = max([c["mde"] for c in sec] + [abs(c["mean"]) for c in sec] + [1e-9])
    ax.plot([1e-6, mx], [1e-6, mx], "k--", lw=.8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("MDE (80% power)")
    ax.set_ylabel("|effect|")
    ax.legend()
    ax.set_title("v15.6 effect vs MDE (deployed, σ = 0.3); above the line = resolvable")
    fe.tight_layout()
    fe.savefig(os.path.join(fig, "std_effect_vs_mde.png"), dpi=200)
    plt.close(fe)

    # P6: M5 per-seed |sin delta_b| vs eta(L0), deployed sigma 0.3 (exploratory)
    fp, ax = plt.subplots(1, 2, figsize=(12, 5))
    for i, kg in enumerate(("K5", "K10")):
        cells = [c for c, K, _ in CELLS if (K == 5) == (kg == "K5")]
        xs, ys_ = [], []
        for S in SEEDS:
            f = npz.get((cells[0], "deployed", "M5", S))
            e = T.get((kg, "deployed", 0.3, "L0", "eta", "M5"), {}).get(S)
            if f and e is not None:
                xs.append(sin_delta_b(f))
                ys_.append(e)
        if len(xs) > 2:
            rho, p = stats.spearmanr(xs, ys_)
            ax[i].scatter(xs, ys_, color=COL["M5"])
            for S, x_, y_ in zip(SEEDS, xs, ys_):
                ax[i].annotate(str(S), (x_, y_), fontsize=7)
            ax[i].set_title("%s: Spearman ρ = %.2f, p = %.3g" % (kg, rho, p))
            print("P6 %s: Spearman rho = %.3f, p = %.3g (n=%d)" % (kg, rho, p, len(xs)))
        ax[i].set_xlabel("|sin δ_b| of the frozen circuit")
        ax[i].set_ylabel("η at L0 (deployed, σ = 0.3)")
    fp.suptitle("v15.6 P6 (exploratory): steeper frozen circuit -> larger noise shift?")
    fp.tight_layout()
    fp.savefig(os.path.join(fig, "figP6_M5_slope_vs_eta.png"), dpi=200)
    plt.close(fp)
    print("wrote %s (metrics, contrasts, primary CSVs; %d figures)" % (out, len(glob.glob(os.path.join(fig, "*.png")))))


CONTRASTS_MAP = tuple((c, (a, b)) for c, a, b in CONTRASTS)

if __name__ == "__main__":
    main()
