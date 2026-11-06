#!/usr/bin/env python3
"""
v12 class-count sweep -- collect results and run the pre-registered contrasts.

    python scripts/collect_v12.py                       # K = 3, 4, 5
    python scripts/collect_v12.py --v10 results_v10_mnist10   # + K=10 point

Reads results_v12_k{K}_{A,B}/result_*.json. Writes:
    v12_runs.csv        one row per run
    v12_summary.csv     mean / sd / n per (K, subset, model) and pooled per K
    v12_contrasts.csv   paired contrasts per (K, subset) and pooled per K

Contrasts (paired over seeds, B subtracted from A):
    M2-M1  does TRAINING the quantum front-end help?          (Q2)
    M3-M2  what does REMOVING the quantum back-end give?      (Q2, back-end cost)
    M3-M4  quantum front-end vs classical, with classical head (Q3)
    M4-M2  gap between classical CNN and our model            (headline)

Pooled per K: each seed's accuracy is averaged over the two subsets first,
then paired over seeds. MDE = (t_.975 + t_.80) * SD(diff)/sqrt(n): the
smallest true difference this design detects with 80% power at alpha=.05.
A non-significant contrast is a POWER statement -- always report the MDE.
"""
import argparse, csv, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats

CONTRASTS = [("M2", "M1"), ("M3", "M2"), ("M3", "M4"), ("M4", "M2")]
MEANING = {"M2-M1": "train front-end", "M3-M2": "drop quantum back-end",
           "M3-M4": "quantum vs classical front-end", "M4-M2": "classical CNN vs ours"}


def load(root, v10):
    rows = []
    for d in sorted(Path(root).glob("results_v12_k*_*")):
        m = re.match(r"results_v12_k(\d+)_(\w+)$", d.name)
        if not m:
            continue
        for f in sorted(d.glob("result_*.json")):
            rows.append(parse(f, int(m.group(1)), m.group(2)))
    if v10:
        for f in sorted(Path(v10).glob("result_*.json")):
            rows.append(parse(f, 10, "all"))
    return rows


def parse(f, k, tag):
    r = json.loads(f.read_text())
    model = r.get("model") or re.match(r"result_(M\d\w?)_", f.name).group(1)
    nparams = next((r[x] for x in ("total_params", "n_params", "params_total") if x in r), "")
    return dict(K=k, subset=tag, model=model, seed=int(r["seed"]),
                test_acc=100 * float(r["test_acc"]), loss=r.get("loss"),
                best_epoch=r.get("best_epoch"), stop_epoch=r.get("stop_epoch"),
                gap_at_best=r.get("gap_at_best"), n_params=nparams, file=str(f))


def paired(a, b):
    seeds = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in seeds])
    n = len(d)
    if n < 2:
        return dict(n=n, diff=float(d.mean()) if n else np.nan, sd=np.nan, lo=np.nan,
                    hi=np.nan, p=np.nan, mde=np.nan, dz=np.nan, per_seed=d.tolist())
    sd = d.std(ddof=1); se = sd / np.sqrt(n); df = n - 1
    t = stats.t.ppf(0.975, df)
    p = stats.ttest_1samp(d, 0).pvalue if sd > 0 else (1.0 if d.mean() == 0 else 0.0)
    mde = (t + stats.t.ppf(0.80, df)) * se
    return dict(n=n, diff=d.mean(), sd=sd, lo=d.mean() - t * se, hi=d.mean() + t * se,
                p=p, mde=mde, dz=d.mean() / sd if sd > 0 else np.nan, per_seed=d.tolist())


def tost(per_seed, margin):
    """Paired TOST: equivalent within +/- margin if BOTH one-sided tests reject,
    i.e. the 90% CI of the mean difference lies entirely inside (-margin, +margin).
    Pre-registered (amendment A1) for M3-M4 only, margin fixed before K=4/5 were read."""
    d = np.asarray(per_seed, float); n = len(d)
    if n < 2 or d.std(ddof=1) == 0:
        return np.nan, np.nan, np.nan, False
    se = d.std(ddof=1) / np.sqrt(n); df = n - 1
    p_lo = 1 - stats.t.cdf((d.mean() + margin) / se, df)   # H0: diff <= -margin
    p_hi = stats.t.cdf((d.mean() - margin) / se, df)       # H0: diff >= +margin
    t90 = stats.t.ppf(0.95, df)
    lo90, hi90 = d.mean() - t90 * se, d.mean() + t90 * se
    p = max(p_lo, p_hi)
    return p, lo90, hi90, bool(p < 0.05)


def holm(ps):
    order = np.argsort(ps); m = len(ps); adj = np.empty(m); running = 0
    for i, j in enumerate(order):
        running = max(running, min(1, (m - i) * ps[j])); adj[j] = running
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--v10", default=None, help="v10 10-class results dir (adds K=10)")
    ap.add_argument("--tost_margin", type=float, default=1.0,
                    help="Equivalence margin (pp) for M3-M4, amendment A1. Do not change after results are read.")
    args = ap.parse_args()
    rows = load(args.root, args.v10)
    if not rows:
        raise SystemExit("No results found. Run from the repo root.")

    bad = [r for r in rows if r["loss"] not in ("ce", None)]
    if bad:
        raise SystemExit(f"{len(bad)} result(s) not stamped loss=ce, e.g. {bad[0]['file']}. Mixed protocols -- stop.")

    with open("v12_runs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    acc = defaultdict(dict)                        # (K, subset, model) -> {seed: acc}
    for r in rows:
        acc[(r["K"], r["subset"], r["model"])][r["seed"]] = r["test_acc"]
    # pooled per K: average over subsets per seed (only seeds present in every subset)
    Ks = sorted({r["K"] for r in rows})
    for k in Ks:
        tags = sorted({t for (kk, t, _) in acc if kk == k and t != "pooled"})
        if len(tags) < 2:      # e.g. K=10 from v10 is already one "all" block
            continue
        for mdl in sorted({m for (kk, _, m) in acc if kk == k}):
            per = [acc.get((k, t, mdl), {}) for t in tags]
            common = set.intersection(*[set(p) for p in per]) if per else set()
            acc[(k, "pooled", mdl)] = {s: float(np.mean([p[s] for p in per])) for s in common}

    summ = []
    print(f"\n{'K':>3} {'subset':>7} {'model':>5} {'n':>3} {'mean%':>7} {'sd':>6}")
    for (k, t, mdl), d in sorted(acc.items()):
        v = np.array(list(d.values()))
        if not len(v):
            continue
        s = dict(K=k, subset=t, model=mdl, n=len(v), mean=v.mean(),
                 sd=v.std(ddof=1) if len(v) > 1 else np.nan, chance=100 / k)
        summ.append(s)
        print(f"{k:>3} {t:>7} {mdl:>5} {len(v):>3} {v.mean():7.2f} {s['sd']:6.2f}")
    with open("v12_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0].keys())); w.writeheader(); w.writerows(summ)

    con = []
    for (k, t) in sorted({(k, t) for (k, t, _) in acc}):
        block = []
        for a, b in CONTRASTS:
            if (k, t, a) in acc and (k, t, b) in acc:
                res = paired(acc[(k, t, a)], acc[(k, t, b)])
                block.append(dict(K=k, subset=t, contrast=f"{a}-{b}",
                                  meaning=MEANING[f"{a}-{b}"], **res))
        if block:
            adj = holm(np.array([c["p"] if np.isfinite(c["p"]) else 1 for c in block]))
            for c, pa in zip(block, adj):
                c["p_holm"] = pa
            con += block

    print(f"\n{'K':>3} {'subset':>7} {'contrast':>7} {'n':>3} {'diff pp':>8} {'95% CI':>17} "
          f"{'p':>8} {'p_holm':>8} {'MDE pp':>7}  verdict")
    for c in con:
        if c["n"] < 2:
            continue
        if c["p_holm"] < 0.05:
            verdict = "DETECTED"
        elif abs(c["diff"]) < c["mde"]:
            verdict = f"null, underpowered below {c['mde']:.1f} pp"
        else:
            verdict = "not significant after Holm"
        print(f"{c['K']:>3} {c['subset']:>7} {c['contrast']:>7} {c['n']:>3} {c['diff']:8.2f} "
              f"[{c['lo']:6.2f},{c['hi']:6.2f}] {c['p']:8.2g} {c['p_holm']:8.2g} {c['mde']:7.2f}  {verdict}")
    print(f"\nEQUIVALENCE (amendment A1): paired TOST on M3-M4, margin +/-{args.tost_margin} pp, alpha 0.05")
    print(f"{'K':>3} {'subset':>7} {'diff pp':>8} {'90% CI':>17} {'p_TOST':>8}  verdict")
    for c in con:
        if c["contrast"] == "M3-M4":
            p, lo9, hi9, eq = tost(c["per_seed"], args.tost_margin)
            c.update(tost_margin=args.tost_margin, p_tost=p, ci90_lo=lo9, ci90_hi=hi9, equivalent=eq)
            if np.isfinite(p):
                v = (f"EQUIVALENT within +/-{args.tost_margin} pp" if eq
                     else "equivalence NOT shown (90% CI crosses the margin)")
                print(f"{c['K']:>3} {c['subset']:>7} {c['diff']:8.2f} [{lo9:6.2f},{hi9:6.2f}] {p:8.2g}  {v}")
        else:
            c.update(tost_margin="", p_tost="", ci90_lo="", ci90_hi="", equivalent="")
    keys = [k for k in con[0] if k != "per_seed"] + ["per_seed"]
    with open("v12_contrasts.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for c in con:
            w.writerow({**c, "per_seed": ";".join(f"{x:.3f}" for x in c["per_seed"])})

    np_seen = defaultdict(set)
    for r in rows:
        if r["n_params"] != "":
            np_seen[(r["K"], r["model"])].add(r["n_params"])
    multi = {k: v for k, v in np_seen.items() if len(v) > 1}
    if multi:
        print("\nWARNING: parameter count differs across runs of the same (K, model):", multi)
    print("\nWrote v12_runs.csv, v12_summary.csv, v12_contrasts.csv")


if __name__ == "__main__":
    main()
