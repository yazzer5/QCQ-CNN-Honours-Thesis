#!/usr/bin/env python3
"""
v13 readout width: collect results and run the PRE-REGISTERED contrasts.
Run from the v13 repo root AFTER both grids pass the gate:

    python3 scripts/collect_v13.py
    python3 scripts/collect_v13.py --no_v10      # only if amendment A1-v13 fired
                                                 # (K=10 width-3 re-run with all 10 seeds)

Width-3 arm (n_obs=3, Z basis) is REUSED, not re-run:
    K=5   v12 results_v12_k5_{A,B}               (paths from pbs/config13.sh)
    K=10  v10 seeds 42-46 + v13 top-up seeds 47-51 (results_v13_k10_all_w3)
Both are only poolable because the reproduction smoke matched exactly.

Writes v13_runs.csv, v13_summary.csv, v13_contrasts.csv.

PRIMARY family (Holm within each K, 3 tests), paired over seeds:
    A  M2(w6)  - M2(w3)    reading ALL cosine features (Z -> Z+X basis)
    B  M2(w15) - M2(w6)    adding the sine features (Y-containing observables)
    C  M3(w15) - M2(w15)   back-end cost left at full readout
       + TOST on C, margin +/-1 pp: "gap closed" iff the 90% CI is inside +/-1.
SECONDARY (reported with MDE, not used for the headline):
    gap(w) = M3(w)-M2(w), w = 3, 6, 15          the gap curve
    dGap   = gap(w3) - gap(w15)                  how much of the gap the readout removes
    M2(w15)-M2(w3)                               total readout effect on the hybrid
    M3(w15)-M3(w3) + TOST +/-1                   classical head is width-insensitive (control)
K=10 v12 COMPLETION (width 3, now 10 seeds; Holm over 4, same family as v12):
    M2-M1, M3-M2, M3-M4 (+TOST +/-1), M4-M2
Every null is reported with its MDE (80% power, alpha .05). A null is a power
statement, never equivalence; only TOST supports "no difference".
"""
import argparse, csv, json, re, subprocess
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats

MARGIN = 1.0


def cfg_paths():
    out = subprocess.run(["bash", "-c", "source pbs/config13.sh && echo $V12_ROOT && echo $V10_RES"],
                         capture_output=True, text=True, check=True).stdout.split()
    return Path(out[0]), Path(out[1])


def parse(f, K, tag, W, src):
    r = json.loads(Path(f).read_text())
    model = r.get("model") or re.match(r"result_(M\d)_", Path(f).name).group(1)
    n_obs = r.get("n_obs")
    if n_obs is not None and int(n_obs) != W:
        raise SystemExit(f"{f}: JSON says n_obs={n_obs} but it sits in a width-{W} cell. Stop.")
    if r.get("loss") not in ("ce", None):
        raise SystemExit(f"{f}: loss={r.get('loss')}, expected ce. Mixed protocols, stop.")
    return dict(K=K, subset=tag, W=W, model=model, seed=int(r["seed"]),
                test_acc=100 * float(r["test_acc"]), best_epoch=r.get("best_epoch"),
                stop_epoch=r.get("stop_epoch"),
                n_params=next((r[x] for x in ("total_params", "n_params") if x in r), ""),
                source=src, file=str(f))


def load(no_v10):
    v12, v10 = cfg_paths()
    rows = []
    for d in sorted(Path(".").glob("results_v13_k*_*_w*")):
        m = re.match(r"results_v13_k(\d+)_(\w+?)_w(\d+)$", d.name)
        if m:
            rows += [parse(f, int(m[1]), m[2], int(m[3]), "v13") for f in sorted(d.glob("result_*.json"))]
    for tag in ("A", "B"):
        rows += [parse(f, 5, tag, 3, "v12") for f in sorted((v12 / f"results_v12_k5_{tag}").glob("result_*.json"))]
    if not no_v10:
        rows += [parse(f, 10, "all", 3, "v10") for f in sorted(v10.glob("result_*.json"))]
    dup = defaultdict(list)
    for r in rows:
        dup[(r["K"], r["subset"], r["W"], r["model"], r["seed"])].append(r["source"])
    clash = {k: v for k, v in dup.items() if len(v) > 1}
    if clash:
        raise SystemExit(f"Same run found twice (did the fallback fire without --no_v10?): {list(clash)[:3]}")
    return rows


def paired(a, b, margin=None):
    seeds = sorted(set(a) & set(b))
    d = np.array([a[s] - b[s] for s in seeds]); n = len(d)
    out = dict(n=n, diff=d.mean() if n else np.nan, sd=np.nan, lo=np.nan, hi=np.nan, p=np.nan,
               mde=np.nan, lo90=np.nan, hi90=np.nan, p_tost=np.nan, equivalent="", per_seed=d.tolist())
    if n < 2:
        return out
    sd = d.std(ddof=1); se = sd / np.sqrt(n); df = n - 1
    t975, t95, t80 = stats.t.ppf(.975, df), stats.t.ppf(.95, df), stats.t.ppf(.80, df)
    out.update(sd=sd, lo=d.mean() - t975 * se, hi=d.mean() + t975 * se,
               p=stats.ttest_1samp(d, 0).pvalue if sd > 0 else (1.0 if d.mean() == 0 else 0.0),
               mde=(t975 + t80) * se)
    if margin is not None and sd > 0:
        lo90, hi90 = d.mean() - t95 * se, d.mean() + t95 * se
        p_lo = 1 - stats.t.cdf((d.mean() + margin) / se, df)
        p_hi = stats.t.cdf((d.mean() - margin) / se, df)
        out.update(lo90=lo90, hi90=hi90, p_tost=max(p_lo, p_hi),
                   equivalent="yes" if (lo90 > -margin and hi90 < margin) else "no")
    return out


def holm(ps):
    ps = np.array([p if np.isfinite(p) else 1.0 for p in ps]); order = np.argsort(ps)
    adj = np.empty(len(ps)); run = 0
    for i, j in enumerate(order):
        run = max(run, min(1, (len(ps) - i) * ps[j])); adj[j] = run
    return adj


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no_v10", action="store_true")
    args = ap.parse_args()
    rows = load(args.no_v10)
    if not rows:
        raise SystemExit("No results found. Run from the v13 repo root.")
    with open("v13_runs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    acc = defaultdict(dict)                                # (K, subset, W, model) -> {seed: acc}
    for r in rows:
        acc[(r["K"], r["subset"], r["W"], r["model"])][r["seed"]] = r["test_acc"]
    for (K, W, m) in sorted({(k, w, m) for (k, t, w, m) in acc if k == 5}):
        A, B = acc.get((5, "A", W, m), {}), acc.get((5, "B", W, m), {})
        acc[(5, "pooled", W, m)] = {s: (A[s] + B[s]) / 2 for s in set(A) & set(B)}

    summ = []
    print(f"\n{'K':>3} {'subset':>7} {'w':>3} {'model':>5} {'n':>3} {'mean%':>7} {'sd':>6}")
    for (K, t, W, m), d in sorted(acc.items()):
        v = np.array(list(d.values()))
        if len(v):
            s = dict(K=K, subset=t, W=W, model=m, n=len(v), mean=v.mean(), sd=v.std(ddof=1) if len(v) > 1 else np.nan)
            summ.append(s); print(f"{K:>3} {t:>7} {W:>3} {m:>5} {len(v):>3} {v.mean():7.2f} {s['sd']:6.2f}")
    with open("v13_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0])); w.writeheader(); w.writerows(summ)

    G = lambda K, t, W, m: acc.get((K, t, W, m), {})
    con = []

    def add(fam, K, t, name, meaning, a, b, margin=None):
        con.append(dict(family=fam, K=K, subset=t, contrast=name, meaning=meaning, **paired(a, b, margin)))

    for K in (5, 10):
        blocks = ["pooled", "A", "B"] if K == 5 else ["all"]
        for t in blocks:
            add("primary", K, t, "M2w6-M2w3", "all cosine features (Z->Z+X)", G(K, t, 6, "M2"), G(K, t, 3, "M2"))
            add("primary", K, t, "M2w15-M2w6", "add sine features (Y bases)", G(K, t, 15, "M2"), G(K, t, 6, "M2"))
            add("primary", K, t, "M3w15-M2w15", "back-end cost at full readout", G(K, t, 15, "M3"), G(K, t, 15, "M2"), MARGIN)
            for W in (3, 6, 15):
                add("secondary", K, t, f"gap_w{W}", f"M3-M2 at width {W}", G(K, t, W, "M3"), G(K, t, W, "M2"))
            g3 = {s: G(K, t, 3, "M3")[s] - G(K, t, 3, "M2")[s] for s in set(G(K, t, 3, "M3")) & set(G(K, t, 3, "M2"))}
            g15 = {s: G(K, t, 15, "M3")[s] - G(K, t, 15, "M2")[s] for s in set(G(K, t, 15, "M3")) & set(G(K, t, 15, "M2"))}
            add("secondary", K, t, "dGap_w3-w15", "gap removed by full readout", g3, g15)
            add("secondary", K, t, "M2w15-M2w3", "total readout effect", G(K, t, 15, "M2"), G(K, t, 3, "M2"))
            add("secondary", K, t, "M3w15-M3w3", "classical head width control", G(K, t, 15, "M3"), G(K, t, 3, "M3"), MARGIN)
    for a, b, mg in (("M2", "M1", None), ("M3", "M2", None), ("M3", "M4", MARGIN), ("M4", "M2", None)):
        add("k10_completion", 10, "all", f"{a}-{b}", "v12 family at K=10, 10 seeds", G(10, "all", 3, a), G(10, "all", 3, b), mg)

    for key in {(c["family"], c["K"], c["subset"]) for c in con}:
        blk = [c for c in con if (c["family"], c["K"], c["subset"]) == key]
        if key[0] in ("primary", "k10_completion"):
            for c, pa in zip(blk, holm([c["p"] for c in blk])):
                c["p_holm"] = pa
        else:
            for c in blk:
                c["p_holm"] = np.nan

    print(f"\n{'family':>15} {'K':>3} {'sub':>6} {'contrast':>13} {'n':>3} {'diff':>7} {'95% CI':>17} "
          f"{'p':>8} {'p_holm':>8} {'MDE':>6} {'TOST':>12}  verdict")
    for c in sorted(con, key=lambda c: (c["family"] != "primary", c["family"], c["K"], c["subset"] != "pooled")):
        if c["n"] < 2:
            print(f"{c['family']:>15} {c['K']:>3} {c['subset']:>6} {c['contrast']:>13} {c['n']:>3}  (missing data)"); continue
        ph = c["p_holm"]; sig = (ph if np.isfinite(ph) else c["p"]) < .05
        verdict = (("DETECTED" if np.isfinite(ph) else "p<.05 (uncorrected, secondary)") if sig else
                   f"null, MDE {c['mde']:.1f} pp" if abs(c["diff"]) < c["mde"] else "not significant")
        tost = f"{c['equivalent']} p={c['p_tost']:.3g}" if c["equivalent"] else ""
        print(f"{c['family']:>15} {c['K']:>3} {c['subset']:>6} {c['contrast']:>13} {c['n']:>3} {c['diff']:7.2f} "
              f"[{c['lo']:6.2f},{c['hi']:6.2f}] {c['p']:8.2g} {ph:8.2g} {c['mde']:6.2f} {tost:>12}  {verdict}")
    keys = [k for k in con[0] if k != "per_seed"] + ["per_seed"]
    with open("v13_contrasts.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for c in con:
            w.writerow({**c, "per_seed": ";".join(f"{x:.3f}" for x in c["per_seed"])})

    seen = defaultdict(set)
    for r in rows:
        if r["n_params"] != "":
            seen[(r["K"], r["W"], r["model"])].add(r["n_params"])
    multi = {k: v for k, v in seen.items() if len(v) > 1}
    if multi:
        print("\nWARNING: parameter count differs across runs of one (K, width, model):", multi)
    caps = [r for r in rows if r["stop_epoch"] not in (None, "") and int(r["stop_epoch"]) >= 200]
    print(f"\nRuns at the 200-epoch cap: {len(caps)}" + ("  <-- comparisons may be budget-bound, log it" if caps else ""))
    print("Wrote v13_runs.csv, v13_summary.csv, v13_contrasts.csv")


if __name__ == "__main__":
    main()
