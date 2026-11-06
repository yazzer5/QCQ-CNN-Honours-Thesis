#!/usr/bin/env python3
"""
v14 (low data): collect results and run the PRE-REGISTERED contrasts.
Run from the v14 repo root AFTER the grid gate passes (pbs/.grid_ok):

    python3 scripts/collect_v14.py

Every head is at width 15 (all 15 Paulis). From this repo (results_v14_k{K}_{TAG}_n{N}):
    N = 50, 100, 250   M1-M5
    N = 500            M1, M4 (never run at w15 before)
READ from the stored w15 runs (paths from pbs/config14.sh), never re-run:
    N = 500  M2, M3  v13    results_v13_k{K}_{TAG}_w15
    N = 500  M5      v13.5  results_v135_k{K}_{TAG}_w15
Pooling those with v14 is valid only because the v14 smoke reproductions
(M3 vs v13, M5 vs v13.5, N=500 w15 K=5 A seed 42) matched exactly.

Contrasts (paired over seeds; K=5 pooled = per-seed mean of A and B, df=9):
    F  = M3 - M4   trained quantum front-end vs none           (+TOST +/-1)
    T  = M3 - M5   training the quantum front-end              (+TOST +/-1)
    R  = M5 - M4   frozen random quantum front-end vs none     (+TOST +/-1)
    BE = M3 - M2   back-end cost (classical vs quantum head)
    Q  = M2 - M1   training the front-end, quantum head (front-end circuits differ)
    slope(X) = per-seed OLS slope of X on log2(N) over N = 50,100,250,500,
               in pp per doubling of the data; one-sample t over seeds.

PRIMARY (Holm within each K, 4 tests):  F@50, T@50, R@50, slope(F)
SECONDARY (uncorrected, every one with MDE): every contrast at N = 100, 250, 500;
    slope(T), slope(R), slope(BE), slope(Q); AULC contrasts; per-subset results.
Writes v14_runs.csv, v14_summary.csv, v14_contrasts.csv.
"""
import csv
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

MARGIN = 1.0
W = 15
NS = (50, 100, 250, 500)
MODELS = ("M1", "M2", "M3", "M4", "M5")
TAGS = {5: ("A", "B"), 10: ("all",)}
CON = {"F": ("M3", "M4"), "T": ("M3", "M5"), "R": ("M5", "M4"), "BE": ("M3", "M2"), "Q": ("M2", "M1")}
TOST = {"F", "T", "R"}
MEANING = {"F": "trained quantum front-end vs none", "T": "training the quantum front-end",
           "R": "frozen random quantum front-end vs none", "BE": "back-end cost (classical - quantum head)",
           "Q": "training the front-end, quantum head (descriptive)"}


def cfg():
    out = subprocess.run(["bash", "-c", "source pbs/config14.sh && echo $V13_ROOT && echo $V135_ROOT && "
                          "echo $GRID_KS && echo $W_OBS"],
                         capture_output=True, text=True, check=True).stdout.strip().split("\n")
    return Path(out[0]), Path(out[1]), [int(k) for k in out[2].split()], int(out[3])


def parse(f, K, tag, N, src):
    r = json.loads(Path(f).read_text())
    model = r.get("model") or re.match(r"result_(M\d)_", Path(f).name).group(1)
    if r.get("n_obs") is not None and int(r["n_obs"]) != W:
        raise SystemExit(f"{f}: n_obs={r['n_obs']}, v14 is width {W} only. Stop.")
    if r.get("loss") not in ("ce", None):
        raise SystemExit(f"{f}: loss={r.get('loss')}, expected ce. Stop.")
    return dict(K=K, subset=tag, N=N, model=model, seed=int(r["seed"]), test_acc=100 * float(r["test_acc"]),
                best_epoch=r.get("best_epoch"), stop_epoch=r.get("stop_epoch"), source=src, file=str(f))


def load():
    global W
    v13, v135, KS, W = cfg()
    rows = []
    for d in sorted(Path(".").glob("results_v14_k*_*_n*")):
        m = re.match(r"results_v14_k(\d+)_(\w+?)_n(\d+)$", d.name)
        if m:
            rows += [parse(f, int(m[1]), m[2], int(m[3]), "v14") for f in sorted(d.glob("result_M*_*.json"))]
    for K in KS:
        for t in TAGS[K]:
            rows += [parse(f, K, t, 500, "v13") for f in sorted((v13 / f"results_v13_k{K}_{t}_w{W}").glob("result_M[23]_*.json"))]
            rows += [parse(f, K, t, 500, "v13.5") for f in sorted((v135 / f"results_v135_k{K}_{t}_w{W}").glob("result_M5_*.json"))]
    seen = defaultdict(list)
    for r in rows:
        seen[(r["K"], r["subset"], r["N"], r["model"], r["seed"])].append(r["source"])
    clash = {k: v for k, v in seen.items() if len(v) > 1}
    if clash:
        raise SystemExit(f"Same run found twice: {list(clash.items())[:3]}")
    return rows, KS


def paired(d, margin=None):
    d = np.asarray(d, float); n = len(d)
    out = dict(n=n, diff=d.mean() if n else np.nan, sd=np.nan, lo=np.nan, hi=np.nan, p=np.nan, mde=np.nan,
               lo90=np.nan, hi90=np.nan, p_tost=np.nan, equivalent="")
    if n < 2:
        return out
    sd = d.std(ddof=1); se = sd / np.sqrt(n); df = n - 1
    t975, t95, t80 = stats.t.ppf(.975, df), stats.t.ppf(.95, df), stats.t.ppf(.80, df)
    out.update(sd=sd, lo=d.mean() - t975 * se, hi=d.mean() + t975 * se, mde=(t975 + t80) * se,
               p=stats.ttest_1samp(d, 0).pvalue if sd > 0 else (1.0 if d.mean() == 0 else 0.0))
    if margin is not None and sd > 0:
        lo90, hi90 = d.mean() - t95 * se, d.mean() + t95 * se
        out.update(lo90=lo90, hi90=hi90,
                   p_tost=max(1 - stats.t.cdf((d.mean() + margin) / se, df), stats.t.cdf((d.mean() - margin) / se, df)),
                   equivalent="yes" if (lo90 > -margin and hi90 < margin) else "no")
    return out


def holm(ps):
    ps = np.array([p if np.isfinite(p) else 1.0 for p in ps]); order = np.argsort(ps)
    adj = np.empty(len(ps)); run = 0
    for i, j in enumerate(order):
        run = max(run, min(1, (len(ps) - i) * ps[j])); adj[j] = run
    return adj


def aulc(curve):
    """Normalised area under accuracy-vs-log2(N), N = 50..500. Same units as accuracy."""
    x = np.log2(NS); y = np.array([curve[n] for n in NS])
    return float(np.sum((y[1:] + y[:-1]) / 2 * np.diff(x)) / (x[-1] - x[0]))


def main():
    rows, KS = load()
    if not any(r["source"] == "v14" for r in rows):
        raise SystemExit("No v14 results found. Run from the v14 repo root.")
    with open("v14_runs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    acc = defaultdict(dict)                                  # (K, block, N, model) -> {seed: acc}
    for r in rows:
        acc[(r["K"], r["subset"], r["N"], r["model"])][r["seed"]] = r["test_acc"]
    for (K, t, N, m) in list(acc):
        if t == "A" and (K, "B", N, m) in acc:
            A, B = acc[(K, "A", N, m)], acc[(K, "B", N, m)]
            acc[(K, "pooled", N, m)] = {s: (A[s] + B[s]) / 2 for s in set(A) & set(B)}

    print("\nCompleteness (seeds per cell):")
    for K in KS:
        for t in TAGS[K]:
            for N in NS:
                c = "  ".join(f"{m}:{len(acc.get((K, t, N, m), {}))}" for m in MODELS)
                flag = "" if all(len(acc.get((K, t, N, m), {})) == 10 for m in MODELS) else "   <-- INCOMPLETE"
                print(f"  K={K:<2} {t:>3} N={N:<3}  {c}{flag}")

    summ = []
    print(f"\n{'K':>3} {'block':>7} {'N':>4} {'model':>5} {'n':>3} {'mean%':>7} {'sd':>6}")
    for (K, t, N, m), d in sorted(acc.items()):
        v = np.array(list(d.values()))
        s = dict(K=K, subset=t, N=N, model=m, n=len(v), mean=v.mean(), sd=v.std(ddof=1) if len(v) > 1 else np.nan)
        summ.append(s); print(f"{K:>3} {t:>7} {N:>4} {m:>5} {len(v):>3} {v.mean():7.2f} {s['sd']:6.2f}")
    with open("v14_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0])); w.writeheader(); w.writerows(summ)

    def dif(K, t, N, c):
        a, b = acc.get((K, t, N, CON[c][0]), {}), acc.get((K, t, N, CON[c][1]), {})
        return {s: a[s] - b[s] for s in set(a) & set(b)}

    con = []

    def add(fam, K, t, N, name, per_seed, margin=None, meaning=""):
        seeds = sorted(per_seed)
        con.append(dict(family=fam, K=K, subset=t, N=N, contrast=name, meaning=meaning,
                        **paired([per_seed[s] for s in seeds], margin),
                        per_seed=";".join(f"{s}:{per_seed[s]:.3f}" for s in seeds)))

    for K in KS:
        blocks = ["pooled", "A", "B"] if K == 5 else ["all"]
        for t in blocks:
            for c in CON:
                for N in NS:
                    fam = "primary" if (N == 50 and c in TOST and t in ("pooled", "all")) else "secondary"
                    add(fam, K, t, N, c, dif(K, t, N, c), MARGIN if c in TOST else None, MEANING[c])
                # slope over log2 N (seeds with all four N only)
                per = {}
                ds = {N: dif(K, t, N, c) for N in NS}
                for s in set.intersection(*[set(ds[N]) for N in NS]) if all(ds.values()) else ():
                    y = np.array([ds[N][s] for N in NS]); x = np.log2(NS)
                    per[s] = float(np.polyfit(x, y, 1)[0])
                fam = "primary" if (c == "F" and t in ("pooled", "all")) else "secondary"
                add(fam, K, t, "slope", f"slope({c})", per, None, f"change in {c} per doubling of N (pp)")
                # AULC contrast
                if c in TOST:
                    a = {}
                    for s in set.intersection(*[set(acc.get((K, t, N, CON[c][0]), {})) & set(acc.get((K, t, N, CON[c][1]), {}))
                                                for N in NS]) if all(acc.get((K, t, N, m)) for N in NS for m in CON[c]) else ():
                        a[s] = aulc({N: acc[(K, t, N, CON[c][0])][s] for N in NS}) - aulc({N: acc[(K, t, N, CON[c][1])][s] for N in NS})
                    add("secondary", K, t, "AULC", f"AULC({c})", a, MARGIN, "area under the learning curve, difference")

    for K in KS:
        blk = [c for c in con if c["family"] == "primary" and c["K"] == K]
        for c, pa in zip(blk, holm([c["p"] for c in blk])):
            c["p_holm"] = pa
    for c in con:
        c.setdefault("p_holm", np.nan)

    print(f"\n{'family':>9} {'K':>3} {'block':>6} {'N':>5} {'contrast':>10} {'n':>3} {'diff':>7} {'95% CI':>17} "
          f"{'p':>8} {'p_holm':>8} {'MDE':>6} {'TOST':>12}  verdict")
    order = lambda c: (c["family"] != "primary", c["K"], c["subset"] not in ("pooled", "all"), c["subset"],
                       list(CON).index(re.sub(r"^\w+\(|\)$", "", c["contrast"])), str(c["N"]))
    for c in sorted(con, key=order):
        if c["n"] < 2:
            print(f"{c['family']:>9} {c['K']:>3} {c['subset']:>6} {c['N']!s:>5} {c['contrast']:>10} {c['n']:>3}  (missing data)")
            continue
        ph = c["p_holm"]; sig = (ph if np.isfinite(ph) else c["p"]) < .05
        verdict = (("DETECTED" if np.isfinite(ph) else "p<.05 (uncorrected, secondary)") if sig else
                   f"null, MDE {c['mde']:.1f} pp" if abs(c["diff"]) < c["mde"] else "not significant")
        tost = f"{c['equivalent']} p={c['p_tost']:.3g}" if c["equivalent"] else ""
        print(f"{c['family']:>9} {c['K']:>3} {c['subset']:>6} {c['N']!s:>5} {c['contrast']:>10} {c['n']:>3} {c['diff']:7.2f} "
              f"[{c['lo']:6.2f},{c['hi']:6.2f}] {c['p']:8.2g} {ph:8.2g} {c['mde']:6.2f} {tost:>12}  {verdict}")
    with open("v14_contrasts.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(con[0])); w.writeheader(); w.writerows(con)

    print("\nSeed SD (pp) by N, K=5 pooled / K=10:")
    for K in KS:
        t = "pooled" if K == 5 else "all"
        for m in MODELS:
            sds = [np.std(list(acc.get((K, t, N, m), {0: np.nan}).values()), ddof=1) if len(acc.get((K, t, N, m), {})) > 1 else np.nan
                   for N in NS]
            print(f"  K={K:<2} {m}: " + "  ".join(f"N={N}:{s:5.2f}" for N, s in zip(NS, sds)))
    caps = [r for r in rows if r["stop_epoch"] not in (None, "") and int(r["stop_epoch"]) >= 200 * 500 // r["N"]]
    print(f"\nRuns at their max-epoch cap (200*500/N; N=500 includes the stored v13/v13.5 runs): {len(caps)}" + ("  <-- log them; cells with >2/10 are budget-limited" if caps else ""))
    for r in caps:
        print(f"  K={r['K']} {r['subset']} N={r['N']} {r['model']} seed {r['seed']} best {r['best_epoch']} ({r['source']})")
    print("Wrote v14_runs.csv, v14_summary.csv, v14_contrasts.csv")


if __name__ == "__main__":
    main()
