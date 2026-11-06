#!/usr/bin/env python3
"""
v13.5 (M5 = M3 with frozen quantum front-end): collect results and run the
PRE-REGISTERED contrasts. Run from the v13.5 repo root AFTER every grid passes:

    python3 scripts/collect_v135.py

M5 comes from this repo (results_v135_k{K}_{TAG}_w{W}). M1-M4 are READ from the
existing runs, never re-run (paths from pbs/config135.sh):
    K=3,4,5  w3      v12  results_v12_k{K}_{A,B}
    K=10     w3      v10 seeds 42-46 + v13 top-up seeds 47-51 (results_v13_k10_all_w3)
    K=5,10   w6,w15  v13  results_v13_k{K}_{TAG}_w{W}  (M2, M3)
Pooling M5 with these is valid only because the v13.5 reproduction smoke
(M3, K=5 A w3, seed 42) matched v12 exactly.

Writes v135_runs.csv (M5 + every comparator row used), v135_summary.csv,
v135_contrasts.csv.

PRIMARY (Holm within each K and block, over every primary test at that K):
    F   M3 - M5  (+TOST +/-1)  at every width     does TRAINING the quantum front-end
                                                   help, with a classical head?
    R   M5 - M4  (+TOST +/-1)  at w3 only         does an UNTRAINED Circuit-4 front-end
                                                   beat no front-end?
SECONDARY (reported with MDE, uncorrected):
    I   (M3-M5) - (M2-M1)   at w3     is the training effect different with a quantum head?
                                      (M1's frozen front-end is Henderson, not Circuit 4:
                                      read as descriptive)
    H   M5 - M1             at w3     both front-ends frozen; head AND front-end type differ:
                                      descriptive only
    C   M5(w15) - M5(w3) (+TOST)      K=5, 10: width control for M5 (expected ~0, rank-2 fc2)
K=3,4,5 primary is pooled over subsets A and B (per-seed mean, df=9), with each
subset reported alongside. Every null is reported with its MDE (80% power,
alpha .05). A null is a power statement; only TOST supports "no difference".
"""
import csv, json, re, subprocess
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy import stats

MARGIN = 1.0
KS = (3, 4, 5, 10)
WIDTHS = {3: (3,), 4: (3,), 5: (3, 6, 15), 10: (3, 6, 15)}
TAGS = {3: ("A", "B"), 4: ("A", "B"), 5: ("A", "B"), 10: ("all",)}


def cfg():
    out = subprocess.run(["bash", "-c", "source pbs/config135.sh && echo $V12_ROOT && echo $V13_ROOT && echo $V10_RES"],
                         capture_output=True, text=True, check=True).stdout.split()
    return Path(out[0]), Path(out[1]), Path(out[2])


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


def load():
    v12, v13, v10 = cfg()
    rows = []
    for d in sorted(Path(".").glob("results_v135_k*_*_w*")):
        m = re.match(r"results_v135_k(\d+)_(\w+?)_w(\d+)$", d.name)
        if m:
            rows += [parse(f, int(m[1]), m[2], int(m[3]), "v13.5") for f in sorted(d.glob("result_M5_*.json"))]
    for K in (3, 4, 5):
        for tag in ("A", "B"):
            rows += [parse(f, K, tag, 3, "v12") for f in sorted((v12 / f"results_v12_k{K}_{tag}").glob("result_*.json"))]
    rows += [parse(f, 10, "all", 3, "v10") for f in sorted(v10.glob("result_*.json"))]
    for K in (5, 10):
        for tag in TAGS[K]:
            for W in (3, 6, 15):
                d = v13 / f"results_v13_k{K}_{tag}_w{W}"
                rows += [parse(f, K, tag, W, "v13") for f in sorted(d.glob("result_*.json"))]
    dup = defaultdict(list)
    for r in rows:
        dup[(r["K"], r["subset"], r["W"], r["model"], r["seed"])].append(r["source"])
    clash = {k: v for k, v in dup.items() if len(v) > 1}
    if clash:
        raise SystemExit(f"Same run found twice: {list(clash.items())[:3]}")
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


def diff(a, b):
    return {s: a[s] - b[s] for s in set(a) & set(b)}


def main():
    rows = load()
    if not any(r["model"] == "M5" for r in rows):
        raise SystemExit("No M5 results found. Run from the v13.5 repo root.")
    with open("v135_runs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    acc = defaultdict(dict)                                # (K, subset, W, model) -> {seed: acc}
    for r in rows:
        acc[(r["K"], r["subset"], r["W"], r["model"])][r["seed"]] = r["test_acc"]
    for (K, t, W, m) in list(acc):
        if t == "A" and (K, "B", W, m) in acc:
            A, B = acc[(K, "A", W, m)], acc[(K, "B", W, m)]
            acc[(K, "pooled", W, m)] = {s: (A[s] + B[s]) / 2 for s in set(A) & set(B)}

    # completeness: every M5 cell must have 10 seeds, and every comparator its seeds
    print("\nM5 completeness:")
    for K in KS:
        for t in TAGS[K]:
            for W in WIDTHS[K]:
                n = len(acc.get((K, t, W, "M5"), {}))
                print(f"  K={K:<2} {t:>3} w{W:<2}: {n}/10" + ("" if n == 10 else "   <-- INCOMPLETE"))

    summ = []
    print(f"\n{'K':>3} {'subset':>7} {'w':>3} {'model':>5} {'n':>3} {'mean%':>7} {'sd':>6}")
    for (K, t, W, m), d in sorted(acc.items()):
        v = np.array(list(d.values()))
        if len(v):
            s = dict(K=K, subset=t, W=W, model=m, n=len(v), mean=v.mean(), sd=v.std(ddof=1) if len(v) > 1 else np.nan)
            summ.append(s); print(f"{K:>3} {t:>7} {W:>3} {m:>5} {len(v):>3} {v.mean():7.2f} {s['sd']:6.2f}")
    with open("v135_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0])); w.writeheader(); w.writerows(summ)

    G = lambda K, t, W, m: acc.get((K, t, W, m), {})
    con = []

    def add(fam, K, t, W, name, meaning, a, b, margin=None):
        con.append(dict(family=fam, K=K, subset=t, W=W, contrast=name, meaning=meaning, **paired(a, b, margin)))

    for K in KS:
        blocks = ["pooled", "A", "B"] if K != 10 else ["all"]
        for t in blocks:
            for W in WIDTHS[K]:
                add("primary", K, t, W, "M3-M5", "training the quantum front-end (classical head)",
                    G(K, t, W, "M3"), G(K, t, W, "M5"), MARGIN)
            add("primary", K, t, 3, "M5-M4", "untrained Circuit-4 front-end vs none",
                G(K, t, 3, "M5"), G(K, t, 3, "M4"), MARGIN)
            add("secondary", K, t, 3, "(M3-M5)-(M2-M1)", "training effect: classical vs quantum head",
                diff(G(K, t, 3, "M3"), G(K, t, 3, "M5")), diff(G(K, t, 3, "M2"), G(K, t, 3, "M1")))
            add("secondary", K, t, 3, "M5-M1", "both frozen; head + front-end type differ (descriptive)",
                G(K, t, 3, "M5"), G(K, t, 3, "M1"))
            if K in (5, 10):
                add("secondary", K, t, 15, "M5w15-M5w3", "M5 width control",
                    G(K, t, 15, "M5"), G(K, t, 3, "M5"), MARGIN)

    for key in {(c["family"], c["K"], c["subset"]) for c in con}:
        blk = [c for c in con if (c["family"], c["K"], c["subset"]) == key]
        if key[0] == "primary":
            for c, pa in zip(blk, holm([c["p"] for c in blk])):
                c["p_holm"] = pa
        else:
            for c in blk:
                c["p_holm"] = np.nan

    print(f"\n{'family':>9} {'K':>3} {'sub':>6} {'w':>3} {'contrast':>16} {'n':>3} {'diff':>7} {'95% CI':>17} "
          f"{'p':>8} {'p_holm':>8} {'MDE':>6} {'TOST':>12}  verdict")
    for c in sorted(con, key=lambda c: (c["family"] != "primary", c["K"], c["subset"] != "pooled", c["subset"], c["W"])):
        if c["n"] < 2:
            print(f"{c['family']:>9} {c['K']:>3} {c['subset']:>6} {c['W']:>3} {c['contrast']:>16} {c['n']:>3}  (missing data)"); continue
        ph = c["p_holm"]; sig = (ph if np.isfinite(ph) else c["p"]) < .05
        verdict = (("DETECTED" if np.isfinite(ph) else "p<.05 (uncorrected, secondary)") if sig else
                   f"null, MDE {c['mde']:.1f} pp" if abs(c["diff"]) < c["mde"] else "not significant")
        tost = f"{c['equivalent']} p={c['p_tost']:.3g}" if c["equivalent"] else ""
        print(f"{c['family']:>9} {c['K']:>3} {c['subset']:>6} {c['W']:>3} {c['contrast']:>16} {c['n']:>3} {c['diff']:7.2f} "
              f"[{c['lo']:6.2f},{c['hi']:6.2f}] {c['p']:8.2g} {ph:8.2g} {c['mde']:6.2f} {tost:>12}  {verdict}")
    keys = [k for k in con[0] if k != "per_seed"] + ["per_seed"]
    with open("v135_contrasts.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader()
        for c in con:
            w.writerow({**c, "per_seed": ";".join(f"{x:.3f}" for x in c["per_seed"])})

    seen = defaultdict(set)
    for r in rows:
        if r["n_params"] != "" and r["model"] in ("M3", "M5"):
            seen[(r["K"], r["W"], r["model"])].add(r["n_params"])
    print("\nParameter counts M3 vs M5 (expected identical architecture):")
    for (K, W, m) in sorted(seen):
        print(f"  K={K} w={W} {m}: {sorted(seen[(K, W, m)])}")
    caps = [r for r in rows if r["model"] == "M5" and r["stop_epoch"] not in (None, "") and int(r["stop_epoch"]) >= 200]
    print(f"\nM5 runs at the 200-epoch cap: {len(caps)}" + ("  <-- log it" if caps else ""))
    print("Wrote v135_runs.csv, v135_summary.csv, v135_contrasts.csv")


if __name__ == "__main__":
    main()
