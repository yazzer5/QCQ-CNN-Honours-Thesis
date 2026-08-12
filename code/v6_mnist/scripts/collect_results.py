#!/usr/bin/env python
"""
collect_results.py — aggregate result_*.json into the per-model summary table
the thesis needs (mean +/- std test accuracy across seeds, plus the pairwise
attribution deltas).

    python scripts/collect_results.py results_v5_mnist

Also flags anything that would make the comparison invalid:
  * a run tagged reduced_scale
  * a batch size that doesn't match the rest
  * a missing seed
"""
import sys, json, glob, os, math
from collections import defaultdict


# ── Paired two-sample t-test, dependency-free ───────────────────────────
# Thesis A Sec 5.3 commits to "a paired two-sample t-test for pairwise"
# comparisons across the five seeds. Implemented here rather than via
# scipy because scipy is not pinned in the qcqcnn environment and adding a
# dependency to the analysis path is not worth it. Verified against
# scipy.stats.ttest_rel.
#
# PAIRED is the right test here (not independent two-sample): the same
# five seeds are used for every model, so seed 42's M1 and seed 42's M2
# share initialisation and data-shuffle state. Pairing removes that shared
# variance and is substantially more sensitive than an unpaired test on
# n=5.

def _betacf(a, b, x, itmax=200, eps=3e-16):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < eps:
            break
    return h


def _betai(a, b, x):
    """Regularised incomplete beta function I_x(a,b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lbeta) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta) * _betacf(b, a, 1.0 - x) / b


def paired_ttest(xs, ys):
    """Returns (t, df, two-tailed p). None if fewer than 2 pairs or zero variance."""
    d = [a - b for a, b in zip(xs, ys)]
    n = len(d)
    if n < 2:
        return None
    md = sum(d) / n
    var = sum((v - md) ** 2 for v in d) / (n - 1)
    if var == 0.0:
        return (float("inf") if md != 0 else 0.0), n - 1, (0.0 if md != 0 else 1.0)
    t = md / math.sqrt(var / n)
    df = n - 1
    p = _betai(0.5 * df, 0.5, df / (df + t * t))
    return t, df, p

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    root = sys.argv[1]
    files = sorted(glob.glob(os.path.join(root, "**", "result_*.json"), recursive=True))
    if not files:
        print(f"No result_*.json under {root}")
        sys.exit(1)

    runs = []
    for f in files:
        with open(f) as fh:
            runs.append(json.load(fh))

    warnings = []
    batch_sizes = {r.get("protocol", {}).get("batch_size") for r in runs}
    if len(batch_sizes) > 1:
        warnings.append(f"MIXED BATCH SIZES across runs: {batch_sizes}. The "
                        f"comparison has an uncontrolled second variable.")
    for r in runs:
        if r.get("protocol", {}).get("reduced_scale"):
            warnings.append(f"REDUCED-SCALE run included: {r['model']} seed "
                            f"{r['seed']} -- must not be mixed with full-protocol "
                            f"results.")

    by_model = defaultdict(list)
    for r in runs:
        by_model[r["model"]].append(r)

    def mean(v):
        return sum(v) / len(v)

    def std(v):
        if len(v) < 2:
            return 0.0
        m = mean(v)
        return (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5

    print(f"\n{'Model':6s} {'n seeds':>8s} {'test acc mean':>14s} {'std':>8s} "
          f"{'best val mean':>14s}  seeds")
    print("-" * 78)
    summary = {}
    for m in ["M1", "M2", "M3", "M4"]:
        rs = by_model.get(m, [])
        if not rs:
            print(f"{m:6s} {'--':>8s}  (no results)")
            continue
        accs = [r["test_acc"] for r in rs]
        vals = [r["best_val_acc"] for r in rs]
        seeds = sorted(r["seed"] for r in rs)
        summary[m] = mean(accs)
        print(f"{m:6s} {len(rs):8d} {mean(accs):14.4f} {std(accs):8.4f} "
              f"{mean(vals):14.4f}  {seeds}")
        missing = sorted({42, 43, 44, 45, 46} - set(seeds))
        if missing:
            warnings.append(f"{m}: missing seeds {missing}")

    print("\nAttribution deltas — paired two-sample t-test across seeds")
    print("(Thesis A Sec 5.3. Paired because every model uses the same five seeds.)")
    print(f"\n  {'pair':9s} {'Δ mean acc':>11s} {'t':>8s} {'df':>3s} {'p':>9s}   isolates")
    print("  " + "-" * 76)
    pairs = [
        ("M2", "M1", "front-end trainability (trainable Circuit 4 vs frozen random)"),
        ("M2", "M3", "quantum back-end       (QNN head vs classical head)"),
        ("M3", "M4", "quantum front-end      (Circuit 4 vs bilinear resize)"),
    ]
    by_seed = {m: {r["seed"]: r["test_acc"] for r in rs} for m, rs in by_model.items()}
    for a, b, label in pairs:
        if a not in summary or b not in summary:
            print(f"  {a}-{b:6s} {'--':>11s} {'--':>8s} {'--':>3s} {'--':>9s}   {label} (incomplete)")
            continue
        shared = sorted(set(by_seed[a]) & set(by_seed[b]))
        xs = [by_seed[a][s_] for s_ in shared]
        ys = [by_seed[b][s_] for s_ in shared]
        res = paired_ttest(xs, ys)
        delta = summary[a] - summary[b]
        if res is None:
            print(f"  {a}-{b:6s} {delta:+11.4f} {'--':>8s} {'--':>3s} {'--':>9s}   {label}")
            continue
        t, df, p = res
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"  {a}-{b:6s} {delta:+11.4f} {t:8.3f} {df:3d} {p:9.4f}   {label}  [{star}]")
        if len(shared) < 5:
            warnings.append(f"{a} vs {b}: only {len(shared)} shared seeds — "
                            f"t-test underpowered.")
    print("\n  * p<0.05  ** p<0.01  *** p<0.001  ns = not significant")
    print("  With n=5 seeds (df=4) this test detects only large effects. A 'ns'")
    print("  result means the data does not support a difference -- it is NOT")
    print("  evidence that the architectures are equivalent.")

    if warnings:
        print("\n" + "=" * 70)
        print("WARNINGS")
        print("=" * 70)
        for w in dict.fromkeys(warnings):
            print(f"  ! {w}")
    else:
        print("\nNo protocol warnings. All runs comparable.")


if __name__ == "__main__":
    main()
