#!/usr/bin/env python3
"""McNemar's test on per-image predictions from a v10 grid.

WHY THIS EXISTS
Comparing two models by averaging each into a single accuracy figure and
running a paired t-test on five such figures gives df=4 and an MDE around
2 pp. The binary Fashion-MNIST grids all returned effects below that. But
each run makes thousands of individual decisions, and the informative
question is not "which mean is higher" -- it is "on the images where the two
models DISAGREE, which one is right more often". That is McNemar's test, and
on 2000 images x 5 seeds it detects a 0.8 pp effect at roughly z = 2.5.

    python scripts/mcnemar.py results_v10_mnist10 --a M2 --b M1

Requires runs produced with --save_predictions. Predictions are stored in
test-loader order (shuffle=False), so index i is the same image in every
run of the same seed. The script asserts that alignment rather than
assuming it.
"""
import argparse, json, math
from pathlib import Path
from collections import defaultdict


def load(dirpath, model):
    out = {}
    for f in sorted(Path(dirpath).glob(f"result_{model}_*.json")):
        d = json.loads(f.read_text())
        if d.get("test_predictions") is None:
            raise SystemExit(f"{f.name} has no predictions. Re-run with --save_predictions.")
        out[d["seed"]] = (d["test_predictions"], d["test_labels"], d["test_acc"])
    if not out:
        raise SystemExit(f"No result files for {model} in {dirpath}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--a", default="M2", help="model expected to be better")
    ap.add_argument("--b", default="M1", help="baseline")
    args = ap.parse_args()

    A, B = load(args.results_dir, args.a), load(args.results_dir, args.b)
    seeds = sorted(set(A) & set(B))
    if not seeds:
        raise SystemExit("No shared seeds between the two models.")

    print(f"McNemar: {args.a} vs {args.b}   seeds {seeds}\n")
    print(f"{'seed':>5} {'n':>6} {args.a+' only':>10} {args.b+' only':>10} "
          f"{'both':>7} {'neither':>8} {'Δacc pp':>9}")
    tot_b = tot_c = tot_n = 0
    per_seed = []
    for s in seeds:
        pa, la, acc_a = A[s]
        pb, lb, acc_b = B[s]
        assert la == lb, (
            f"seed {s}: label sequences differ between {args.a} and {args.b}. "
            "The two runs did not see the same test set in the same order -- "
            "McNemar is invalid here. Check the loader was shuffle=False and "
            "that both runs used the same --seed and --n_test.")
        b = sum(1 for p, q, y in zip(pa, pb, la) if p == y and q != y)   # a right, b wrong
        c = sum(1 for p, q, y in zip(pa, pb, la) if p != y and q == y)   # b right, a wrong
        both = sum(1 for p, q, y in zip(pa, pb, la) if p == y and q == y)
        neither = len(la) - b - c - both
        tot_b += b; tot_c += c; tot_n += len(la)
        per_seed.append((s, b, c))
        print(f"{s:>5} {len(la):>6} {b:>10} {c:>10} {both:>7} {neither:>8} "
              f"{(acc_a-acc_b)*100:>+9.2f}")

    disc = tot_b + tot_c
    print(f"\nPooled over {len(seeds)} seeds: {tot_n} decisions, {disc} discordant "
          f"({disc/tot_n*100:.1f}%)")
    if disc == 0:
        print("No discordant pairs -- the models are making identical predictions.")
        return
    # Exact-ish: continuity-corrected chi-square, plus the binomial exact p.
    chi2 = (abs(tot_b - tot_c) - 1) ** 2 / disc
    z = (tot_b - tot_c) / math.sqrt(disc)
    print(f"  {args.a} right / {args.b} wrong : {tot_b}")
    print(f"  {args.b} right / {args.a} wrong : {tot_c}")
    print(f"  net = {tot_b - tot_c:+d}  ({(tot_b-tot_c)/tot_n*100:+.2f} pp)")
    print(f"  z = {z:+.3f}   chi2(1) with continuity correction = {chi2:.3f}")
    try:
        from scipy import stats
        p = stats.chi2.sf(chi2, 1)
        pexact = stats.binomtest(tot_b, disc, 0.5).pvalue
        print(f"  p (chi2) = {p:.5f}      p (exact binomial) = {pexact:.5f}")
    except ImportError:
        print("  install scipy for exact p-values")
    print()
    print("CAVEAT: pooling seeds treats every decision as independent, which "
          "overstates certainty because the five runs share a test set. Report "
          "the per-seed table above alongside the pooled figure, and treat the "
          "pooled p as an upper bound on the evidence, not a headline number.")


if __name__ == "__main__":
    main()
