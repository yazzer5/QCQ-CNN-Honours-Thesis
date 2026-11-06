#!/usr/bin/env python3
"""Paired per-image test of model A vs model B, aggregated across seeds.

WHY THIS EXISTS ALONGSIDE mcnemar.py
------------------------------------
`mcnemar.py` pools discordant pairs across all five seeds. Its own caveat
says so honestly, but the headline figure in its docstring ("2000 images x
5 seeds detects 0.8 pp at z ~ 2.5") is the pooled one, and pooling is not
valid here: the same 2,000 test images are reused by every seed, so the
effective sample size is 2,000, not 10,000. Errors on the same image are
correlated across seeds. Pooling inflates n fivefold and understates the
standard error, in the direction of finding significance.

The fix is available because of a property of the data pipeline worth
stating explicitly: `_balanced_subset()` in train_gadi.py takes the FIRST
n images per class with no RNG, so the test set is byte-identical across
every seed and every model. That is verified below, not assumed.

Given that, each test image i gets a score

    d_i = (fraction of seeds where A is correct on i)
        - (fraction of seeds where B is correct on i)

Averaging within an image absorbs the seed correlation; images remain
independent, so a paired test over the 2,000 d_i values is legitimate. Its
unit of analysis is the image, and n = 2000 genuinely.

Reported alongside:
  - per-seed exact McNemar, so each seed can be inspected on its own
  - the paired-over-seeds t-test on the 5 accuracy scalars (df=4), which is
    what the pre-registration specifies as primary

This script is a SECONDARY analysis. It does not replace the pre-registered
test; it addresses the fact that the pre-registered test has df=4.

    python scripts/mcnemar_seedwise.py results_v10_mnist10 --a M2 --b M1
"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy import stats


def load(d, model):
    runs = {}
    for f in sorted(Path(d).glob(f"result_{model}_*.json")):
        r = json.loads(f.read_text())
        # Keys are test_predictions / test_labels (train_gadi.py line 1296).
        if r.get("test_predictions") is None or r.get("test_labels") is None:
            raise SystemExit(f"{f} has no per-image predictions. "
                             f"Was --save_predictions passed?")
        runs[r["seed"]] = (np.asarray(r["test_predictions"]),
                           np.asarray(r["test_labels"]),
                           r.get("loss"), r.get("test_acc"))
    if not runs:
        raise SystemExit(f"No {model} results in {d}/")
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    args = ap.parse_args()

    A, B = load(args.results_dir, args.a), load(args.results_dir, args.b)
    seeds = sorted(set(A) & set(B))
    if not seeds:
        raise SystemExit("No seeds shared between the two models.")
    print(f"{args.a} vs {args.b}   seeds: {seeds}")

    # ---- integrity: identical test set, identical protocol --------------
    ref = None
    losses = set()
    for m, runs in ((args.a, A), (args.b, B)):
        for s in seeds:
            pred, lab, loss, _ = runs[s]
            losses.add(loss)
            if ref is None:
                ref = lab
            elif not np.array_equal(lab, ref):
                raise SystemExit(
                    f"[INTEGRITY] {m}/seed{s} has a DIFFERENT test set from "
                    f"the first run. The seed-invariance this analysis "
                    f"depends on does not hold. Stop.")
    if len(losses) > 1:
        raise SystemExit(f"[INTEGRITY] mixed losses across runs: {losses}. "
                         f"These results are not one experiment.")
    n = len(ref)
    print(f"[INTEGRITY] test set identical across all runs, n={n}, "
          f"loss={losses.pop()}\n")

    # ---- 1. pre-registered primary: paired t over seed accuracies -------
    acc_a = np.array([A[s][3] for s in seeds])
    acc_b = np.array([B[s][3] for s in seeds])
    diff = acc_a - acc_b
    t, p_t = stats.ttest_rel(acc_a, acc_b)
    sd = diff.std(ddof=1)
    mde = (stats.t.ppf(0.975, len(seeds) - 1) + stats.t.ppf(0.80, len(seeds) - 1)) \
        * sd / np.sqrt(len(seeds))
    print("1. PRE-REGISTERED PRIMARY — paired t over seed accuracies (df=%d)"
          % (len(seeds) - 1))
    for s in seeds:
        print(f"     seed {s}: {args.a} {A[s][3]:.4f}  {args.b} {B[s][3]:.4f} "
              f"  diff {A[s][3]-B[s][3]:+.4f}")
    print(f"     mean diff = {diff.mean()*100:+.2f} pp   SD = {sd*100:.2f} pp")
    print(f"     t = {t:+.3f}   p = {p_t:.4f}")
    print(f"     MDE (80% power, alpha=.05) = {mde*100:.2f} pp")
    if p_t >= 0.05:
        print(f"     NULL. This bounds the effect below ~{mde*100:.2f} pp. "
              f"It is NOT evidence of equivalence.")

    # ---- 2. per-seed exact McNemar --------------------------------------
    print("\n2. PER-SEED EXACT McNEMAR (each valid on its own, n=%d)" % n)
    for s in seeds:
        ca = A[s][0] == ref
        cb = B[s][0] == ref
        b_only, c_only = int((ca & ~cb).sum()), int((~ca & cb).sum())
        p = stats.binomtest(b_only, b_only + c_only, 0.5).pvalue \
            if (b_only + c_only) else 1.0
        print(f"     seed {s}: {args.a}-only {b_only:4d}   {args.b}-only "
              f"{c_only:4d}   p = {p:.4f}")

    # ---- 3. image-level aggregate ---------------------------------------
    ca = np.stack([A[s][0] == ref for s in seeds]).mean(0)
    cb = np.stack([B[s][0] == ref for s in seeds]).mean(0)
    d = ca - cb
    t2, p2 = stats.ttest_1samp(d, 0.0)
    nz = d[d != 0]
    p_sign = stats.binomtest(int((nz > 0).sum()), len(nz), 0.5).pvalue \
        if len(nz) else 1.0
    print(f"\n3. IMAGE-LEVEL AGGREGATE (n={n} images, seed correlation "
          f"absorbed within image)")
    print(f"     mean per-image advantage to {args.a} = {d.mean()*100:+.2f} pp")
    print(f"     paired t over images: t = {t2:+.3f}   p = {p2:.4g}")
    print(f"     sign test on {len(nz)} images that differ: p = {p_sign:.4g}")
    print(f"\n   This is a SECONDARY analysis. It has more power than (1) "
          f"because\n   its unit is the image, but it generalises to THIS "
          f"test set and these\n   five seeds, not to reruns of the "
          f"experiment. Report (1) as primary.")


if __name__ == "__main__":
    main()
