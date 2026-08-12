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
import sys, json, glob, os
from collections import defaultdict

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

print("\nAttribution deltas (mean test accuracy):")
pairs = [
    ("M2", "M1", "front-end trainability  (trainable Circuit 4 vs frozen random)"),
    ("M2", "M3", "quantum back-end        (QNN head vs classical head)"),
    ("M3", "M4", "quantum front-end       (Circuit 4 vs bilinear resize)"),
]
for a, b, label in pairs:
    if a in summary and b in summary:
        print(f"  {a} - {b}  {summary[a] - summary[b]:+.4f}   {label}")
    else:
        print(f"  {a} - {b}     --      {label}  (incomplete)")

if warnings:
    print("\n" + "=" * 70)
    print("WARNINGS")
    print("=" * 70)
    for w in dict.fromkeys(warnings):
        print(f"  ! {w}")
else:
    print("\nNo protocol warnings. All runs comparable.")
