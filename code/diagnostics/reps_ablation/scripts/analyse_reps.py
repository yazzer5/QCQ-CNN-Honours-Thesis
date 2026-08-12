#!/usr/bin/env python
"""Compare reps=1 vs reps=2 across seeds. Paired per (model, seed)."""
import json, glob, sys
import numpy as np
try:
    from scipy import stats
except ImportError:
    stats = None

d = sys.argv[1] if len(sys.argv) > 1 else "results_reps_fashion_mnist_c06"
R = {}
for f in glob.glob(f"{d}/result_*.json"):
    j = json.load(open(f))
    R[(j["model"], j["seed"], j["protocol"]["be_reps"])] = j

models = sorted({k[0] for k in R})
seeds = sorted({k[1] for k in R})
print(f"{d}   ({len(R)} runs)\n")
print(f"{'model':6} {'reps':>5} {'n':>3} {'test acc':>16} {'params':>7}")
print("-" * 48)
acc = {}
for m in models:
    for r in (1, 2):
        v = np.array([R[(m, s, r)]["test_acc"] for s in seeds if (m, s, r) in R])
        if not v.size:
            continue
        acc[(m, r)] = v
        p = R[[k for k in R if k[0] == m and k[2] == r][0]]["n_params"]
        print(f"{m:6} {r:5} {v.size:3} {v.mean()*100:8.2f} ± {v.std(ddof=1)*100:4.2f}% {p:7}")

print("\nreps=2 minus reps=1, paired across seeds")
print("-" * 48)
for m in models:
    if (m, 1) in acc and (m, 2) in acc and acc[(m, 1)].size == acc[(m, 2)].size:
        diff = acc[(m, 2)] - acc[(m, 1)]
        line = f"  {m}: {diff.mean()*100:+.2f} pp"
        if stats is not None and diff.size > 1:
            t, p = stats.ttest_rel(acc[(m, 2)], acc[(m, 1)])
            sig = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"
            line += f"   t={t:+.3f}  p={p:.4f}  [{sig}]"
        print(line)

sd = np.mean([v.std(ddof=1) for v in acc.values()])
print(f"\nmean within-arm SD = {sd*100:.2f} pp  ->  MDE ~ {2*sd*100:.2f} pp")
print("A null here means the ansatz depth does not matter at this scale --")
print("it is NOT evidence that reps=1 and reps=2 are equivalent in general.")
print("\nNote: fixed-seed runs vary ~0.8 pp from GPU non-determinism, so part")
print("of the paired difference is machine noise rather than the ansatz.")
