#!/usr/bin/env python
"""
aggregate_results.py — Run AFTER the Phase 1 Gadi jobs finish.

Reads every result_*.json in --results_dir, builds a summary table
(mean/std test accuracy per model x dataset, plus reliability / convergence
stats), and produces comparison plots.

Usage:
  python aggregate_results.py --results_dir ./results_phase1 --out_dir ./analysis
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_results(results_dir):
    rows = []
    for f in sorted(Path(results_dir).glob("result_*.json")):
        with open(f) as fh:
            r = json.load(fh)
        rows.append({
            "model": r["model"],
            "dataset": r["dataset"],
            "seed": r["seed"],
            "test_acc": r["test_acc"],
            "best_val_acc": r["best_val_acc"],
            "n_params": r["n_params"],
            "n_epochs_run": len(r["history"]),
        })
    return pd.DataFrame(rows)


def summarize(df):
    summary = df.groupby(["model", "dataset"]).agg(
        mean_test_acc=("test_acc", "mean"),
        std_test_acc=("test_acc", "std"),
        min_test_acc=("test_acc", "min"),
        max_test_acc=("test_acc", "max"),
        n_seeds=("seed", "count"),
        mean_epochs=("n_epochs_run", "mean"),
    ).reset_index()
    return summary


def plot_bar_comparison(summary, out_dir):
    datasets = summary["dataset"].unique()
    models = ["M1", "M2", "M3", "M4"]
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4.5), sharey=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        sub = summary[summary["dataset"] == ds].set_index("model").reindex(models)
        ax.bar(models, sub["mean_test_acc"], yerr=sub["std_test_acc"], capsize=4,
               color=["#B4B2A9", "#534AB7", "#AFA9EC", "#5F5E5A"])
        ax.set_title(ds)
        ax.set_ylim(0, 1.0)
        ax.set_ylabel("Test accuracy (mean ± std, 5 seeds)")
    fig.suptitle("M1–M4 test accuracy comparison")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "m1_m4_accuracy_comparison.png", dpi=150)
    plt.close(fig)


def plot_reliability(df, out_dir):
    # Reliability = std of test_acc across seeds, per model/dataset
    rel = df.groupby(["model", "dataset"])["test_acc"].std().unstack()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    rel.plot(kind="bar", ax=ax)
    ax.set_ylabel("Std dev of test accuracy across 5 seeds")
    ax.set_title("Cross-seed reliability (lower = more reliable)")
    fig.tight_layout()
    fig.savefig(Path(out_dir) / "reliability_across_seeds.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(args.results_dir)
    if df.empty:
        print(f"No result_*.json files found in {args.results_dir} yet.")
        return

    df.to_csv(out_dir / "all_runs_raw.csv", index=False)
    summary = summarize(df)
    summary.to_csv(out_dir / "summary_by_model_dataset.csv", index=False)
    print(summary.to_string(index=False))

    plot_bar_comparison(summary, out_dir)
    plot_reliability(df, out_dir)
    print(f"\nSaved: {out_dir}/all_runs_raw.csv, summary_by_model_dataset.csv, "
          f"m1_m4_accuracy_comparison.png, reliability_across_seeds.png")


if __name__ == "__main__":
    main()
