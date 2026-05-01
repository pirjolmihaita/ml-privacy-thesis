"""
variability_tradeoff.py
Privacy-utility tradeoff analysis for classification.

Two directions:
  1. Aggregate trend  — mean F1/AUC across all datasets vs epsilon,
                        one line per method (DP, DP-PHE, FHE), norm=1 fixed.
  2. Variability      — identify representative datasets (most stable vs
                        most variable response to epsilon).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_PATH   = "results/metrics/results_wide.csv"
OUTPUT_DIR     = "results/analysis/variability_tradeoff"
NORM           = 1.0          # data_norm fixed at 1 (best overall)
METRIC         = "AUC"        # "F1" or "AUC"
EPSILONS       = [0.1, 0.5, 1.0, 5.0, 10.0]
CLASS_DATASETS = ["adult", "data", "cervical", "compas", "creditcard", "heart"]
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

df  = pd.read_csv(RESULTS_PATH)
clf = df[df["Task_Type"] == "classification"].copy()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_dp_metric(df_cls, metric, eps, norm, prefix="DP"):
    col = f"{prefix}_{metric}_Eps{eps}_Norm{norm}"
    if col not in df_cls.columns:
        return None
    return df_cls.groupby("Dataset")[col].mean()


def baseline_metric(df_cls, metric):
    col = f"Baseline_{metric}"
    return df_cls.groupby("Dataset")[col].mean()


# ─── 1. AGGREGATE TREND ──────────────────────────────────────────────────────

def plot_aggregate_trend():
    methods = {
        "DP":     ("DP",     "tab:blue",   "-o"),
        "DP-PHE": ("DP_PHE", "tab:orange", "-s"),
        "FHE":    ("FHE",    "tab:green",  "-^"),
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    for label, (prefix, color, marker) in methods.items():
        means, stds = [], []
        for eps in EPSILONS:
            vals = get_dp_metric(clf, METRIC, eps, NORM, prefix=prefix)
            if vals is None:
                means.append(np.nan); stds.append(np.nan)
                continue
            vals = vals[vals.index.isin(CLASS_DATASETS)].dropna()
            means.append(vals.mean())
            stds.append(vals.std())

        means = np.array(means)
        stds  = np.array(stds)
        ax.plot(EPSILONS, means, marker, color=color, label=label, linewidth=2, markersize=7)
        ax.fill_between(EPSILONS, means - stds, means + stds, alpha=0.12, color=color)

    # Baseline horizontal line
    base = baseline_metric(clf, METRIC)
    base = base[base.index.isin(CLASS_DATASETS)].mean()
    ax.axhline(base, color="black", linestyle="--", linewidth=1.5, label=f"Baseline ({METRIC}={base:.3f})")

    ax.set_xscale("log")
    ax.set_xlabel("Epsilon (log scale)", fontsize=12)
    ax.set_ylabel(f"Mean {METRIC} (classification)", fontsize=12)
    ax.set_title(
        f"Privacy-Utility Tradeoff: {METRIC} vs Epsilon\n"
        f"(norm={NORM}, averaged over {len(CLASS_DATASETS)} datasets)",
        fontsize=12
    )
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks(EPSILONS)
    ax.grid(True, alpha=0.3)

    path = os.path.join(OUTPUT_DIR, f"aggregate_trend_{METRIC.lower()}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    # Numeric table
    rows = []
    for label, (prefix, _, _) in methods.items():
        row = {"Method": label}
        for eps in EPSILONS:
            vals = get_dp_metric(clf, METRIC, eps, NORM, prefix=prefix)
            if vals is not None:
                vals = vals[vals.index.isin(CLASS_DATASETS)].dropna()
                row[f"eps={eps}"] = round(vals.mean(), 4)
        rows.append(row)
    tbl = pd.DataFrame(rows).set_index("Method")
    print("\n=== Aggregate trend ===")
    print(tbl.to_string())
    tbl.to_csv(os.path.join(OUTPUT_DIR, f"aggregate_trend_{METRIC.lower()}.csv"))


# ─── 2. VARIABILITY PER DATASET ──────────────────────────────────────────────

def plot_per_dataset():
    """
    Per dataset: F1/AUC vs epsilon for each method (norm=1).
    Identifies stable vs volatile datasets.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=False)
    axes = axes.flatten()

    variability = {}

    for i, ds in enumerate(CLASS_DATASETS):
        ax = axes[i]
        ds_df = clf[clf["Dataset"] == ds]

        # Baseline
        base_val = ds_df[f"Baseline_{METRIC}"].mean()
        ax.axhline(base_val, color="black", linestyle="--", linewidth=1.2, label="Baseline")

        for label, (prefix, color, marker) in {
            "DP":     ("DP",     "tab:blue",   "-o"),
            "DP-PHE": ("DP_PHE", "tab:orange", "-s"),
            "FHE":    ("FHE",    "tab:green",  "-^"),
        }.items():
            vals = []
            for eps in EPSILONS:
                col = f"{prefix}_{METRIC}_Eps{eps}_Norm{NORM}"
                if col in ds_df.columns:
                    v = ds_df[col].mean()
                    vals.append(v if not np.isnan(v) else np.nan)
                else:
                    vals.append(np.nan)
            ax.plot(EPSILONS, vals, marker, color=color, label=label, linewidth=1.8, markersize=6)
            if label == "DP":
                clean = [v for v in vals if not np.isnan(v)]
                variability[ds] = round(np.std(clean), 4) if clean else np.nan

        ax.set_xscale("log")
        ax.set_title(ds.capitalize(), fontsize=11, fontweight="bold")
        ax.set_xlabel("Epsilon", fontsize=9)
        ax.set_ylabel(METRIC, fontsize=9)
        ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax.set_xticks(EPSILONS)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)

    fig.suptitle(
        f"{METRIC} vs Epsilon per Dataset (norm={NORM})",
        fontsize=13, fontweight="bold"
    )
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"per_dataset_{METRIC.lower()}.png")
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    # Variability ranking
    var_df = pd.Series(variability).sort_values()
    print(f"\n=== DP {METRIC} variability per dataset (std across epsilons) ===")
    print("Most STABLE (low std):")
    print(var_df.head(2).to_string())
    print("Most VARIABLE (high std):")
    print(var_df.tail(2).to_string())
    var_df.to_csv(os.path.join(OUTPUT_DIR, f"variability_ranking_{METRIC.lower()}.csv"), header=["std"])


# ─── 3. DEGRADATION VS BASELINE ──────────────────────────────────────────────

def plot_degradation():
    """
    How much % of baseline F1/AUC is lost at each epsilon, per method.
    Degradation = (baseline - method) / baseline * 100
    """
    methods = {
        "DP":     "DP",
        "DP-PHE": "DP_PHE",
        "FHE":    "FHE",
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    for label, prefix in methods.items():
        degradations = []
        for eps in EPSILONS:
            vals_method = get_dp_metric(clf, METRIC, eps, NORM, prefix=prefix)
            vals_base   = baseline_metric(clf, METRIC)
            if vals_method is None:
                degradations.append(np.nan); continue
            common = [d for d in vals_method.index.intersection(vals_base.index) if d in CLASS_DATASETS]
            if not common:
                degradations.append(np.nan); continue
            deg = ((vals_base[common] - vals_method[common]) / vals_base[common] * 100).mean()
            degradations.append(deg)

        color = {"DP": "tab:blue", "DP-PHE": "tab:orange", "FHE": "tab:green"}[label]
        ax.plot(EPSILONS, degradations, "-o", color=color, label=label, linewidth=2, markersize=7)

    ax.axhline(0, color="black", linestyle="--", linewidth=1, label="Baseline (0% loss)")
    ax.set_xscale("log")
    ax.set_xlabel("Epsilon (log scale)", fontsize=12)
    ax.set_ylabel(f"Mean {METRIC} loss vs Baseline (%)", fontsize=11)
    ax.set_title(f"{METRIC} Loss vs Epsilon (norm={NORM})\n(positive = worse than baseline)", fontsize=12)
    ax.legend(fontsize=10)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.set_xticks(EPSILONS)
    ax.grid(True, alpha=0.3)

    path = os.path.join(OUTPUT_DIR, f"degradation_{METRIC.lower()}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Metric: {METRIC} | Norm: {NORM}\n")
    plot_aggregate_trend()
    print()
    plot_per_dataset()
    print()
    plot_degradation()
    print("\nDone.")
