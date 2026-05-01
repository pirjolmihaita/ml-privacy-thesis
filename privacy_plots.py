"""
privacy_plots.py
================
Privacy-utility tradeoff visualizations.

Run:
    python privacy_plots.py

Output saved to: results/analysis/figures/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
CHECKPOINTS = BASE_DIR / "checkpoints"
OUT_DIR     = BASE_DIR / "results" / "analysis" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLS_DATASETS = ["adult", "cervical", "compas", "creditcard", "data", "heart"]

DS_LABELS = {
    "adult":      "Adult Census",
    "cervical":   "Cervical Cancer",
    "compas":     "COMPAS",
    "creditcard": "Credit Card Fraud",
    "data":       "Breast Cancer",
    "heart":      "Heart Disease",
}

EPSILONS  = ["0.1", "0.5", "1.0", "5.0", "10.0"]
EPS_FLOAT = [0.1,   0.5,   1.0,   5.0,   10.0]

# Privacy methods: display label -> column prefix in the CSV
METHODS = {
    "DP":      "DP",
    "DP-PHE":  "DP_PHE",
    "FHE":     "FHE",
    "DP-FHE":  "FHE_DP",
}

# Visual style per method
METHOD_STYLE = {
    "DP":     {"color": "#1f77b4", "marker": "o", "lw": 2.2, "ms": 8},
    "DP-PHE": {"color": "#ff7f0e", "marker": "s", "lw": 2.0, "ms": 7},
    "FHE":    {"color": "#2ca02c", "marker": "^", "lw": 2.0, "ms": 8},
    "DP-FHE": {"color": "#9467bd", "marker": "D", "lw": 2.0, "ms": 7},
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_dataset(name: str, model: str = "lr", norm: float = 1.0) -> dict:
    """
    Load one checkpoint CSV and extract F1 values at a fixed data_norm.

    norm=1.0 is the standard default in the DP literature.
    Each point on the curve = F1 of that method at that epsilon, with norm fixed.

    Returns:
      'baseline_f1' : float  — F1 with no privacy applied
      'curves'      : dict   — {method: [f1_at_eps0.1, ..., f1_at_eps10.0]}
    """
    path = CHECKPOINTS / f"{name}_results.csv"
    df   = pd.read_csv(path)

    # Baseline row: no noise, no epsilon
    bl = df[(df["Type"] == "Baseline") & (df["Model"] == model)]
    baseline_f1 = float(bl["Baseline_F1"].values[0]) if len(bl) > 0 else None

    # Column name norm suffix: 1.0 stays "1.0", 10.0 becomes "10", 100.0 becomes "100"
    ns = "1.0" if norm == 1.0 else str(int(norm))

    dp_rows = df[(df["Type"] == "DP") & (df["Model"] == model)].copy()
    dp_rows["Epsilon"]   = dp_rows["Epsilon"].astype(str)
    dp_rows["Data_Norm"] = dp_rows["Data_Norm"].astype(float)
    dp_norm = dp_rows[dp_rows["Data_Norm"] == norm]

    curves = {m: [] for m in METHODS}
    for eps in EPSILONS:
        row = dp_norm[dp_norm["Epsilon"] == eps]
        for label, prefix in METHODS.items():
            col = f"{prefix}_F1_Eps{eps}_Norm{ns}"
            if len(row) > 0 and col in row.columns and pd.notna(row[col].values[0]):
                curves[label].append(float(row[col].values[0]))
            else:
                curves[label].append(np.nan)

    return {"baseline_f1": baseline_f1, "curves": curves}


# ── RQ1 — F1 vs Epsilon, fixed norm=1.0, single dataset ──────────────────────

def plot_rq1_single(dataset: str = "adult", model: str = "lr", norm: float = 1.0):
    """
    RQ1: How does F1 change as epsilon (privacy budget) decreases?

    Each LINE = one privacy method:
      - DP      : Differential Privacy noise added during training, normal inference
      - DP-PHE  : same DP training, but inference on partially encrypted data (Paillier)
      - FHE     : fully encrypted inference (Concrete ML quantized model)
      - DP-FHE  : DP on weights + fully encrypted inference

    X axis : epsilon (log scale). Lower = stronger privacy = more noise.
    Y axis : F1 Score (raw). Higher = better model performance.
    Dashed : Baseline — same model with NO privacy protection.

    data_norm is fixed at {norm} (standard DP clipping bound).
    """
    data = load_dataset(dataset, model=model, norm=norm)
    bl   = data["baseline_f1"]

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("white")

    # ── Baseline horizontal line ──────────────────────────────────────────────
    ax.axhline(
        bl, color="#333333", linestyle="--", linewidth=1.8,
        zorder=2, label=f"Baseline  (F1 = {bl:.3f})"
    )

    # ── One line per method ───────────────────────────────────────────────────
    for method, style in METHOD_STYLE.items():
        ys = data["curves"][method]
        xs = [EPS_FLOAT[i] for i, v in enumerate(ys) if not np.isnan(v)]
        ys = [v             for v    in ys             if not np.isnan(v)]

        if not xs:
            continue

        ax.plot(
            xs, ys,
            color=style["color"],
            marker=style["marker"],
            linewidth=style["lw"],
            markersize=style["ms"],
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=3,
        )

        # Print the F1 value next to each point
        for x, y in zip(xs, ys):
            ax.annotate(
                f"{y:.2f}",
                xy=(x, y),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center", fontsize=8,
                color=style["color"],
            )

        # Label the method at the last point on the right
        ax.annotate(
            method,
            xy=(xs[-1], ys[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center", fontsize=10,
            color=style["color"],
            fontweight="bold",
        )

    # ── Axes ─────────────────────────────────────────────────────────────────
    ax.set_xscale("log")
    ax.set_xticks(EPS_FLOAT)
    ax.set_xticklabels(EPSILONS, fontsize=11)
    ax.set_xlabel(
        "epsilon  (privacy budget, log scale)\n"
        "<-- more private                    less private -->",
        fontsize=11, labelpad=8
    )
    ax.set_ylabel("F1 Score", fontsize=12)

    # Zoom Y to where the data actually lives — no empty space at bottom
    all_vals = [v for m in METHODS for v in data["curves"][m] if not np.isnan(v)]
    y_min = max(0, min(all_vals) - 0.08) if all_vals else 0
    ax.set_ylim(y_min, bl + 0.12)

    ax.set_title(
        f"RQ1 — Privacy-Utility Tradeoff:  F1 vs epsilon\n"
        f"{DS_LABELS[dataset]}  |  Model: {model.upper()}  |  data_norm = {norm}",
        fontsize=13, fontweight="bold",
    )

    ax.yaxis.grid(True, alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Legend ────────────────────────────────────────────────────────────────
    handles = [
        mlines.Line2D([], [], color="#333333", linestyle="--", linewidth=1.8,
                      label=f"Baseline — no privacy  (F1 = {bl:.3f})"),
        mlines.Line2D([], [], color=METHOD_STYLE["DP"]["color"],
                      marker="o", markersize=7, markeredgecolor="white",
                      linewidth=2, label="DP — Differential Privacy"),
        mlines.Line2D([], [], color=METHOD_STYLE["DP-PHE"]["color"],
                      marker="s", markersize=7, markeredgecolor="white",
                      linewidth=2, label="DP-PHE — DP + partial encryption (Paillier)"),
        mlines.Line2D([], [], color=METHOD_STYLE["FHE"]["color"],
                      marker="^", markersize=7, markeredgecolor="white",
                      linewidth=2, label="FHE — fully encrypted inference (Concrete ML)"),
        mlines.Line2D([], [], color=METHOD_STYLE["DP-FHE"]["color"],
                      marker="D", markersize=7, markeredgecolor="white",
                      linewidth=2, label="DP-FHE — DP weights + full encryption"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9,
              framealpha=0.92, title="Privacy Method", title_fontsize=10)

    fig.tight_layout()
    out_path = OUT_DIR / f"rq1_{dataset}_{model}_norm{int(norm)}.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("RQ1 - data (Breast Cancer), norm=1.0...")
    plot_rq1_single(dataset="data", model="lr", norm=1.0)
    print("Done.")
