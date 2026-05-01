"""
re_identification_attack_regression.py

Empirical demonstration of k-anonymity fragility on a REGRESSION task.
Mirrors the classification scenario from re_identification_attack.py (Adult)
but applied to the Insurance dataset (continuous sensitive attribute: medical charges).

Reference: Basu et al. (2015) - "k-anonymity: Risks and the Reality"

Dataset: Medical Insurance Cost (~1338 records)

Scenario:
  - QI (quasi-identifiers) = numeric attributes protected by k-anonymity
    => age, bmi, children
  - EXTRA attributes (background knowledge) = known to adversary but NOT in QI set
    => sex, smoker, region
  - Sensitive attribute = charges (continuous medical cost)

  k-anonymity guarantees re-id probability <= 1/k using QI only.
  With extra background knowledge, actual re-id probability >> 1/k.

  ML utility is measured by MSE / MAE / R2 on the anonymized dataset.
"""

import sys
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Mondrian — numeric-only, widest normalized span (same as classification script)
# ---------------------------------------------------------------------------

class MondrianKAnonymizer:
    """
    Mondrian k-anonymity on NUMERIC columns.
    Split on widest normalized span dimension.
    Generalization: replace with partition mean.
    """
    def __init__(self, k=10):
        self.k = k
        self.partitions = []

    def fit(self, X: pd.DataFrame):
        self.partitions = []
        self._partition(X, X.index.to_numpy())
        return self

    def _partition(self, df, indices):
        if len(indices) < 2 * self.k:
            self.partitions.append(indices)
            return

        best_col, best_span = None, -1
        for col in df.columns:
            vals = df.loc[indices, col]
            span = vals.max() - vals.min()
            col_range = df[col].max() - df[col].min()
            norm_span = span / col_range if col_range > 0 else 0
            if norm_span > best_span:
                best_span, best_col = norm_span, col

        if best_col is None or best_span == 0:
            self.partitions.append(indices)
            return

        median = df.loc[indices, best_col].median()
        lhs = df.loc[indices, best_col][lambda x: x <= median].index.to_numpy()
        rhs = df.loc[indices, best_col][lambda x: x > median].index.to_numpy()

        if len(lhs) < self.k or len(rhs) < self.k:
            self.partitions.append(indices)
            return

        self._partition(df, lhs)
        self._partition(df, rhs)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy().astype(float)
        for partition in self.partitions:
            for col in X.columns:
                X_out.loc[partition, col] = X.loc[partition, col].mean()
        return X_out

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "regression", "insurance.csv")

QI_COLS    = ["age", "bmi", "children"]
EXTRA_COLS = ["sex", "smoker", "region"]
SENSITIVE  = "charges"

ALL_FEATURE_COLS = QI_COLS + EXTRA_COLS

K_VALUES          = [2, 5, 10, 50, 100, 200]
SUCCESS_THRESHOLD = 0.2
N_SAMPLE          = 1000   # insurance has ~1338 records total

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_insurance():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()
    for c in df.select_dtypes("object").columns:
        df[c] = df[c].str.strip()
    df = df.dropna().reset_index(drop=True)
    return df


def encode_features(X: pd.DataFrame) -> np.ndarray:
    X_enc = X.copy()
    for col in X_enc.select_dtypes("object").columns:
        X_enc[col] = LabelEncoder().fit_transform(X_enc[col].astype(str))
    return X_enc.values.astype(float)


def train_and_evaluate(X_train, X_test, y_train, y_test):
    reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    reg.fit(X_train, y_train)
    preds = reg.predict(X_test)
    return {
        "mse":  round(mean_squared_error(y_test, preds), 2),
        "mae":  round(mean_absolute_error(y_test, preds), 2),
        "r2":   round(r2_score(y_test, preds), 4),
    }

# ---------------------------------------------------------------------------
# Attack functions (identical logic to classification script)
# ---------------------------------------------------------------------------

def build_pid_map(anonymizer):
    pid_to_indices = defaultdict(list)
    idx_to_pid     = {}
    for pid, indices in enumerate(anonymizer.partitions):
        for idx in indices:
            pid_to_indices[pid].append(idx)
            idx_to_pid[idx] = pid
    return idx_to_pid, pid_to_indices


def attack_basic(idx_to_pid, pid_to_indices, sample_indices):
    """Adversary knows only generalized QI. Re-id prob = 1 / partition_size."""
    probs = []
    for idx in sample_indices:
        pid  = idx_to_pid[idx]
        size = len(pid_to_indices[pid])
        probs.append(1.0 / size)
    return np.array(probs)


def attack_enhanced(df, idx_to_pid, pid_to_indices, sample_indices, extra_cols):
    """Adversary knows QI + extra attributes. Filters partition by exact extra values."""
    probs = []
    for idx in sample_indices:
        pid           = idx_to_pid[idx]
        partition_idx = pid_to_indices[pid]
        target_extra  = df.loc[idx, extra_cols]
        candidates    = df.loc[partition_idx, extra_cols]
        n_match       = (candidates == target_extra).all(axis=1).sum()
        probs.append(1.0 / max(n_match, 1))
    return np.array(probs)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run():
    print("=" * 65)
    print("RE-IDENTIFICATION ATTACK — Insurance Dataset (Regression)")
    print("Basu et al. (2015): k-anonymity: Risks and the Reality")
    print("=" * 65)

    df = load_insurance()
    print(f"\nDataset: {len(df)} records")
    print(f"QI (protected): {QI_COLS}")
    print(f"Extra attributes (adversary background knowledge): {EXTRA_COLS}")
    print(f"Sensitive attribute: {SENSITIVE} (continuous)")

    # Baseline ML on original QI cols (no anonymization)
    y = df[SENSITIVE].values
    X_qi_enc = encode_features(df[QI_COLS])
    X_tr, X_te, y_tr, y_te = train_test_split(X_qi_enc, y, test_size=0.3, random_state=42)
    baseline = train_and_evaluate(X_tr, X_te, y_tr, y_te)
    print(f"\nBaseline (original QI cols, no anonymization): "
          f"MSE={baseline['mse']}  MAE={baseline['mae']}  R2={baseline['r2']}")

    X_qi = df[QI_COLS].copy()

    rng            = np.random.RandomState(42)
    sample_indices = rng.choice(df.index.tolist(), size=min(N_SAMPLE, len(df)), replace=False)

    rows = []

    for k in K_VALUES:
        anon = MondrianKAnonymizer(k=k)
        anon.fit(X_qi)

        part_sizes   = [len(p) for p in anon.partitions]
        n_partitions = len(anon.partitions)
        min_size     = min(part_sizes)
        mean_size    = np.mean(part_sizes)

        idx_to_pid, pid_to_indices = build_pid_map(anon)
        valid_sample = [i for i in sample_indices if i in idx_to_pid]

        probs_basic    = attack_basic(idx_to_pid, pid_to_indices, valid_sample)
        probs_enhanced = attack_enhanced(df, idx_to_pid, pid_to_indices, valid_sample, EXTRA_COLS)

        theoretical = 1.0 / k

        success_basic    = (probs_basic    >= SUCCESS_THRESHOLD).mean() * 100
        success_enhanced = (probs_enhanced >= SUCCESS_THRESHOLD).mean() * 100

        ratio_basic    = probs_basic.mean()    / theoretical
        ratio_enhanced = probs_enhanced.mean() / theoretical

        # ML utility on anonymized QI cols
        X_qi_anon  = anon.transform(X_qi)
        X_anon_enc = encode_features(X_qi_anon)
        X_tr_a, X_te_a, y_tr_a, y_te_a = train_test_split(X_anon_enc, y, test_size=0.3, random_state=42)
        ml_anon = train_and_evaluate(X_tr_a, X_te_a, y_tr_a, y_te_a)

        mse_increase = round((ml_anon["mse"] - baseline["mse"]) / baseline["mse"] * 100, 1)
        r2_drop      = round((baseline["r2"] - ml_anon["r2"]) / abs(baseline["r2"]) * 100, 1) if baseline["r2"] != 0 else 0.0

        print(f"\n--- k = {k}  (theoretical guarantee: re-id prob <= 1/{k} = {theoretical:.4f}) ---")
        print(f"  Partitions: {n_partitions}  |  min_size={min_size}  |  mean_size={mean_size:.1f}")
        print(f"  BASIC adversary    : mean_prob={probs_basic.mean():.4f}  "
              f"ratio={ratio_basic:.2f}x  "
              f"attacks_succeeded(>={SUCCESS_THRESHOLD})={success_basic:.1f}%")
        print(f"  ENHANCED adversary : mean_prob={probs_enhanced.mean():.4f}  "
              f"ratio={ratio_enhanced:.2f}x  "
              f"attacks_succeeded(>={SUCCESS_THRESHOLD})={success_enhanced:.1f}%")
        print(f"  ML utility         : MSE={ml_anon['mse']} (+{mse_increase}%)  "
              f"MAE={ml_anon['mae']}  R2={ml_anon['r2']} (-{r2_drop}%)")

        rows.append({
            "k": k,
            "Theoretical_prob (1/k)": theoretical,
            "Mean_prob_basic": round(probs_basic.mean(), 5),
            "Mean_prob_enhanced": round(probs_enhanced.mean(), 5),
            "Ratio_basic (actual/theoretical)": round(ratio_basic, 2),
            "Ratio_enhanced (actual/theoretical)": round(ratio_enhanced, 2),
            f"Succeeded_basic_% (>={SUCCESS_THRESHOLD})": round(success_basic, 1),
            f"Succeeded_enhanced_% (>={SUCCESS_THRESHOLD})": round(success_enhanced, 1),
            "Num_partitions": n_partitions,
            "Mean_partition_size": round(mean_size, 1),
            "ML_MSE": ml_anon["mse"],
            "ML_MAE": ml_anon["mae"],
            "ML_R2": ml_anon["r2"],
            "MSE_increase_%": mse_increase,
            "R2_drop_%": r2_drop,
        })

    results_df = pd.DataFrame(rows)
    out_csv = os.path.join(PROJECT_ROOT, "results_re_identification_regression.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"\nResults saved -> {out_csv}")
    return results_df


if __name__ == "__main__":
    run()
