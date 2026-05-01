"""
membership_inference_experiment.py
====================================
Experiment: Membership Inference Attack (ART) on Adult dataset.
Strategies tested: Baseline, DP, and ConcreteW (all epsilons and data norms).

Run from project root:
    source venv_wsl/bin/activate
    python -m src.membership_inference_experiment
"""

import numpy as np
import pandas as pd

from .data_loader import DataLoader
from .models import ModelManager
from .common import get_unique_classes, get_bounds

from .mi_attack import run_mi_attack

# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------
DATASETS     = ["adult", "data", "cervical", "compas", "creditcard", "heart"]  # classification only
MODELS       = ["lr", "nb", "dt", "rf"]
CONCRETEW_MODELS = ["lr"]  # ConcreteW only supports linear models
EPSILONS     = [0.1, 0.5, 1.0, 5.0, 10.0]
DATA_NORMS   = [1.0, 10, 100]
OUTPUT_FILE  = "results_membership_inference.csv"


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def run_mia_for_dataset(ds_name, dl, mm):
    """Runs MIA for all models and strategies on a single dataset."""

    print(f"\n{'='*60}")
    print(f"  Membership Inference Attack — {ds_name.upper()}")
    print(f"{'='*60}\n")

    try:
        X_train, X_test, y_train, y_test, preprocessor, task_type = dl.load_and_preprocess(ds_name)
    except Exception as e:
        print(f"  [SKIP] Failed to load {ds_name}: {e}")
        return []

    if task_type != 'classification':
        print(f"  [SKIP] {ds_name} is regression — MIA not applicable.")
        return []

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc  = preprocessor.transform(X_test)
    n_features   = X_train_proc.shape[1]

    y_train_np = np.array(y_train)
    y_test_np  = np.array(y_test)

    results = []
    classes = get_unique_classes(y_train_np, task_type)

    for m_key in MODELS:
        print(f"\n{'='*50}")
        print(f"  Model: {m_key.upper()}")
        print(f"{'='*50}")

        # ---- BASELINE ----
        print(f"  [Baseline] training...")
        try:
            clf_base = mm.get_baseline_model(m_key)
            clf_base.fit(X_train_proc, y_train_np)
            acc = run_mi_attack(clf_base, X_train_proc, y_train_np, X_test_proc, y_test_np)
            print(f"  [Baseline] MI Attack Accuracy = {acc:.2%}")
            results.append({
                "Dataset": ds_name, "Model": m_key, "Strategy": "Baseline",
                "Epsilon": None, "Data_Norm": None, "MI_AttackAccuracy": acc
            })
        except Exception as e:
            print(f"  [Baseline] ERROR: {e}")

        # ---- DP (all epsilons x all norms) ----
        for norm in DATA_NORMS:
            bounds = get_bounds(norm)
            for eps in EPSILONS:
                print(f"  [DP eps={eps} norm={norm}] training...")
                try:
                    clf_dp = mm.get_dp_model(
                        m_key, epsilon=eps, data_norm=norm,
                        n_features=n_features, classes=classes, bounds=bounds
                    )
                    clf_dp.fit(X_train_proc, y_train_np)
                    acc = run_mi_attack(clf_dp, X_train_proc, y_train_np, X_test_proc, y_test_np)
                    print(f"  [DP eps={eps} norm={norm}] MI Attack Accuracy = {acc:.2%}")
                    results.append({
                        "Dataset": ds_name, "Model": m_key, "Strategy": "DP",
                        "Epsilon": eps, "Data_Norm": norm, "MI_AttackAccuracy": acc
                    })
                except Exception as e:
                    print(f"  [DP eps={eps} norm={norm}] ERROR: {e}")

    # ---- CONCRETEW (lr only) ----
    print(f"\n{'='*50}")
    print(f"  ConcreteW (DP weights + FHE) — LR only")
    print(f"{'='*50}")
    for m_key in CONCRETEW_MODELS:
        for norm in DATA_NORMS:
            for eps in EPSILONS:
                print(f"  [ConcreteW eps={eps} norm={norm}] training...")
                try:
                    model_conc, _, _, _, _ = mm.run_concrete_inference(
                        model_type=m_key,
                        X_train=X_train_proc,
                        X_test=X_test_proc,
                        y_train=y_train_np,
                        n_samples=10,
                        fhe_mode="simulate",
                        apply_dp_weights=True,
                        dp_epsilon=eps,
                        data_norm=norm,
                        max_abs_weight=5.0,
                        dp_mechanism="laplace",
                        random_state=42
                    )
                    mi_acc = run_mi_attack(model_conc, X_train_proc, y_train_np, X_test_proc, y_test_np)
                    if mi_acc is not None:
                        print(f"  [ConcreteW eps={eps} norm={norm}] MI Attack Accuracy = {mi_acc:.2%}")
                        results.append({
                            "Dataset": ds_name, "Model": m_key, "Strategy": "ConcreteW",
                            "Epsilon": eps, "Data_Norm": norm, "MI_AttackAccuracy": mi_acc
                        })
                except Exception as e:
                    print(f"  [ConcreteW eps={eps} norm={norm}] ERROR: {e}")

    return results


def main():
    dl = DataLoader()
    mm = ModelManager()

    # Resume from existing results if available
    import os
    if os.path.exists(OUTPUT_FILE):
        existing = pd.read_csv(OUTPUT_FILE)
        all_results = existing.to_dict("records")
        done_datasets = set(existing["Dataset"].unique())
        print(f"  Resuming — already done: {done_datasets}")
    else:
        all_results = []
        done_datasets = set()

    for ds_name in DATASETS:
        if ds_name in done_datasets:
            print(f"\n  [SKIP] {ds_name} already in results.")
            continue
        ds_results = run_mia_for_dataset(ds_name, dl, mm)
        all_results.extend(ds_results)

        # Save after each dataset to avoid data loss
        pd.DataFrame(all_results).to_csv(OUTPUT_FILE, index=False)
        print(f"\n  Checkpoint saved after {ds_name} → {OUTPUT_FILE}")

    print(f"\n{'='*60}")
    print(f"All datasets done. Results saved to: {OUTPUT_FILE}")
    print(f"{'='*60}\n")
    print(pd.DataFrame(all_results).to_string(index=False))


if __name__ == "__main__":
    main()
