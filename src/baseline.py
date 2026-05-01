import time
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score, roc_auc_score
)

"""
baseline.py: Standard Baseline Execution Module
This module handles the training and evaluation of non-private machine learning models.
It computes benchmarks (F1-score, Accuracy, MSE) that serve as the ground-truth 
reference for all privacy-preserving methods (Differential Privacy, FHE, PHE).
"""

def run_baseline(mm, m_type, X_train_proc, X_test_proc, y_train, y_test, task_type):

    clf = mm.get_baseline_model(m_type)

    t0 = time.perf_counter()
    clf.fit(X_train_proc, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    preds = clf.predict(X_test_proc)
    inf_time = time.perf_counter() - t0

    if task_type == "classification":
        positive_class_auc = clf.predict_proba(X_test_proc)[:, 1]
        return {
            "Baseline_Accuracy": accuracy_score(y_test, preds),
            "Baseline_F1": f1_score(y_test, preds, average="macro", zero_division=0),
            "Baseline_Precision": precision_score(y_test, preds, average="macro", zero_division=0),
            "Baseline_Recall": recall_score(y_test, preds, average="macro", zero_division=0),
            "Baseline_AUC": roc_auc_score(y_test, positive_class_auc),
            "Baseline_TrainTime": train_time,
            "Baseline_InfTime": inf_time,
        }
    else:
        return {
            "Baseline_MSE": mean_squared_error(y_test, preds),
            "Baseline_MAE": mean_absolute_error(y_test, preds),
            "Baseline_R2": r2_score(y_test, preds),
            "Baseline_TrainTime": train_time,
            "Baseline_InfTime": inf_time,
        }
