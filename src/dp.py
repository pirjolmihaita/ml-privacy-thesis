import time
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score,roc_auc_score
)
"""
dp.py: Differential Privacy (DP) Execution Module
Handles the training of Differentially Private models using objective perturbation 
and noise injection. It evaluates the impact of different Epsilon values on 
model utility across various dataset sizes.
"""
def run_dp(mm, m_type, eps, norm, n_features, classes, bounds,
           X_train_proc, X_test_proc, y_train, y_test, task_type, suffix, bounds_y=None):
    """
    Performs Differential Privacy (DP) training and prediction on FULL features (X_train_proc / X_test_proc).
    Returns a dictionary containing the following columns:
      - For classification:
          DP_Accuracy{suffix}, DP_F1{suffix}, DP_Precision{suffix}, DP_Recall{suffix},
          DP_TrainTime{suffix}, DP_InfTime{suffix}
      - For regression:
          DP_MSE{suffix}, DP_MAE{suffix}, DP_R2{suffix}, 
          DP_TrainTime{suffix}, DP_InfTime{suffix}
    """
    clf_dp = mm.get_dp_model(m_type, eps, norm, n_features=n_features, classes=classes, bounds=bounds, bounds_y=bounds_y)

    t0 = time.perf_counter()
    clf_dp.fit(X_train_proc, y_train)
    train_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    preds_dp = clf_dp.predict(X_test_proc)
    inf_time = time.perf_counter() - t0

    if task_type == "classification":
        positive_class_auc = clf_dp.predict_proba(X_test_proc)[:, 1]
        return clf_dp, {
            f"DP_Accuracy{suffix}": accuracy_score(y_test, preds_dp),
            f"DP_F1{suffix}": f1_score(y_test, preds_dp, average="macro", zero_division=0),
            f"DP_Precision{suffix}": precision_score(y_test, preds_dp, average="macro", zero_division=0),
            f"DP_Recall{suffix}": recall_score(y_test, preds_dp, average="macro", zero_division=0),
            f"DP_AUC{suffix}": roc_auc_score(y_test, positive_class_auc),
            f"DP_TrainTime{suffix}": train_time,
            f"DP_InfTime{suffix}": inf_time,
        }
    else:
        return clf_dp, {
            f"DP_MSE{suffix}": mean_squared_error(y_test, preds_dp),
            f"DP_MAE{suffix}": mean_absolute_error(y_test, preds_dp),
            f"DP_R2{suffix}": r2_score(y_test, preds_dp),
            f"DP_TrainTime{suffix}": train_time,
            f"DP_InfTime{suffix}": inf_time,
        }
