import time
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score, roc_auc_score
)

"""
dp_phe.py: Partial Homomorphic Encryption (PHE) Module
Implements encrypted inference using the Paillier cryptosystem. It focuses on 
linear models (Logistic/Linear Regression) where weights are kept in clear 
while the input data and dot-product calculations remain encrypted.
"""

def run_dp_phe(mm, m_type, eps, norm, X_train_he, X_test_he, y_train, y_test, task_type, suffix, he_subset_n=None, bounds_y=None):

    
    if m_type not in ["lr", "lin_reg"]:
        return {}

    clf_he_training = mm.get_dp_model(m_type, eps, norm, n_features=X_train_he.shape[1], bounds_y=bounds_y)

    t0 = time.perf_counter()
    clf_he_training.fit(X_train_he, y_train)
    phe_train_time = time.perf_counter() - t0

    preds_he, he_time = mm.run_he_inference(clf_he_training, X_test_he, n_samples=he_subset_n)

    y_true_sub = y_test[:he_subset_n]

    if task_type == "classification":
        return {
            f"DP_PHE_Accuracy{suffix}": accuracy_score(y_true_sub, preds_he),
            f"DP_PHE_F1{suffix}": f1_score(y_true_sub, preds_he, average="macro", zero_division=0),
            f"DP_PHE_Precision{suffix}": precision_score(y_true_sub, preds_he, average="macro", zero_division=0),
            f"DP_PHE_Recall{suffix}": recall_score(y_true_sub, preds_he, average="macro", zero_division=0),
            f"DP_PHE_AUC{suffix}": roc_auc_score(y_true_sub, preds_he),
            f"DP_PHE_TrainTime{suffix}": phe_train_time,
            f"DP_PHE_InfTime{suffix}": he_time,
        }
    else:
        return {
            f"DP_PHE_MSE{suffix}": mean_squared_error(y_true_sub, preds_he),
            f"DP_PHE_MAE{suffix}": mean_absolute_error(y_true_sub, preds_he),
            f"DP_PHE_R2{suffix}": r2_score(y_true_sub, preds_he),
            f"DP_PHE_TrainTime{suffix}": phe_train_time,
            f"DP_PHE_InfTime{suffix}": he_time,
        }
