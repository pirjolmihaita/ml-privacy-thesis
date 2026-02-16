from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score
)

"""
concrete.py: Fully Homomorphic Encryption (FHE) Wrapper
Integrates the Concrete ML library to perform encrypted inference. It supports 
both standard FHE and a hybrid DP-FHE mode where noise is added to weights 
before the model is compiled into a cryptographic circuit.
"""

def run_concrete_fhe_only(mm, m_type, X_train_he, X_test_he, y_train, y_test, task_type, suffix, he_subset_n=10):
    """
    Run Concrete ML in FHE-only mode (apply_dp_weights=False).
    Returns:
      - Classification: Accuracy, F1, Precision, Recall + Train/Compile/Inference Times.
      - Regression: MSE, MAE, R2 + Train/Compile/Inference Times.
    """
    y_true_sub = y_test[:he_subset_n]

    preds_conc, conc_train_time, conc_compile_time, conc_inf_time = mm.run_concrete_inference(
        model_type=m_type,
        X_train=X_train_he,
        X_test=X_test_he,
        y_train=y_train,
        n_samples=he_subset_n,
        fhe_mode="simulate",
        apply_dp_weights=False
    )

    if task_type == "classification":
        return {
            f"Concrete_Accuracy{suffix}": accuracy_score(y_true_sub, preds_conc),
            f"Concrete_F1{suffix}": f1_score(y_true_sub, preds_conc, average="macro", zero_division=0),
            f"Concrete_Precision{suffix}": precision_score(y_true_sub, preds_conc, average="macro", zero_division=0),
            f"Concrete_Recall{suffix}": recall_score(y_true_sub, preds_conc, average="macro", zero_division=0),
            f"Concrete_CompileTime{suffix}": conc_compile_time,
            f"Concrete_InfTime{suffix}": conc_inf_time,
            f"Concrete_TrainTime{suffix}": conc_train_time,
        }
    else:
        return {
            f"Concrete_MSE{suffix}": mean_squared_error(y_true_sub, preds_conc),
            f"Concrete_MAE{suffix}": mean_absolute_error(y_true_sub, preds_conc),
            f"Concrete_R2{suffix}": r2_score(y_true_sub, preds_conc),
            f"Concrete_CompileTime{suffix}": conc_compile_time,
            f"Concrete_InfTime{suffix}": conc_inf_time,
            f"Concrete_TrainTime{suffix}": conc_train_time,
        }

def run_concrete_dp_weights(mm, m_type, eps, norm, X_train_he, X_test_he, y_train, y_test, task_type, suffix, he_subset_n=10):
    """
    ConcreteW: Hybrid mode applying DP-like noise to weights followed by FHE execution.
    Note: Currently supported only for Linear/Logistic Regression models.
    Returns:
      - Metrics prefixed with 'ConcreteW_' + execution timings.
    """
    if m_type not in ["lr", "lin_reg"]:
        return {}

    y_true_sub = y_test[:he_subset_n]

    preds_conc_w, w_train_time, w_compile_time, w_inf_time = mm.run_concrete_inference(
        model_type=m_type,
        X_train=X_train_he,
        X_test=X_test_he,
        y_train=y_train,
        n_samples=he_subset_n,
        fhe_mode="simulate",
        apply_dp_weights=True,
        dp_epsilon=eps,
        data_norm=norm,
        max_abs_weight=5.0,
        dp_mechanism="laplace",
        random_state=42
    )

    if task_type == "classification":
        return {
            f"ConcreteW_Accuracy{suffix}": accuracy_score(y_true_sub, preds_conc_w),
            f"ConcreteW_F1{suffix}": f1_score(y_true_sub, preds_conc_w, average="macro", zero_division=0),
            f"ConcreteW_Precision{suffix}": precision_score(y_true_sub, preds_conc_w, average="macro", zero_division=0),
            f"ConcreteW_Recall{suffix}": recall_score(y_true_sub, preds_conc_w, average="macro", zero_division=0),
            f"ConcreteW_CompileTime{suffix}": w_compile_time,
            f"ConcreteW_InfTime{suffix}": w_inf_time,
            f"ConcreteW_TrainTime{suffix}": w_train_time,
        }
    else:
        return {
            f"ConcreteW_MSE{suffix}": mean_squared_error(y_true_sub, preds_conc_w),
            f"ConcreteW_MAE{suffix}": mean_absolute_error(y_true_sub, preds_conc_w),
            f"ConcreteW_R2{suffix}": r2_score(y_true_sub, preds_conc_w),
            f"ConcreteW_CompileTime{suffix}": w_compile_time,
            f"ConcreteW_InfTime{suffix}": w_inf_time,
            f"ConcreteW_TrainTime{suffix}": w_train_time,
        }
