import time
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score
)
"""
k_anonymity.py: K-Anonymity (Mondrian) Module
Applies the Mondrian algorithm for data anonymization. It transforms raw 
datasets into k-anonymous representations and evaluates the remaining machine 
learning utility of the generalized data.
"""
from .privacy.k_anonymity import MondrianAnonymizer

def run_k_anonymity_block(mm, preprocessor, X_train, X_test, y_train, y_test, task_type, models, ks, wide_results, ds_name):
    """
      - concat train/test
      - mondrian anonymize
      - split back
      - clone preprocessor
      - train baseline models pe anon data
      - scrie KAnon_* cu K{k}
    """
    if not ((task_type == "classification" and ds_name == "adult") or (task_type == "regression" and ds_name == "communities")):
        return

    for k in ks:
        # 1) Anonymize full
        X_full = pd.concat([X_train, X_test])
        cat_cols = X_full.select_dtypes(include=["object", "category"]).columns.tolist()

        anonymizer = MondrianAnonymizer(k=k)
        t_anon = time.time()
        anonymizer.fit(X_full, categorical_features=cat_cols)
        X_full_anon = anonymizer.transform(X_full)
        anon_duration = time.time() - t_anon

        # 2) Split back
        X_train_anon = X_full_anon.iloc[:len(X_train)]
        X_test_anon = X_full_anon.iloc[len(X_train):]

        # 3) Re-run preprocessing (clone)
        preprocessor_k = clone(preprocessor)
        X_train_anon_proc = preprocessor_k.fit_transform(X_train_anon)
        X_test_anon_proc = preprocessor_k.transform(X_test_anon)

        for m_key in models:
            if task_type == "regression":
                if m_key == "lr": m_type = "lin_reg"
                elif m_key == "nb": 
                    continue
                elif m_key == "dt": m_type = "dt_reg"
                elif m_key == "rf": m_type = "rf_reg"
                else:
                    continue
            else:
                m_type = m_key

            row_key = (ds_name, m_type, "K-Anon", f"K={k}", "None")
            if row_key not in wide_results:
                wide_results[row_key] = {
                    "Dataset": ds_name, "Model": m_type, "Type": "K-Anon",
                    "Epsilon": f"K={k}", "Data_Norm": "None", "Task_Type": task_type
                }

            clf = mm.get_baseline_model(m_type)

            t0 = time.time()
            clf.fit(X_train_anon_proc, y_train)
            train_time = time.time() - t0

            t0 = time.time()
            preds = clf.predict(X_test_anon_proc)
            inf_time = time.time() - t0

            if task_type == "classification":
                wide_results[row_key].update({
                    f"KAnon_Accuracy_K{k}": accuracy_score(y_test, preds),
                    f"KAnon_F1_K{k}": f1_score(y_test, preds, average="macro", zero_division=0),
                    f"KAnon_Precision_K{k}": precision_score(y_test, preds, average="macro", zero_division=0),
                    f"KAnon_Recall_K{k}": recall_score(y_test, preds, average="macro", zero_division=0),
                    f"KAnon_InfTime_K{k}": inf_time,
                    f"KAnon_TrainTime_K{k}": train_time,
                    f"KAnon_ProcessTime_K{k}": anon_duration,
                })
            else:
                wide_results[row_key].update({
                    f"KAnon_MSE_K{k}": mean_squared_error(y_test, preds),
                    f"KAnon_MAE_K{k}": mean_absolute_error(y_test, preds),
                    f"KAnon_R2_K{k}": r2_score(y_test, preds),
                    f"KAnon_TrainTime_K{k}": train_time,
                    f"KAnon_InfTime_K{k}": inf_time,
                    f"KAnon_ProcessTime_K{k}": anon_duration,
                })
