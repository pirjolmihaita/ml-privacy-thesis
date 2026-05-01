"""
mi_attack.py
============
Membership Inference Attack utility using IBM ART.
Shared module used by both the experiment script and concrete inference.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from art.estimators.classification import BlackBoxClassifier
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox


def run_mi_attack(clf, X_train, y_train, X_test, y_test):
    """
    Runs a Membership Inference BlackBox Attack and returns attack accuracy.
    ~50% = random guessing (model is well protected)
    ~100% = attacker can identify training members (model is vulnerable)

    Attack train/eval split: 50/50 on both train and test sets to avoid
    in-sample evaluation of the attack model.
    """
    n_classes  = len(np.unique(y_train))
    n_features = X_train.shape[1]

    # Split for attack training and evaluation (avoids in-sample evaluation)
    X_tr_attack, X_tr_eval, y_tr_attack, y_tr_eval = train_test_split(
        X_train, y_train, test_size=0.5, random_state=42)
    X_te_attack, X_te_eval, y_te_attack, y_te_eval = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42)

    art_model = BlackBoxClassifier(
        predict_fn=lambda x: clf.predict_proba(x),
        input_shape=(n_features,),
        nb_classes=n_classes,
    )

    attack = MembershipInferenceBlackBox(art_model, attack_model_type="rf")
    attack.fit(X_tr_attack, y_tr_attack, X_te_attack, y_te_attack)

    inferred_train = attack.infer(X_tr_eval, y_tr_eval)
    inferred_test  = attack.infer(X_te_eval, y_te_eval)

    # Balanced evaluation: subsample to equal members/non-members so the
    # trivial "predict all as member" baseline is 0.50, not ~0.80.
    n_eval = min(len(inferred_train), len(inferred_test))
    rng = np.random.default_rng(42)
    idx_tr = rng.choice(len(inferred_train), size=n_eval, replace=False)
    idx_te = rng.choice(len(inferred_test),  size=n_eval, replace=False)

    correct_train = np.sum(inferred_train[idx_tr] == 1)
    correct_test  = np.sum(inferred_test[idx_te]  == 0)
    total         = n_eval * 2

    return round((correct_train + correct_test) / total, 4)
