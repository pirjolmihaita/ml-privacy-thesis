import numpy as np
import time
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Conditional imports to avoid crashing if not installed
try:
    from diffprivlib.models import LogisticRegression as DPLR
    from diffprivlib.models import LinearRegression as DPLinearReg
    from diffprivlib.models import GaussianNB as DPGNB
    from diffprivlib.models import DecisionTreeClassifier as DPDT
    from diffprivlib.models import RandomForestClassifier as DPRF
except ImportError:
    DPLR = DPGNB = DPDT = DPRF = None

try:
    from concrete.ml.sklearn import LogisticRegression as ConcreteLR
    from concrete.ml.sklearn import LinearRegression as ConcreteLinearReg
    from concrete.ml.sklearn import DecisionTreeClassifier as ConcreteDT
    from concrete.ml.sklearn import DecisionTreeRegressor as ConcreteDTReg
    from concrete.ml.sklearn import RandomForestClassifier as ConcreteRF
    from concrete.ml.sklearn import RandomForestRegressor as ConcreteRFReg
except ImportError:
    ConcreteLR = ConcreteLinearReg = None
    ConcreteDT = ConcreteDTReg = None
    ConcreteRF = ConcreteRFReg = None

try:
    from phe import paillier
except ImportError:
    paillier = None

from .utils import get_logger
from .mi_attack import run_mi_attack

logger = get_logger(__name__)

"""
models.py: Unified Model Factory and Inference Engine
The central library of the project. It abstracts the instantiation of all 
models (Baseline, DP, Concrete ML, Paillier) and encapsulates the complex 
logic required for running homomorphic and differentially private inference.
"""

class ModelManager:
    def __init__(self):
        pass

    def get_baseline_model(self, model_type):
        if model_type == 'lr':
            return LogisticRegression(solver='lbfgs', max_iter=1000, class_weight='balanced')
        elif model_type == 'lin_reg':
            return LinearRegression()
        elif model_type == 'nb':
            return GaussianNB(priors=[0.5, 0.5])
        elif model_type == 'dt':
            return DecisionTreeClassifier(max_depth=10, random_state=42, class_weight='balanced')
        elif model_type == 'dt_reg':
            return DecisionTreeRegressor(max_depth=10, random_state=42)
        elif model_type == 'rf':
            return RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42, class_weight='balanced')
        elif model_type == 'rf_reg':
            return RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def get_dp_model(self, model_type, epsilon, data_norm, n_features=None, classes=None, bounds=None, bounds_y=None):
        if bounds is None:
            bounds = (-data_norm, data_norm)
        if bounds_y is None:
            bounds_y = bounds

        if model_type == 'lr':
            return DPLR(epsilon=epsilon, data_norm=data_norm, max_iter=1000)
        elif model_type == 'lin_reg':
            return DPLinearReg(epsilon=epsilon, bounds_X=bounds, bounds_y=bounds_y)
        elif model_type == 'nb':
            # Fixed bounds at [-3, 3] — after StandardScaler features are in this range.
            # Using data_norm (e.g. 100) as bounds overestimates sensitivity
            # and produces excessive noise that makes the model more vulnerable to MIA.
            nb_bounds = (-3.0, 3.0)
            return DPGNB(epsilon=epsilon, bounds=nb_bounds)
        elif model_type == 'dt':
            return DPDT(epsilon=epsilon, max_depth=10, bounds=bounds, classes=classes)
        elif model_type == 'rf':
            return DPRF(epsilon=epsilon, n_estimators=50, max_depth=10, bounds=bounds, classes=classes)
        else:
            raise ValueError(f"Unknown DP model type: {model_type}")

    def get_concrete_model(self, model_type, n_estimators=10, max_depth=5):

        if model_type == 'lr':
            if ConcreteLR is None:
                raise ImportError("Concrete LogisticRegression not available")
            return ConcreteLR(n_bits=8)

        if model_type == 'lin_reg':
            if ConcreteLinearReg is None:
                raise ImportError("Concrete LinearRegression not available")
            return ConcreteLinearReg(n_bits=8)

        if model_type == 'dt':
            if ConcreteDT is None:
                raise ImportError("Concrete DecisionTree not available")
            return ConcreteDT(n_bits=8, max_depth=max_depth)

        if model_type == 'dt_reg':
            if ConcreteDTReg is None:
                raise ImportError("Concrete DecisionTreeRegressor not available")
            return ConcreteDTReg(n_bits=8, max_depth=max_depth)

        if model_type == 'rf':
            if ConcreteRF is None:
                raise ImportError("Concrete RandomForest not available")
            return ConcreteRF(n_bits=8, n_estimators=n_estimators, max_depth=max_depth)

        if model_type == 'rf_reg':
            if ConcreteRFReg is None:
                raise ImportError("Concrete RandomForestRegressor not available")
            return ConcreteRFReg(n_bits=8, n_estimators=n_estimators, max_depth=max_depth)

        raise ValueError(f"Unsupported Concrete model type: {model_type}")

    def run_concrete_inference(
        self,
        model_type,
        X_train,
        X_test,
        y_train,
        n_samples=10,
        fhe_mode="simulate",
        apply_dp_weights=False,
        dp_epsilon=None,
        data_norm=None,
        max_abs_weight=5.0,
        dp_mechanism="laplace",
        random_state=None
    ):
        """
        Compiles and runs FHE inference using Concrete ML.

        Notes:
        - Concrete ML models must be:
            1) fitted on clear (quantized training)
            2) compiled on representative calibration data
            3) executed using FHE simulation or real FHE
        - This function encapsulates the full Concrete ML lifecycle.

        Additional (optional) step:
        - If apply_dp_weights=True and model_type in ['lr', 'lin_reg'],
        we add DP-like noise (Laplace) to model coefficients AFTER training,
        then compile + run FHE inference on the noisy model.
        """
        # ---------------------------------------------------------
        # 0. Create Concrete model
        # ---------------------------------------------------------
        model = self.get_concrete_model(model_type)

        # ---------------------------------------------------------
        # 1. Fit (clear / quantized training)
        # ---------------------------------------------------------
        # Concrete ML models follow the sklearn API but operate on
        # quantized representations internally.
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        # ---------------------------------------------------------
        # 1.5 Optional: DP on coefficients (LR / LinearReg only)
        # ---------------------------------------------------------
        if apply_dp_weights:
            if model_type not in ["lr", "lin_reg"]:
                raise ValueError("apply_dp_weights=True este suportat doar pentru 'lr' / 'lin_reg' (modele cu coeficienți).")

            if dp_epsilon is None or float(dp_epsilon) <= 0:
                raise ValueError("dp_epsilon trebuie să fie > 0 când apply_dp_weights=True.")

            # RNG (reproducibilitaty)
            rng = np.random.default_rng(random_state)

            # 1.5.1 Extract coefficients
            if not hasattr(model, "coef_") or not hasattr(model, "intercept_"):
                raise ValueError("Modelul Concrete nu expune coef_ / intercept_. Nu pot aplica DP pe coeficienți.")

            coef = np.array(model.coef_, dtype=float, copy=True)
            intercept = np.array(model.intercept_, dtype=float, copy=True)

            # 1.5.2 Clip coefficients (stability)
            coef = np.clip(coef, -max_abs_weight, max_abs_weight)
            intercept = np.clip(intercept, -max_abs_weight, max_abs_weight)

            # 1.5.3 Add noise (Laplace by default)
            # Heuristic sensitivity: proportional to data_norm (if provided), else 1.0
            dn = float(data_norm) if data_norm is not None else 1.0
            sensitivity = dn
            scale = sensitivity / float(dp_epsilon)

            if dp_mechanism.lower() == "laplace":
                coef_noise = rng.laplace(loc=0.0, scale=scale, size=coef.shape)
                intercept_noise = rng.laplace(loc=0.0, scale=scale, size=intercept.shape)
            else:
                raise ValueError(f"dp_mechanism necunoscut: {dp_mechanism}. Folosește 'laplace'.")

            # 1.5.4 Set noisy coefficients back
            model.coef_ = coef + coef_noise
            model.intercept_ = intercept + intercept_noise

        # ---------------------------------------------------------
        # 2. Compile (mandatory for FHE)
        # ---------------------------------------------------------
        # Compilation builds the FHE circuit and calibrates
        # quantization parameters using representative data.
        # We use a subset for performance reasons.
        t0 = time.perf_counter()
        calibration_data = X_train[:100] if len(X_train) > 100 else X_train
        model.compile(calibration_data)
        compile_time = time.perf_counter() - t0

        # ---------------------------------------------------------
        # 3. FHE Inference (simulation or real execution)
        # ---------------------------------------------------------
        # We run inference only on a small subset to keep runtime manageable.
        X_subset = X_test[:n_samples]

        # IMPORTANT:
        # - fhe="simulate" → fast, deterministic, recommended for experiments
        # - fhe="execute"  → real FHE (slow, cryptographic execution)
        t0 = time.perf_counter()
        y_preds = model.predict(X_subset, fhe=fhe_mode)
        fhe_time = time.perf_counter() - t0

        return model, y_preds, train_time, compile_time, fhe_time




    def run_he_inference(self, model, X_test, n_samples=10):
        """
        Run encrypted inference using Paillier (PHE) on a subset of X_test.
        Model must be a trained sklearn/diffprivlib LogisticRegression or LinearRegression.
        """
        # Compatible check
        is_log_reg = isinstance(model, (DPLR, LogisticRegression)) and model.__class__.__name__ in ['LogisticRegression']
        is_lin_reg = isinstance(model, (LinearRegression, DPLinearReg)) or model.__class__.__name__ in ['LinearRegression']
        
        if not (is_log_reg or is_lin_reg):
            raise ValueError(f"HE inference only implemented for Logistic/Linear Regression (got {type(model)})")
        
        if paillier is None:
            raise ImportError("phe library not installed")

        public_key, private_key = paillier.generate_paillier_keypair()
        
        subset_X = X_test[:n_samples]
        t0 = time.perf_counter()
        encrypted_preds = []
        
        # Get coefficients (handle both sklearn and diffprivlib structure)
        if hasattr(model, 'coef_'):
            # Handle shape differences between Logistic (2D) and Linear Regression (1D)
            if model.coef_.ndim > 1:
                coefs = model.coef_[0]
            else:
                coefs = model.coef_
            
            # Handle intercept shape
            if hasattr(model.intercept_, '__len__') and len(model.intercept_) > 0:
                 intercept = model.intercept_[0]
            else:
                 intercept = model.intercept_
        else:
            raise ValueError("Model does not have coefficients for HE inference")

        for i in range(len(subset_X)):
            x = subset_X[i]
            # Encrypt input vector
            enc_x = [public_key.encrypt(float(val)) for val in x]
            
            # Homomorphic Dot Product
            enc_res = 0
            for j, val in enumerate(x):
                enc_res += coefs[j] * enc_x[j]
            enc_res += intercept
            
            encrypted_preds.append(enc_res)
            
        inference_time = time.perf_counter() - t0
        
        decrypted_scores = [private_key.decrypt(enc) for enc in encrypted_preds]
        
        preds = []
        if is_log_reg:
            preds = [1 if (1 / (1 + np.exp(-s))) > 0.5 else 0 for s in decrypted_scores]
        else:
            # Linear Regression: Score is the prediction
            preds = decrypted_scores
        
        return preds, inference_time
    