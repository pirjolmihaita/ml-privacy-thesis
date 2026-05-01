import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from .utils import get_logger

"""
data_loader.py: Data Ingestion and Preprocessing Pipeline
Manages the lifecycle of datasets (Adult, CreditCard, Insurance, etc.). 
It handles automated cleaning, feature scaling, categorical encoding, 
and stratified splitting for both classification and regression tasks.
"""

logger = get_logger(__name__)

class DataLoader:
    def __init__(self, data_dir=None):
        if data_dir is None:
            # Default to ../data/raw relative to this file (src/data_loader.py)
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.data_dir = os.path.join(base_dir, 'data', 'raw')
        else:
            self.data_dir = data_dir
        
        self.datasets = {
            'adult': self._load_adult,
            'data': self._load_breast_cancer,
            'cervical': self._load_cervical,
            'compas': self._load_compas,
            'creditcard': self._load_creditcard,
            'heart': self._load_heart,
            'insurance': self._load_insurance,
            'communities': self._load_communities
        }
        self.regression_datasets = ['insurance', 'communities']

    def load_and_preprocess(self, name):
        if name not in self.datasets:
            raise ValueError(f"Dataset {name} not found.")
        
        logger.info(f"Loading dataset: {name}")
        df, target_col = self.datasets[name]()
        
        task_type = 'regression' if name in self.regression_datasets else 'classification'
        logger.info(f"Task Type: {task_type}")
        
        # General cleanup
        df = df.dropna(subset=[target_col])
        X = df.drop(columns=[target_col])
        y = df[target_col]

        logger.info(f"Dataset {name} shape: {X.shape}")
        
        # Split
        stratify = y if task_type == 'classification' else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )
        
        # Preprocessing Pipeline
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object', 'bool', 'category']).columns
        
        logger.info(f"Numeric features: {len(numeric_features)}")
        logger.info(f"Categorical features: {len(categorical_features)}")

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # sparse=False for diffprivlib compat
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
            
        return X_train, X_test, y_train, y_test, preprocessor, task_type

    def _load_adult(self):
        df = pd.read_csv(f'{self.data_dir}/classification/adult.csv')
        df = df.replace('?', np.nan)
        df = df.drop(columns=['education', 'fnlwgt'])
        df = df.drop_duplicates() 
        # Map income to 0/1
        df['income'] = df['income'].apply(lambda x: 1 if '>50K' in x else 0)
        return df, 'income'

    def _load_breast_cancer(self):
        df = pd.read_csv(f'{self.data_dir}/classification/data.csv')
        df = df.drop_duplicates()
        df = df.drop(columns=['id', 'Unnamed: 32'], errors='ignore')
        df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
        return df, 'diagnosis'

    def _load_cervical(self):
        df = pd.read_csv(f'{self.data_dir}/classification/kag_risk_factors_cervical_cancer.csv')
        df = df.replace('?', np.nan)
        df = df.dropna(thresh=len(df)*0.5, axis=1)
        # Convert all columns to numeric if possible, as most are numeric disguised as object
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')
        
        # Biopsy is target
        return df, 'Biopsy'

    def _load_compas(self):
        df = pd.read_csv(f'{self.data_dir}/classification/compas-scores-two-years.csv')
        # Filter similar to ProPublica methodology
        df = df[
            (df['days_b_screening_arrest'] <= 30) &
            (df['days_b_screening_arrest'] >= -30) &
            (df['is_recid'] != -1) &
            (df['c_charge_degree'] != "O")
        ]
        
        keep_cols = ['sex', 'age', 'age_cat', 'race', 'juv_fel_count', 'juv_misd_count', 
                     'juv_other_count', 'priors_count', 'c_charge_degree', 'two_year_recid']
        df = df[keep_cols]
        return df, 'two_year_recid'

    def _load_creditcard(self):
        df = pd.read_csv(f'{self.data_dir}/classification/creditcard.csv')
        df = df.drop(columns=['Time'])
        return df, 'Class'

    def _load_heart(self):
        df = pd.read_csv(f'{self.data_dir}/classification/heart.csv')
        return df, 'HeartDisease'

    def _load_insurance(self):
        df = pd.read_csv(f'{self.data_dir}/regression/insurance.csv')
        df = df.drop_duplicates()
        df['charges'] = np.log1p(df['charges'])
        return df, 'charges'

    def _load_communities(self):
        # 1994 Communities and Crime dataset
        # No header in .data file, so we expect names or index
        # We will drop the first 5 non-predictive columns: 
        # state, county, community, communityname, fold
        
        df = pd.read_csv(f'{self.data_dir}/regression/communities.data', header=None, na_values=['?'])
        df = df.dropna(thresh=len(df)*0.5, axis=1)
        # Based on .names file, the goal is the last column (127 index)
        # Drop first 5
        df = df.drop(columns=[0, 1, 2, 3, 4], errors='ignore')
        
        # Name the target. We'll leave features as integers for simplicity unless crucial
        df = df.rename(columns={df.columns[-1]: 'ViolentCrimesPerPop'})
        
        return df, 'ViolentCrimesPerPop'

if __name__ == "__main__":
    # Test loading
    dl = DataLoader()
    for name in ['adult', 'heart']: # Test a few
        try:
            X_tr, X_te, y_tr, y_te, prep = dl.load_and_preprocess(name)
            print(f"Successfully loaded {name}")
        except Exception as e:
            print(f"Failed {name}: {e}")
