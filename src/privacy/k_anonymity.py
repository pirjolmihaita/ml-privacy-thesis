import pandas as pd
import numpy as np

class MondrianAnonymizer:
    def __init__(self, k=10):
        self.k = k
        self.partitions = []
        self.categorical_features = []
        self.feature_spans = {} # Stores ranges/sets for each feature in each partition

    def fit(self, X, categorical_features=None):
        """
        X: pandas DataFrame
        categorical_features: list of column names
        """
        self.partitions = []
        self.categorical_features = categorical_features if categorical_features else []
        
        # Initial partition index - we will start with one partition containing all indices
        initial_indices = X.index.to_numpy()
        self._partition(X, initial_indices)
        
        return self

    def _partition(self, df, indices):
        """
        Recursively split the dataframe indices until size < 2*k
        """
        if len(indices) < 2 * self.k:
            self.partitions.append(indices)
            return

        # Choose dimension to split
        # Heuristic: Choose dimension with widest normalized span
        # Simple heuristic here: just iterate or choose distinct count max
        
        # Let's find columns with valid splits
        valid_cols = []
        for col in df.columns:
            # Check unique values
            unique_vals = df.loc[indices, col].nunique()
            if unique_vals > 1:
                valid_cols.append(col)
        
        if not valid_cols:
             self.partitions.append(indices)
             return

        # Pick a column to split (random or max variance/span)
        # For Mondrian, traditionally choosing the dimension with widest normalized range
        split_col = np.random.choice(valid_cols) # Simplified for now
        
        # Split value
        if split_col in self.categorical_features:
            # Mondrian requires ordered values; categorical columns cannot be split by median.
            # Fallback: keep partition as-is.
             self.partitions.append(indices)
             return
        else:
            # Numeric Median Split
            values = df.loc[indices, split_col]
            median = values.median()
            
            lhs = values[values <= median].index
            rhs = values[values > median].index
            
            if len(lhs) < self.k or len(rhs) < self.k:
                 self.partitions.append(indices) 
                 return
            
            self._partition(df, lhs.to_numpy())
            self._partition(df, rhs.to_numpy())

    def transform(self, X):
        """
        Apply generalization based on partitions computed in fit().
        Numeric columns are replaced with partition mean; categorical with partition mode.
        Note: works only on the same X passed to fit() (index-based).
        In the experiment, train+test are concatenated before anonymization, then split back.
        """
        X_out = X.copy()
        # Ensure numeric columns are float to avoid int coercion warnings when setting means
        for col in X_out.columns:
            if col not in self.categorical_features and pd.api.types.is_numeric_dtype(X_out[col]):
                 X_out[col] = X_out[col].astype(float)

        for partition in self.partitions:
            # Calculate representative
            for col in X.columns:
                if col in self.categorical_features:
                    # Generalize categorical: use mode to keep feature usable for ML
                    mode_val = X.loc[partition, col].mode()[0]
                    X_out.loc[partition, col] = mode_val
                else:
                    mean_val = X.loc[partition, col].mean()
                    X_out.loc[partition, col] = mean_val
                    
        return X_out

if __name__ == "__main__":
    # Test
    df = pd.DataFrame({
        'age': [20, 21, 22, 60, 61, 62],
        'income': [10, 11, 12, 50, 51, 52],
        'sex': ['M','M','M','F','F','F']
    })
    
    mondrian = MondrianAnonymizer(k=2)
    mondrian.fit(df, categorical_features=['sex'])
    df_anon = mondrian.transform(df)
    print(df_anon)
