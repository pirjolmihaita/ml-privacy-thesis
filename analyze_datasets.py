import pandas as pd
import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

from src.data_loader import DataLoader

def analyze_datasets():
    print("Initializing DataLoader...")
    try:
        dl = DataLoader()
    except Exception as e:
        print(f"Error initializing DataLoader: {e}")
        return

    print(f"\n{'='*100}")
    print(f"{'Dataset':<15} | {'Type':<15} | {'Rows':<8} | {'Cols':<5} | {'Missing Rows (%)':<18} | {'Target Distribution / Stats'}")
    print(f"{'='*100}")

    for name in dl.datasets:
        try:
            # We access the load function directly to get the dataframe before preprocessing
            # However, looking at data_loader.py, the load functions are stored in self.datasets
            load_func = dl.datasets[name]
            
            # Load the data
            df, target_col = load_func()
            
            # Basic stats
            n_rows, n_cols = df.shape
            
            # Missing values
            rows_with_nulls = df.isnull().any(axis=1).sum()
            pct_missing = (rows_with_nulls / n_rows) * 100
            
            # Target info
            task_type = 'regression' if name in dl.regression_datasets else 'classification'
            
            target_info = "N/A"
            if target_col in df.columns:
                if task_type == 'classification':
                    # Value counts for classification
                    counts = df[target_col].value_counts(normalize=True)
                    # Format as "Class: Pct%"
                    # If too many classes, just show top 2 or range
                    if len(counts) <= 5:
                        formatted_counts = [f"{cls}: {pct:.1%}" for cls, pct in counts.items()]
                        target_info = ", ".join(formatted_counts)
                    else:
                        target_info = f"{len(counts)} classes. Top: {counts.index[0]} ({counts.iloc[0]:.1%})"
                else:
                    # Stats for regression
                    mean_val = df[target_col].mean()
                    std_val = df[target_col].std()
                    target_info = f"Mean: {mean_val:.2f}, Std: {std_val:.2f}"
            else:
                target_info = f"Target '{target_col}' not found in df"

            print(f"{name:<15} | {task_type:<15} | {n_rows:<8} | {n_cols:<5} | {rows_with_nulls} ({pct_missing:.1f}%)   | {target_info}")

        except Exception as e:
            print(f"{name:<15} | ERROR: {str(e)}")

    print(f"{'='*100}")

if __name__ == "__main__":
    analyze_datasets()
