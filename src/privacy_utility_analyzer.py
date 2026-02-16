import pandas as pd
import os
import re

"""
privacy_utility_analizer.py: Utility Trade-off Analysis
Aggregates experimental metrics to identify the optimal balance between 
privacy levels (Epsilon/K) and model utility. It extracts the best performance 
scores for each method to facilitate comparative research analysis.
"""

# Paths configurations
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_PATH = os.path.join(BASE_DIR, 'results', 'metrics', 'results_wide.csv')
BASE_TRADE_DIR = os.path.join(BASE_DIR, 'results', 'analysis', 'privacy_tradeoff')

def load_utility_data():
    if not os.path.exists(METRICS_PATH):
        print(f"Error: {METRICS_PATH} not found.")
        return None

    df = pd.read_csv(METRICS_PATH)
    id_vars = ['Dataset', 'Model', 'Task_Type']
    
    # Regex patterns to capture Metric, Epsilon, and Norm (Utility)
    patterns = {
        'Baseline': re.compile(r"^Baseline_(F1|R2)$"),
        'DP': re.compile(r"^DP_(F1|R2)_Eps([0-9.]+)_Norm([0-9.]+)$"),
        'DP-PHE': re.compile(r"^PHE_(F1|R2)_Eps([0-9.]+)_Norm([0-9.]+)$"),
        'DP-FHE': re.compile(r"^Concrete_(F1|R2)_Eps([0-9.]+)_Norm([0-9.]+)$"),
        'DP-FHE-W': re.compile(r"^ConcreteW_(F1|R2)_Eps([0-9.]+)_Norm([0-9.]+)$")
    }

    rows = []
    for _, row in df.iterrows():
        base_info = {col: row[col] for col in id_vars}
        for col in df.columns:
            if col in id_vars: continue
            
            val = pd.to_numeric(row[col], errors='coerce')
            if pd.isna(val): continue

            for method_name, pattern in patterns.items():
                match = pattern.match(col)
                if match:
                    metric_name = match.group(1)
                    eps = match.group(2) if len(match.groups()) >= 2 else "None"
                    norm = match.group(3) if len(match.groups()) >= 3 else "None"
                    
                    rows.append({
                        **base_info, 
                        'Method': method_name, 
                        'Metric_Type': metric_name, 
                        'Score': val,
                        'Epsilon': eps,
                        'Data_Norm': norm
                    })
                    break
    return pd.DataFrame(rows)

def process_tradeoff(df):
    if df is None or df.empty: return

    # For each Dataset, Model, and Method, find the row with the best score
    # Classification -> maximum F1, Regression -> maximum R2
    idx = df.groupby(['Dataset', 'Model', 'Method'])['Score'].idxmax()
    best_results = df.loc[idx]

    for (ds, task), ds_df in best_results.groupby(['Dataset', 'Task_Type']):
        # Create the folder: analysis/privacy_tradeoff/classification/adult/
        folder_path = os.path.join(BASE_TRADE_DIR, task, ds)
        os.makedirs(folder_path, exist_ok=True)

        # Sort the DataFrame by Score in descending order to see the best method at the top
        ds_df = ds_df.sort_values(by='Score', ascending=False)

        # Rename the Score column based on the task for clarity
        metric_label = 'Best_F1_Score' if task == 'classification' else 'Best_R2_Score'
        ds_df = ds_df.rename(columns={'Score': metric_label})

        output_file = os.path.join(folder_path, 'tabel_utility.csv')
        
        # Select the final columns
        final_cols = ['Method', 'Model', metric_label, 'Epsilon', 'Data_Norm']
        ds_df[final_cols].to_csv(output_file, index=False)
        print(f"Saved: {task}/{ds}/tabel_utility.csv")

if __name__ == "__main__":
    print("Analiză Privacy-Utility Tradeoff în curs...")
    utility_df = load_utility_data()
    process_tradeoff(utility_df)
    print("\nFinalizat! Rezultatele sunt în results/analysis/privacy_tradeoff/")