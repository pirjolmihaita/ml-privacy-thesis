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
        'Baseline': re.compile(r"^Baseline_(F1|R2|AUC)$"),
        'DP': re.compile(r"^DP_(F1|R2|AUC)_Eps([0-9.]+)_Norm([0-9.]+)$"),
        'DP-PHE': re.compile(r"^DP_PHE_(F1|R2|AUC)_Eps([0-9.]+)_Norm([0-9.]+)$"),
        'DP-FHE': re.compile(r"^FHE_(F1|R2|AUC)_Eps([0-9.]+)_Norm([0-9.]+)$"),
        'DP-FHE-W': re.compile(r"^FHE_DP_(F1|R2|AUC)_Eps([0-9.]+)_Norm([0-9.]+)$")
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

    for (ds, task), ds_df in df.groupby(['Dataset', 'Task_Type']):
        folder_path = os.path.join(BASE_TRADE_DIR, task, ds)
        os.makedirs(folder_path, exist_ok=True)

        if task == 'classification':
            # Best F1 per (Model, Method)
            f1_df = ds_df[ds_df['Metric_Type'] == 'F1']
            auc_df = ds_df[ds_df['Metric_Type'] == 'AUC']

            best_f1 = f1_df.loc[f1_df.groupby(['Model', 'Method'])['Score'].idxmax()] \
                           .rename(columns={'Score': 'Best_F1_Score'}) \
                           [['Method', 'Model', 'Best_F1_Score', 'Epsilon', 'Data_Norm']]

            best_auc = auc_df.loc[auc_df.groupby(['Model', 'Method'])['Score'].idxmax()] \
                             .rename(columns={'Score': 'Best_AUC_Score'}) \
                             [['Method', 'Model', 'Best_AUC_Score', 'Epsilon', 'Data_Norm']]

            # Merge F1 and AUC on Method + Model
            result = best_f1.merge(best_auc, on=['Method', 'Model'], suffixes=('_f1', '_auc'))
            result = result.sort_values('Best_F1_Score', ascending=False)

            final_cols = ['Method', 'Model', 'Best_F1_Score', 'Epsilon_f1', 'Data_Norm_f1',
                          'Best_AUC_Score', 'Epsilon_auc', 'Data_Norm_auc']

        else:
            # Regression: best R2 per (Model, Method)
            r2_df = ds_df[ds_df['Metric_Type'] == 'R2']
            best_r2 = r2_df.loc[r2_df.groupby(['Model', 'Method'])['Score'].idxmax()] \
                           .rename(columns={'Score': 'Best_R2_Score'}) \
                           [['Method', 'Model', 'Best_R2_Score', 'Epsilon', 'Data_Norm']]
            result = best_r2.sort_values('Best_R2_Score', ascending=False)
            final_cols = ['Method', 'Model', 'Best_R2_Score', 'Epsilon', 'Data_Norm']

        output_file = os.path.join(folder_path, 'tabel_utility.csv')
        result[final_cols].to_csv(output_file, index=False)
        print(f"Saved: {task}/{ds}/tabel_utility.csv")

if __name__ == "__main__":
    print("Privacy-Utility Tradeoff analysis running...")
    utility_df = load_utility_data()
    process_tradeoff(utility_df)
    print("\nDone! Results saved to results/analysis/privacy_tradeoff/")