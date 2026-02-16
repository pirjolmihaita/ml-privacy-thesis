import pandas as pd
import os
import re

# Main path configurations
# BASE_DIR points to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Path to the raw results file
METRICS_PATH = os.path.join(BASE_DIR, 'results', 'metrics', 'results_wide.csv')
# Root directory for the computational cost analysis output
BASE_COST_DIR = os.path.join(BASE_DIR, 'results', 'analysis', 'cost_computational')

"""
computational_cost_analyzer.py: Performance and Overhead Analysis
This script parses raw metrics to calculate total computational costs across 
different privacy layers. It specifically computes training, inference, and 
compilation times, generating overhead ratios relative to baseline performance.
"""

# Dataset row counts (Total Rows) for context in the final reports
DATASET_INFO = {
    'adult': 32562, 'creditcard': 284808, 'heart': 919,
    'insurance': 1339, 'data': 570, 'compas': 7214,
    'communities': 1994, 'cervical': 859
}

def load_timing_data():
    """
    Loads results_wide.csv and extracts only timing-related metrics using regex.
    """
    if not os.path.exists(METRICS_PATH):
        print(f"Error: {METRICS_PATH} not found.")
        return None

    df = pd.read_csv(METRICS_PATH)
    id_vars = ['Dataset', 'Model', 'Task_Type']
    
    # Strict regex patterns to identify timing metrics across different methods
    patterns = {
        'Baseline': re.compile(r"^Baseline_(TrainTime|InfTime)$"),
        'DP-PHE': re.compile(r"^PHE_(TrainTime|InfTime)_Eps([0-9.]+)_Norm([0-9.]+)$"),
        'DP-FHE': re.compile(r"^Concrete_(TrainTime|InfTime|CompileTime)_Eps([0-9.]+)_Norm([0-9.]+)$"),
        'DP-FHE-W': re.compile(r"^ConcreteW_(TrainTime|InfTime|CompileTime)_Eps([0-9.]+)_Norm([0-9.]+)$")
    }

    rows = []
    for _, row in df.iterrows():
        base_info = {col: row[col] for col in id_vars}
        for col in df.columns:
            if col in id_vars: continue
            
            # Ensure numeric conversion to handle potential string/formatting errors
            val = pd.to_numeric(row[col], errors='coerce')
            if pd.isna(val) or val <= 0: continue

            for method_name, pattern in patterns.items():
                match = pattern.match(col)
                if match:
                    # Capture Method, Metric type (Train/Inf/Compile), and the time in seconds
                    rows.append({**base_info, 'Method': method_name, 'Metric': match.group(1), 'Seconds': val})
                    break
    return pd.DataFrame(rows)

def process_and_organize_costs(df):
    """
    Pivots the data, calculates overhead ratios, and saves files into a 
    structured directory hierarchy: task_type / dataset / tabel_cost.csv
    """
    if df is None or df.empty: return

    # Pivot the table so that different timing metrics become individual columns
    pivot = df.pivot_table(
        index=['Dataset', 'Model', 'Task_Type', 'Method'],
        columns='Metric',
        values='Seconds',
        aggfunc='mean'
    ).reset_index().fillna(0)

    # Calculate total execution time by summing available phases (Train, Inf, and Compile)
    time_cols = [c for c in ['TrainTime', 'InfTime', 'CompileTime'] if c in pivot.columns]
    pivot['Total_Time_Sec'] = pivot[time_cols].sum(axis=1)

    # Calculate Overhead Factor relative to the Baseline of the same Model/Dataset
    baseline_costs = pivot[pivot['Method'] == 'Baseline'].set_index(['Dataset', 'Model'])['Total_Time_Sec'].to_dict()

    def get_overhead(row):
        base_val = baseline_costs.get((row['Dataset'], row['Model']))
        # Ratio of (Total Privacy Method Time) / (Total Baseline Time)
        return row['Total_Time_Sec'] / base_val if base_val else 1.0

    pivot['Overhead_Factor'] = pivot.apply(get_overhead, axis=1)

    # Organize outputs into folders: cost_computational / task_type / dataset / tabel_cost.csv
    for (ds, task), ds_df in pivot.groupby(['Dataset', 'Task_Type']):
        ds_df = ds_df.copy()
        # Attach total row count to the current dataset rows
        ds_df['Dataset_Rows'] = ds_df['Dataset'].map(DATASET_INFO)
        
        # Sort from cheapest (Baseline) to most expensive (highest overhead)
        ds_df = ds_df.sort_values('Overhead_Factor')

        # Define specific folder path for each dataset and task
        folder_path = os.path.join(BASE_COST_DIR, task, ds)
        os.makedirs(folder_path, exist_ok=True)

        # Select the final columns for the CSV report
        final_cols = ['Method', 'Model', 'Dataset_Rows'] + time_cols + ['Total_Time_Sec', 'Overhead_Factor']
        
        output_file = os.path.join(folder_path, 'tabel_cost.csv')
        ds_df[final_cols].to_csv(output_file, index=False)
        print(f"Saved: {task}/{ds}/tabel_cost.csv")

if __name__ == "__main__":
    print("Starting detailed organization of computational costs...")
    timing_df = load_timing_data()
    process_and_organize_costs(timing_df)
    print("\nStructure successfully created in results/analysis/cost_computational/")