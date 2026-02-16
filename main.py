import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))



try:
    from src.runner import run_experiments
    from src.computational_cost_analyzer import load_timing_data, process_and_organize_costs
    from src.privacy_utility_analyzer import load_utility_data, process_tradeoff


    
    def run_analysis():
        print("\nRunning Computational Cost Analysis...")
        timing_df = load_timing_data()
        process_and_organize_costs(timing_df)
        print("Computational Cost saved in: results/analysis/cost_computational/")

        print("\nRunning Privacy-Utility Tradeoff Analysis...")
        utility_df = load_utility_data()
        process_tradeoff(utility_df)
        print("Privacy-Utility saved in: results/analysis/privacy_tradeoff/")

    if __name__ == "__main__":
        if "--analysis-only" in sys.argv:
            run_analysis()
        else:
            print("Starting Thesis Project Experiments...")
            run_experiments()
            run_analysis()
except ImportError as e:
    print(f"Error importing src modules: {e}")
    print("Ensure you are running 'python main.py' from the project root.")
