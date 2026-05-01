import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def upload_results_to_gcs():
    try:
        from google.cloud import storage
        key_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'confidentiality-of-data-f9f87fd4ab4d.json')
        client = storage.Client.from_service_account_json(key_path)
        bucket = client.bucket('confidentiality-of-data-results')

        files = [
            'results/metrics/results_wide.csv',
            'results/metrics/resource_usage.csv',
            'results_membership_inference.csv',
        ]
        for f in files:
            if os.path.exists(f):
                bucket.blob(os.path.basename(f)).upload_from_filename(f)
                print(f"Uploaded: {f}")
            else:
                print(f"Skipped (not found): {f}")
        print("Upload to GCS complete.")
    except Exception as e:
        print(f"GCS upload failed: {e}")



try:
    from src.runner import run_experiments
    from src.computational_cost_analyzer import load_timing_data, process_and_organize_costs
    from src.privacy_utility_analyzer import load_utility_data, process_tradeoff
    from src.membership_inference_experiment import main as run_membership_inference


    
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

            print("\nRunning Membership Inference Attack Experiment...")
            run_membership_inference()
            print("Membership Inference results saved in: results_membership_inference.csv")

            print("\nUploading results to Google Cloud Storage...")
            upload_results_to_gcs()
except ImportError as e:
    print(f"Error importing src modules: {e}")
    print("Ensure you are running 'python main.py' from the project root.")
