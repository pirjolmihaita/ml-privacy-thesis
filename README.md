# Confidentiality-of-Data: Comparative Analysis of PPML Techniques

This project provides a comprehensive framework for evaluating Privacy-Preserving Machine Learning (PPML) techniques. It compares traditional machine learning models (Baseline) against three major privacy paradigms, measuring the trade-off between data utility and computational overhead.

## Overview of Privacy Techniques

The system evaluates four distinct execution modes:

1. **Baseline**: Standard machine learning models trained and tested on cleartext data. This serves as the performance and accuracy benchmark.
2. **Differential Privacy (DP)**: Implemented via the `diffprivlib` library. It adds mathematical noise to the model or data to provide formal privacy guarantees (Epsilon-level privacy).
3. **Partially Homomorphic Encryption (PHE)**: Implemented using the Paillier cryptosystem. It allows linear models to perform encrypted inference, where the data remains encrypted while the model weights are in cleartext.
4. **Fully Homomorphic Encryption (FHE)**: Implemented via **Concrete ML**. This converts machine learning models into cryptographic boolean circuits, allowing for total data confidentiality during execution.
5. **K-Anonymity**: Data generalization before training, evaluated with k ∈ {2, 10, 50, 100}.

## Research Comparison
The project compares these techniques across multiple datasets (Adult, CreditCard, Heart, Insurance, Communities, etc.) using the following metrics:

* **Utility**: F1-Score and AUC for classification, R2-Score for regression.
* **Execution Time**: Training time, inference time, and FHE compilation time.
* **Computational Overhead**: Slowdown factor relative to Baseline.
* **Membership Inference Attack (MIA)**: Black-box attack accuracy measuring privacy leakage — 0.50 = fully protected, 1.0 = fully vulnerable.

## Environment Setup (WSL)

The project is designed to run in a Linux environment via the Windows Subsystem for Linux (WSL).

### 1. Virtual Environment Configuration
To ensure all dependencies are isolated, create and activate a virtual environment:

```bash
# Create the virtual environment
python3 -m venv venv_wsl

# Activate the environment
source venv_wsl/bin/activate

# Install required libraries
pip install -r requirements.txt
```

### 2. Git LFS Configuration (Crucial for Datasets)
This project uses Git Large File Storage (LFS) to manage large datasets (e.g., creditcard.csv, adult.csv). If you have just cloned the repository, the files in data/raw/ are currently just small pointer files.

Run the following commands to download the actual data:

# Ensure git-lfs is initialized on your system
```bash
git lfs install
git lfs pull
```

### 3. Execution
To run the full pipeline, ensure your virtual environment is active:

```bash
source venv_wsl/bin/activate
```

# Run the complete experiment suite
```bash
python main.py
```

## Results and Output Location
After completion, results are automatically uploaded to a Google Cloud Storage bucket:
- results_wide.csv — full metrics for all methods and datasets
- resource_usage.csv — CPU, RAM and timing per method
- results_membership_inference.csv — MIA results

If upload credentials are not configured, this step is skipped and results remain available locally.
