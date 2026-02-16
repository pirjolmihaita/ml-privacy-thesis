# Confidentiality-of-Data: Comparative Analysis of PPML Techniques

This project provides a comprehensive framework for evaluating Privacy-Preserving Machine Learning (PPML) techniques. It compares traditional machine learning models (Baseline) against three major privacy paradigms, measuring the trade-off between data utility and computational overhead.

## Overview of Privacy Techniques

The system evaluates four distinct execution modes:

1. **Baseline**: Standard machine learning models trained and tested on cleartext data. This serves as the performance and accuracy benchmark.
2. **Differential Privacy (DP)**: Implemented via the `diffprivlib` library. It adds mathematical noise to the model or data to provide formal privacy guarantees (Epsilon-level privacy).
3. **Partially Homomorphic Encryption (PHE)**: Implemented using the Paillier cryptosystem. It allows linear models to perform encrypted inference, where the data remains encrypted while the model weights are in cleartext.
4. **Fully Homomorphic Encryption (FHE)**: Implemented via **Concrete ML**. This converts machine learning models into cryptographic boolean circuits, allowing for total data confidentiality during execution.

## Research Comparison
The project compares these techniques across multiple datasets (Adult, CreditCard, Heart, Insurance, Communities, etc.) using the following metrics:

* **Utility**: F1-Score for classification tasks and R2-Score for regression tasks.
* **Execution Time**: Training time, inference time, and FHE compilation time.
* **Computational Overhead**: The relative slowdown factor of privacy-preserving methods compared to the Baseline.

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

#. Execution
To run the full pipeline, including data loading, preprocessing, privacy experiments, and automated report generation, ensure your virtual environment is active and execute the main entry point:
# Ensure you are in the project root and venv is active
source venv_wsl/bin/activate

# Run the complete experiment suite
python main.py

##Results and Output Location

The automated analyzers generate research-ready tables in the following folders:

Utility Analysis:
Located in results/analysis/privacy_tradeoff/. This folder contains tabel_utility.csv files for each dataset, highlighting the best F1 or R2 scores achieved across different privacy settings.

Computational Cost Analysis:
Located in results/analysis/computational_costs/. This folder contains tabel_cost.csv files for each dataset, providing a detailed breakdown of training times, inference times, and the calculated Overhead Factor compared to the Baseline.