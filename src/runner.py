import os
import time
import pandas as pd
import numpy as np
import concurrent.futures
from .data_loader import DataLoader
from .models import ModelManager
from .utils import get_logger

from .common import make_suffix, get_unique_classes, get_bounds, slice_he_features
from .baseline import run_baseline
from .dp import run_dp
from .dp_phe import run_dp_phe
from .concrete import run_concrete_fhe_only, run_concrete_dp_weights
from .k_anonymity import run_k_anonymity_block

logger = get_logger(__name__)

# Config: Checkpoint Directory
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints')

def get_checkpoint_path(ds_name):
    """Returns the path to the checkpoint file for a given dataset."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    return os.path.join(CHECKPOINT_DIR, f"{ds_name}_results.csv")

def check_checkpoint(ds_name):
    """Checks if a checkpoint exists for the dataset."""
    path = get_checkpoint_path(ds_name)
    if os.path.exists(path):
        logger.info(f"Checkpoint found for {ds_name}, loading results...")
        try:
            df = pd.read_csv(path)
            return df.to_dict('records')
        except Exception as e:
            logger.error(f"Failed to read checkpoint for {ds_name}: {e}")
            return None
    return None

def save_checkpoint(ds_name, results_list):
    """Saves the results for a specific dataset to a CSV file."""
    if not results_list:
        return
    path = get_checkpoint_path(ds_name)
    try:
        pd.DataFrame(results_list).to_csv(path, index=False)
        logger.info(f"Saved checkpoint for {ds_name} at {path}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint for {ds_name}: {e}")

def process_dataset(ds_name):
    """
    Worker function to process a single dataset.
    Runs all models, privacy settings, and metrics.
    """
    # Initialize logger within the process (though it might duplicate logs to console depending on config)
    # Using the global logger from import is usually fine in fork-based multiprocessing, 
    # but in 'spawn' (default on Windows) we might need to re-setup. 
    # For now, we assume standard logging setup handles this or we rely on stdout.
    
    logger.info(f"Worker started for: {ds_name}")

    # 1. Check for existing checkpoint
    existing_results = check_checkpoint(ds_name)
    if existing_results is not None:
        logger.info(f"Skipping {ds_name} as it was already processed.")
        return existing_results

    # 2. Initialize Resources (DataLoader, ModelManager) inside worker
    dl = DataLoader()
    mm = ModelManager()

    datasets = ['adult', 'data', 'cervical', 'compas', 'creditcard', 'heart', 'insurance', 'communities']
    epsilons = [0.1, 0.5, 1.0, 5.0, 10.0]
    data_norms = [1.0, 10, 100]
    ks = [2, 10, 50, 100]
    models = ['lr', 'nb', 'dt', 'rf']

    # Local results storage for this dataset
    # keyed by tuple signature to allow updates (like baseline + DP metrics merging)
    wide_results = {}

    try:
        try:
            X_train, X_test, y_train, y_test, preprocessor, task_type = dl.load_and_preprocess(ds_name)
        except Exception as e:
            logger.error(f"[{ds_name}] Failed to load: {e}")
            return []

        logger.info(f"[{ds_name}] Task: {task_type}")

        # FIT PREPROCESSOR
        X_train_proc = preprocessor.fit_transform(X_train)
        X_test_proc = preprocessor.transform(X_test)
        n_features = X_train_proc.shape[1]

        # HE/FHE feature subset (top 10 columns)
        he_n_features_limit = 10
        X_train_he, X_test_he, full_n_features, he_used = slice_he_features(
            X_train_proc, X_test_proc, he_n_features_limit=he_n_features_limit
        )
        """
        if full_n_features > he_n_features_limit:
            logger.info(f"[{ds_name}] HE Optimization: Selected top {he_n_features_limit} features from {full_n_features}")
        """
        # --- BASELINE LOOP ---
        for m_key in models:
            if task_type == 'regression':
                if m_key == 'lr': m_type = 'lin_reg'
                elif m_key == 'nb': continue
                elif m_key == 'dt': m_type = 'dt_reg'
                elif m_key == 'rf': m_type = 'rf_reg'
                else: continue
            else:
                m_type = m_key

            row_key = (ds_name, m_type, 'Baseline', 'None', 'None')
            if row_key not in wide_results:
                wide_results[row_key] = {
                    'Dataset': ds_name, 'Model': m_type, 'Type': 'Baseline',
                    'Epsilon': 'None', 'Data_Norm': 'None', 'Task_Type': task_type
                }

            logger.info(f"[{ds_name}] Baseline: {m_type}")
            try:
                base_metrics = run_baseline(mm, m_type, X_train_proc, X_test_proc, y_train, y_test, task_type)
                wide_results[row_key].update(base_metrics)
            except Exception as e:
                logger.error(f"[{ds_name}] Failed Baseline {m_type}: {e}")

        # --- DP & HE LOOP ---
        for eps in epsilons:
            for norm in data_norms:
                for m_key in models:
                    if task_type == 'regression':
                        if m_key == 'lr': m_type = 'lin_reg'
                        else: continue
                    else:
                        m_type = m_key

                    suffix = make_suffix(eps, norm)
                    row_key = (ds_name, m_type, 'DP', eps, norm)
                    if row_key not in wide_results:
                        wide_results[row_key] = {
                            'Dataset': ds_name, 'Model': m_type, 'Type': 'DP',
                            'Epsilon': eps, 'Data_Norm': norm, 'Task_Type': task_type
                        }

                    logger.info(f"[{ds_name}] DP {m_type} (eps={eps}, norm={norm})")

                    try:
                        unique_classes = get_unique_classes(y_train, task_type)
                        dp_bounds = get_bounds(norm)

                        # 1) DP Train + Predict (full features)
                        clf_dp, dp_metrics = run_dp(
                            mm=mm, m_type=m_type, eps=eps, norm=norm,
                            n_features=n_features, classes=unique_classes, bounds=dp_bounds,
                            X_train_proc=X_train_proc, X_test_proc=X_test_proc,
                            y_train=y_train, y_test=y_test,
                            task_type=task_type, suffix=suffix
                        )
                        wide_results[row_key].update(dp_metrics)

                        # 2) PHE (DP + PHE) subset 10 features
                        # Force strict sample limit of 10
                        logger.info(f"[{ds_name}] Running HE Inference (PHE)...")
                        try:
                            phe_metrics = run_dp_phe(
                                mm=mm, m_type=m_type, eps=eps, norm=norm,
                                X_train_he=X_train_he, X_test_he=X_test_he,
                                y_train=y_train, y_test=y_test,
                                task_type=task_type, suffix=suffix, he_subset_n=10
                            )
                            wide_results[row_key].update(phe_metrics)
                        except Exception as e:
                            logger.error(f"[{ds_name}] PHE Failed: {e}")

                        # 3) Concrete ML (FHE-only) + ConcreteW (DP+weights)
                        # Force strict sample limit of 10
                        try:
                            conc_metrics = run_concrete_fhe_only(
                                mm=mm, m_type=m_type,
                                X_train_he=X_train_he, X_test_he=X_test_he,
                                y_train=y_train, y_test=y_test,
                                task_type=task_type, suffix=suffix, he_subset_n=10
                            )
                            wide_results[row_key].update(conc_metrics)
                        except ImportError:
                            pass
                        except Exception as e:
                            logger.error(f"[{ds_name}] Concrete FHE-only Failed {m_type}: {e}")

                        try:
                            concw_metrics = run_concrete_dp_weights(
                                mm=mm, m_type=m_type, eps=eps, norm=norm,
                                X_train_he=X_train_he, X_test_he=X_test_he,
                                y_train=y_train, y_test=y_test,
                                task_type=task_type, suffix=suffix, he_subset_n=10
                            )
                            wide_results[row_key].update(concw_metrics)
                        except ImportError:
                            pass
                        except Exception as e:
                            logger.error(f"[{ds_name}] Concrete DP+FHE (Weights) Failed {m_type}: {e}")

                    except Exception as e:
                        logger.error(f"[{ds_name}] DP Failed {m_type}: {e}")

        # --- K-ANONYMITY BLOCK ---
        try:
            run_k_anonymity_block(
                mm=mm,
                preprocessor=preprocessor,
                X_train=X_train, X_test=X_test,
                y_train=y_train, y_test=y_test,
                task_type=task_type,
                models=models,
                ks=ks,
                wide_results=wide_results,
                ds_name=ds_name
            )
        except Exception as e:
            logger.error(f"[{ds_name}] K-Anonymity block failed: {e}")

        # Save Checkpoint for this dataset
        results_list = list(wide_results.values())
        save_checkpoint(ds_name, results_list)
        
        logger.info(f"[{ds_name}] Finished successfully.")
        return results_list

    except Exception as e:
        logger.error(f"[{ds_name}] Process failed with fatal error: {e}")
        return []

def run_experiments():
    # List of datasets to process
    datasets = ['adult', 'data', 'cervical', 'compas', 'creditcard', 'heart', 'insurance', 'communities']
    
    logger.info(f"Starting experiments for {len(datasets)} datasets with Multiprocessing...")

    all_results = []
    
    # Use ProcessPoolExecutor to parallelize dataset processing
    # Using 'spawn' context is generally safer on Windows/macOS but default is 'spawn' on Windows anyway.
    # However, to avoid issues with pickling certain objects, we initialize them inside the worker.
    
    # Determined max_workers based on simple heuristic or system CPUs. 
    # Since these are heavy tasks (FHE), we might want to limit concurrent workers if memory is an issue.
    # We will let ProcessPoolExecutor decide default (os.cpu_count()).
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        future_to_ds = {executor.submit(process_dataset, ds): ds for ds in datasets}
        
        for future in concurrent.futures.as_completed(future_to_ds):
            ds = future_to_ds[future]
            try:
                data = future.result()
                if data:
                    all_results.extend(data)
            except Exception as exc:
                logger.error(f"{ds} generated an exception: {exc}")

    # Final Aggregation
    results_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'results', 'metrics', 'results_wide.csv'
    )
    
    if all_results:
        # Create directory if checks/saving failed previously (robustness)
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        
        pd.DataFrame(all_results).to_csv(results_path, index=False)
        logger.info(f"Global results aggregated and saved to {results_path}")
    else:
        logger.warning("No results to save (all datasets failed or empty).")
