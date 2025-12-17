# main_train.py
from __future__ import annotations
import argparse
import os
import numpy as np
import logging
import datetime
import sys
from sklearn.metrics import accuracy_score

from config import cfg
from data_loader import load_subject_data, filter_dataset, split_train_test_by_blocks
from model import FGSFTMIModel

def get_time_window_str():
    """Returns a string describing the current time window settings."""
    # Get from preprocessing defaults or cfg
    t_start = 0.5
    t_end = t_start + cfg.trial_len_sec
    return f"{t_start:.1f}s-{t_end:.1f}s"

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    time_window = get_time_window_str()
    log_file = os.path.join(output_dir, f"train_{time_window}_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Logging started. Saving to {log_file}")
    logging.info(f"Time Window: {time_window}")
    return log_file

def run_experiment_for_subject_lobo(subject_id, output_dir):
    """
    Run LOBO (Leave-One-Block-Out) cross-validation for a subject.
    Each block is used as test set once, providing a robust accuracy estimate.
    """
    logging.info(f"==========================================")
    logging.info(f"Starting LOBO Training for Subject {subject_id}")
    logging.info(f"==========================================")
    
    # 1. Load Data
    logging.info(f"Loading data for Subject {subject_id}...")
    try:
        ds_T, meta_T = load_subject_data(subject_id, file_suffix='T')
    except FileNotFoundError as e:
        logging.error(f"Data for subject {subject_id} not found: {e}")
        return None, None

    # Filter classes
    logging.info(f"Filtering classes to {cfg.selected_labels}...")
    ds_full = filter_dataset(ds_T, cfg.selected_labels)
    logging.info(f"Filtered Data: X={ds_full.X.shape}, y={ds_full.y.shape}")

    # Get unique blocks
    blocks = sorted(np.unique(ds_full.blocks).tolist())
    logging.info(f"Available blocks: {blocks} (Total: {len(blocks)})")

    # LOBO Cross-Validation
    fold_accs = []
    fold_samples = []
    
    for test_block in blocks:
        ds_train, ds_test = split_train_test_by_blocks(ds_full, test_block)
        
        if len(ds_train.y) == 0 or len(ds_test.y) == 0:
            logging.warning(f"  Skipping block {test_block}: empty train or test set")
            continue
            
        logging.info(f"  Fold {test_block}: Train={len(ds_train.y)}, Test={len(ds_test.y)}")
        
        # Train model
        model = FGSFTMIModel()
        model.fit(ds_train)
        
        # Evaluate
        y_pred = model.predict(ds_test)
        acc = accuracy_score(ds_test.y, y_pred)
        
        fold_accs.append(acc)
        fold_samples.append(len(ds_test.y))
        logging.info(f"    Block {test_block} Accuracy: {acc:.4f}")

    # Calculate mean accuracy
    if fold_accs:
        mean_acc = np.mean(fold_accs)
        total_samples = sum(fold_samples)
        logging.info(f"Subject {subject_id} LOBO Mean Accuracy: {mean_acc:.6f} (Total samples: {total_samples})")
    else:
        mean_acc = 0.0
        total_samples = 0
        logging.warning(f"Subject {subject_id}: No valid folds!")

    # Train final model on ALL data for deployment
    logging.info(f"Training final model on all data for Subject {subject_id}...")
    final_model = FGSFTMIModel()
    final_model.fit(ds_full)
    
    # Save final model
    save_path = os.path.join(output_dir, f"subject_{subject_id}_model.pkl")
    final_model.save(save_path)
    logging.info(f"Final model saved to {save_path}")
    logging.info(f"Finished Subject {subject_id}")
    
    return mean_acc, total_samples

def print_summary_table(results, time_window):
    """Print a formatted summary table of results."""
    logging.info("")
    logging.info("=" * 50)
    logging.info(f"TRAINING SUMMARY (Time Window: {time_window})")
    logging.info("=" * 50)
    
    header = f"{'Subject':>8}  {'Accuracy':>10}  {'Num_Samples':>12}"
    logging.info(header)
    
    for sub, acc, num_samples in results:
        if acc is not None:
            row = f"{sub:>8}  {acc:>10.6f}  {num_samples:>12}"
            logging.info(row)
    
    # Calculate and display mean accuracy
    valid_results = [(s, a, n) for s, a, n in results if a is not None]
    if valid_results:
        mean_acc = np.mean([r[1] for r in valid_results])
        total_samples = sum([r[2] for r in valid_results])
        logging.info("-" * 35)
        logging.info(f"{'Mean':>8}  {mean_acc:>10.6f}  {total_samples:>12}")
    
    logging.info("=" * 50)

def main():
    parser = argparse.ArgumentParser(description="Train FGSFT-MI Model with LOBO CV")
    parser.add_argument("--subject", type=int, default=1, help="Subject ID (1-9)")
    parser.add_argument("--all_subjects", action="store_true", help="Train all subjects (1-9) sequentially")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save logs")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of parallel jobs")
    parser.add_argument("--use_gpu", action="store_true", help="Enable GPU acceleration")
    
    args = parser.parse_args()
    
    # Setup logging with time window in filename
    log_file = setup_logging(args.log_dir)
    
    start_time = datetime.datetime.now()
    logging.info(f"Execution started at: {start_time}")
    logging.info(f"Using LOBO (Leave-One-Block-Out) Cross-Validation for evaluation")

    # Update config
    if args.data_dir:
        cfg.data_path = args.data_dir
    cfg.train_cfg.n_jobs = args.n_jobs
    cfg.train_cfg.use_gpu = args.use_gpu
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    time_window = get_time_window_str()
    results = []
    
    if args.all_subjects:
        logging.info("Training ALL subjects (1-9) with LOBO CV...")
        for sub in range(1, 10):
            try:
                acc, num_samples = run_experiment_for_subject_lobo(sub, args.output_dir)
                results.append((sub, acc, num_samples))
            except Exception as e:
                logging.error(f"Failed to train subject {sub}: {e}", exc_info=True)
                results.append((sub, None, 0))
        
        # Print summary table
        print_summary_table(results, time_window)
    else:
        acc, num_samples = run_experiment_for_subject_lobo(args.subject, args.output_dir)
        results.append((args.subject, acc, num_samples))
        print_summary_table(results, time_window)

    end_time = datetime.datetime.now()
    duration = end_time - start_time
    logging.info(f"Execution finished at: {end_time}")
    logging.info(f"Total execution time: {duration}")
    logging.info(f"Log saved to: {log_file}")

if __name__ == "__main__":
    main()
