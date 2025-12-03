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
from data_loader import load_subject_data, make_splits, subset_dataset
from model import FGSFTMIModel

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"train_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Logging started. Saving to {log_file}")

def run_experiment_for_subject(subject_id, output_dir):
    logging.info(f"==========================================")
    logging.info(f"Starting training for Subject {subject_id}")
    logging.info(f"==========================================")
    
    # 1. Load Data
    logging.info(f"Loading data for Subject {subject_id}...")
    try:
        ds_T, meta_T = load_subject_data(subject_id, file_suffix='T')
    except FileNotFoundError as e:
        logging.error(f"Data for subject {subject_id} not found: {e}")
        return

    ds_train = None
    ds_test = None

    if cfg.exp_cfg.use_E_as_test:
        logging.info("Mode: Train on T file, Test on E file.")
        try:
            ds_E, meta_E = load_subject_data(subject_id, file_suffix='E')
            if len(ds_E.X) == 0:
                logging.warning(f"E file for subject {subject_id} is empty (no known labels). Falling back to splitting T file.")
                cfg.exp_cfg.use_E_as_test = False
            else:
                ds_train = ds_T
                ds_test = ds_E
        except FileNotFoundError:
            logging.warning(f"E file for subject {subject_id} not found. Falling back to splitting T file.")
            cfg.exp_cfg.use_E_as_test = False

    
    if not cfg.exp_cfg.use_E_as_test:
        logging.info(f"Mode: Within-subject split on T file (Test Ratio: {cfg.exp_cfg.test_ratio})")
        splits = make_splits(
            meta_T, 
            mode=cfg.exp_cfg.mode, 
            test_ratio=cfg.exp_cfg.test_ratio, 
            random_state=cfg.exp_cfg.random_state
        )
        # Assuming single fold for now as per make_splits implementation
        split = splits[0]
        train_idx = split['train']
        test_idx = split['test']
        
        logging.info(f"Split sizes: Train={len(train_idx)}, Test={len(test_idx)}")
        
        ds_train = subset_dataset(ds_T, train_idx)
        ds_test = subset_dataset(ds_T, test_idx)

    logging.info(f"Train Data: X={ds_train.X.shape}, y={ds_train.y.shape}")
    logging.info(f"Test Data: X={ds_test.X.shape}, y={ds_test.y.shape}")

    # Filter classes
    logging.info(f"Filtering classes to {cfg.selected_labels}...")
    from data_loader import filter_dataset
    ds_train = filter_dataset(ds_train, cfg.selected_labels)
    ds_test = filter_dataset(ds_test, cfg.selected_labels)
    logging.info(f"Filtered Train Data: X={ds_train.X.shape}, y={ds_train.y.shape}")
    logging.info(f"Filtered Test Data: X={ds_test.X.shape}, y={ds_test.y.shape}")

    # 2. Train Model

    # Note: Preprocessing is now handled inside model.fit() on the training data
    # and model.predict() will apply the same transformation.
    
    logging.info("Initializing model...")
    model = FGSFTMIModel()
    
    logging.info("Fitting model (this may take a while)...")
    if cfg.train_cfg.use_gpu:
        logging.info("GPU Acceleration Enabled.")
    
    model.fit(ds_train)
    
    # 3. Evaluate
    logging.info("Evaluating on test set...")
    y_pred = model.predict(ds_test)
    acc = accuracy_score(ds_test.y, y_pred)
    logging.info(f"Subject {subject_id} Test Accuracy: {acc:.4f}")
    
    # Save
    save_path = os.path.join(output_dir, f"subject_{subject_id}_model.pkl")
    model.save(save_path)
    logging.info(f"Model saved to {save_path}")
    logging.info(f"Finished Subject {subject_id}")

def main():
    parser = argparse.ArgumentParser(description="Train FGSFT-MI Model")
    parser.add_argument("--subject", type=int, default=1, help="Subject ID (1-9)")
    parser.add_argument("--all_subjects", action="store_true", help="Train all subjects (1-9) sequentially")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save logs")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of parallel jobs")
    
    # Experiment Config Args
    parser.add_argument("--mode", type=str, default="within-subject", help="Experiment mode")
    parser.add_argument("--use_E_as_test", action="store_true", help="Use E file as test set")
    parser.add_argument("--no_E_as_test", action="store_false", dest="use_E_as_test", help="Do not use E file as test set")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Test ratio if splitting T file")
    parser.add_argument("--use_gpu", action="store_true", help="Enable GPU acceleration")
    
    parser.set_defaults(use_E_as_test=True)

    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_dir)
    
    start_time = datetime.datetime.now()
    logging.info(f"Execution started at: {start_time}")

    # Update config
    if args.data_dir:
        cfg.data_path = args.data_dir
    cfg.train_cfg.n_jobs = args.n_jobs
    cfg.train_cfg.use_gpu = args.use_gpu
    
    cfg.exp_cfg.mode = args.mode
    cfg.exp_cfg.use_E_as_test = args.use_E_as_test
    cfg.exp_cfg.test_ratio = args.test_ratio
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.all_subjects:
        logging.info("Training ALL subjects (1-9)...")
        for sub in range(1, 10):
            try:
                run_experiment_for_subject(sub, args.output_dir)
            except Exception as e:
                logging.error(f"Failed to train subject {sub}: {e}", exc_info=True)
    else:
        run_experiment_for_subject(args.subject, args.output_dir)

    end_time = datetime.datetime.now()
    duration = end_time - start_time
    logging.info(f"Execution finished at: {end_time}")
    logging.info(f"Total execution time: {duration}")

if __name__ == "__main__":
    main()
