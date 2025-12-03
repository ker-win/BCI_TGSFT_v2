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
from data_loader import load_subject_data, split_train_test_by_blocks, filter_dataset
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

def train_subject(subject_id, output_dir):
    logging.info(f"==========================================")
    logging.info(f"Starting training for Subject {subject_id}")
    logging.info(f"==========================================")
    
    logging.info(f"Loading data for Subject {subject_id}...")
    try:
        dataset = load_subject_data(subject_id)
    except FileNotFoundError as e:
        logging.error(f"Data for subject {subject_id} not found: {e}")
        return

    logging.info(f"Data loaded: X={dataset.X.shape}, y={dataset.y.shape}, blocks={np.unique(dataset.blocks)}")
    
    # Filter classes
    logging.info(f"Filtering classes to {cfg.selected_labels}...")
    dataset = filter_dataset(dataset, cfg.selected_labels)
    logging.info(f"Filtered data: X={dataset.X.shape}, y={dataset.y.shape}")
    
    # Simple Train/Test split for demonstration
    blocks = np.unique(dataset.blocks)
    if len(blocks) > 1:
        test_block = blocks[-1]
        logging.info(f"Holding out Block {test_block} for testing.")
        ds_train, ds_test = split_train_test_by_blocks(dataset, test_block)
    else:
        logging.info("Only 1 block found. Using random 80/20 split.")
        # Random split
        n_trials = len(dataset.y)
        perm = np.random.permutation(n_trials)
        n_train = int(0.8 * n_trials)
        train_idx = perm[:n_train]
        test_idx = perm[n_train:]
        
        def subset(idx):
            return type(dataset)(
                X=dataset.X[idx],
                y=dataset.y[idx],
                blocks=dataset.blocks[idx],
                ch_names=dataset.ch_names,
                fs=dataset.fs
            )
        ds_train = subset(train_idx)
        ds_test = subset(test_idx)

    # Train
    logging.info("Initializing model...")
    model = FGSFTMIModel()
    
    logging.info("Fitting model (this may take a while)...")
    model.fit(ds_train)
    
    # Evaluate on held-out set
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
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_dir)
    
    start_time = datetime.datetime.now()
    logging.info(f"Execution started at: {start_time}")

    # Update config
    if args.data_dir:
        cfg.data_path = args.data_dir
    cfg.train_cfg.n_jobs = args.n_jobs
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.all_subjects:
        logging.info("Training ALL subjects (1-9)...")
        for sub in range(1, 10):
            try:
                train_subject(sub, args.output_dir)
            except Exception as e:
                logging.error(f"Failed to train subject {sub}: {e}", exc_info=True)
    else:
        train_subject(args.subject, args.output_dir)

    end_time = datetime.datetime.now()
    duration = end_time - start_time
    logging.info(f"Execution finished at: {end_time}")
    logging.info(f"Total execution time: {duration}")

if __name__ == "__main__":
    main()
