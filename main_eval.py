# main_eval.py
from __future__ import annotations
import argparse
import os
import numpy as np
import logging
import datetime
import sys
from sklearn.metrics import accuracy_score, classification_report

from config import cfg
from data_loader import load_subject_data, split_train_test_by_blocks, filter_dataset
from model import FGSFTMIModel

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"eval_{timestamp}.log")
    
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

def main():
    parser = argparse.ArgumentParser(description="Evaluate FGSFT-MI Model")
    parser.add_argument("--subject", type=int, default=1, help="Subject ID (1-9)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to dataset")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save logs")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_dir)
    
    if args.data_dir:
        cfg.data_path = args.data_dir
        
    logging.info(f"Loading data for Subject {args.subject}...")
    try:
        dataset = load_subject_data(args.subject)
    except FileNotFoundError as e:
        logging.error(e)
        return

    # Filter classes
    logging.info(f"Filtering classes to {cfg.selected_labels}...")
    dataset = filter_dataset(dataset, cfg.selected_labels)

    # We need to know which part was test set. 
    # Ideally, we should save the split info or use a standard split.
    # For this demo, we assume the same split strategy as main_train: last block is test.
    blocks = np.unique(dataset.blocks)
    if len(blocks) > 1:
        test_block = blocks[-1]
        logging.info(f"Using Block {test_block} as test set.")
        _, ds_test = split_train_test_by_blocks(dataset, test_block)
    else:
        logging.info("Only 1 block found. Cannot replicate random split without seed info.")
        logging.info("Evaluating on ALL data.")
        ds_test = dataset

    logging.info("Loading model...")
    model = FGSFTMIModel()
    model.load(args.model_path)
    
    logging.info("Predicting...")
    y_pred = model.predict(ds_test)
    
    acc = accuracy_score(ds_test.y, y_pred)
    logging.info(f"Accuracy: {acc:.4f}")
    logging.info("\nClassification Report:")
    logging.info("\n" + classification_report(ds_test.y, y_pred))

if __name__ == "__main__":
    main()
