from __future__ import annotations
import argparse
import os
import numpy as np
import logging
import datetime
import sys
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from config import cfg
from data_loader import load_subject_data, filter_dataset, make_splits, subset_dataset

from model import FGSFTMIModel

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"eval_all_{timestamp}.log")
    
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

def evaluate_subject(subject_id, model_dir, data_dir, use_test_data=False):
    model_path = os.path.join(model_dir, f"subject_{subject_id}_model.pkl")
    if not os.path.exists(model_path):
        logging.warning(f"Model for Subject {subject_id} not found at {model_path}. Skipping.")
        return None

    if data_dir:
        cfg.data_path = data_dir
        
    logging.info(f"Evaluating Subject {subject_id}...")
    
    file_suffix = 'E' if use_test_data else 'T'
    try:
        dataset, trials_meta = load_subject_data(subject_id, file_suffix=file_suffix)
    except FileNotFoundError as e:

        logging.error(f"Data for Subject {subject_id} not found: {e}")
        return None

    # Filter classes
    # We will filter AFTER splitting if we are splitting, to match main_train.py logic.
    # If using test data (E file), we filter immediately.
    if use_test_data:
        dataset = filter_dataset(dataset, cfg.selected_labels)


    if use_test_data:
        ds_test = dataset
    else:
        # Replicate split strategy: use make_splits with same random state as training
        # Assuming default config used in training (test_ratio=0.2, random_state=42)
        
        # We already have trials_meta from above
        
        # Filter metadata to match filtered dataset (binary classes)
        # However, filter_dataset only filters the Dataset object.
        # The indices from make_splits are based on the FULL dataset (trials_meta).
        # So we should split FIRST, then subset, then filter.
        
        splits = make_splits(trials_meta, mode="within-subject", test_ratio=0.2, random_state=42)
        test_idx = splits[0]['test']
        
        ds_test_unfiltered = subset_dataset(dataset, test_idx)
        ds_test = filter_dataset(ds_test_unfiltered, cfg.selected_labels)


    logging.info(f"Loading model for Subject {subject_id}...")
    model = FGSFTMIModel()
    try:
        model.load(model_path)
    except Exception as e:
        logging.error(f"Failed to load model for Subject {subject_id}: {e}")
        return None
    
    logging.info(f"Predicting for Subject {subject_id}...")
    y_pred = model.predict(ds_test)
    
    acc = accuracy_score(ds_test.y, y_pred)
    logging.info(f"Subject {subject_id} Accuracy: {acc:.4f}")
    
    return {
        "Subject": subject_id,
        "Accuracy": acc,
        "Num_Samples": len(ds_test.y)
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate FGSFT-MI Model for All Subjects")
    parser.add_argument("--models_dir", type=str, default="models", help="Directory containing trained models")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to dataset")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save logs")
    parser.add_argument("--use_test_data", action="store_true", help="Use evaluation dataset (E files) instead of splitting training data")
    
    args = parser.parse_args()
    
    setup_logging(args.log_dir)
    
    results = []
    
    for subject_id in range(1, 10):
        res = evaluate_subject(subject_id, args.models_dir, args.data_dir, args.use_test_data)
        if res:
            results.append(res)
            
    if results:
        df = pd.DataFrame(results)
        logging.info("\n" + "="*40)
        logging.info("Evaluation Summary")
        logging.info("="*40)
        logging.info("\n" + df.to_string(index=False))
        logging.info("-" * 40)
        logging.info(f"Average Accuracy: {df['Accuracy'].mean():.4f}")
        logging.info("="*40)
    else:
        logging.warning("No results obtained.")

if __name__ == "__main__":
    main()
