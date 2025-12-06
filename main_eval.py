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

def evaluate_subject(subject_id, model_path, args):
    logging.info(f"==========================================")
    logging.info(f"Starting evaluation for Subject {subject_id}")
    logging.info(f"Model Path: {model_path}")
    logging.info(f"==========================================")

    if not os.path.exists(model_path):
        logging.error(f"Model file not found: {model_path}")
        return None, None

    logging.info(f"Loading data for Subject {subject_id}...")
    
    file_suffix = 'E' if args.use_test_data else 'T'
    try:
        dataset, _ = load_subject_data(subject_id, file_suffix=file_suffix)
    except FileNotFoundError as e:
        logging.error(e)
        return None, None

    # Filter classes
    logging.info(f"Filtering classes to {cfg.selected_labels}...")
    dataset = filter_dataset(dataset, cfg.selected_labels)

    if args.use_test_data:
        logging.info("Using Evaluation dataset (E file) as test set.")
        ds_test = dataset
    else:
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
    model.load(model_path)
    
    logging.info("Predicting...")
    y_pred = model.predict(ds_test)
    
    acc = accuracy_score(ds_test.y, y_pred)
    num_samples = len(ds_test.y)
    logging.info(f"Subject {subject_id} Test Accuracy: {acc:.4f}")
    logging.info("\nClassification Report:")
    logging.info("\n" + classification_report(ds_test.y, y_pred))
    
    return acc, num_samples


def main():
    parser = argparse.ArgumentParser(description="Evaluate FGSFT-MI Model")
    parser.add_argument("--subject", type=int, default=1, help="Subject ID (1-9)")
    parser.add_argument("--all_subjects", action="store_true", help="Evaluate all subjects (1-9)")
    parser.add_argument("--model_path", type=str, default=None, help="Path to trained model (for single subject)")
    parser.add_argument("--model_dir", type=str, default="models", help="Directory containing models (for --all_subjects)")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to dataset")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save logs")
    parser.add_argument("--use_test_data", action="store_true", help="Use evaluation dataset (E files) instead of splitting training data")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_dir)
    
    if args.data_dir:
        cfg.data_path = args.data_dir
        
    if args.all_subjects:
        logging.info("Evaluating ALL subjects (1-9)...")
        results = []
        for sub in range(1, 10):
            model_path = os.path.join(args.model_dir, f"subject_{sub}_model.pkl")
            try:
                acc, num_samples = evaluate_subject(sub, model_path, args)
                if acc is not None:
                    results.append((sub, acc, num_samples))
            except Exception as e:
                logging.error(f"Failed to evaluate subject {sub}: {e}", exc_info=True)
        
        # Print summary table
        if results:
            logging.info("\n" + "=" * 50)
            logging.info("EVALUATION SUMMARY")
            logging.info("=" * 50)
            header = f"{'Subject':>8}  {'Accuracy':>8}  {'Num_Samples':>11}"
            logging.info(header)
            for sub, acc, num_samples in results:
                row = f"{sub:>8}  {acc:>8.6f}  {num_samples:>11}"
                logging.info(row)
            
            # Calculate and display mean accuracy
            mean_acc = np.mean([r[1] for r in results])
            logging.info("-" * 35)
            logging.info(f"{'Mean':>8}  {mean_acc:>8.6f}")
            logging.info("=" * 50)
    else:
        if not args.model_path:
            # Try to infer model path if not provided
            args.model_path = os.path.join(args.model_dir, f"subject_{args.subject}_model.pkl")
            logging.info(f"Model path not provided. Inferring from model_dir: {args.model_path}")
            
        evaluate_subject(args.subject, args.model_path, args)

if __name__ == "__main__":
    main()
