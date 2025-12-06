# verify_time_window.py
"""
Verification script to confirm time window and data shapes after modifications.
"""
import numpy as np
from config import cfg
from data_loader import load_subject_data, filter_dataset
from preprocessing import Preprocessor, crop_trials

def main():
    print("=" * 60)
    print("Time Window and Data Shape Verification")
    print("=" * 60)
    
    # 1. Configuration check
    print("\n1. Configuration Check:")
    print(f"   - cfg.fs: {cfg.fs} Hz")
    print(f"   - cfg.trial_len_sec: {cfg.trial_len_sec} seconds")
    print(f"   - Expected samples: {int(cfg.trial_len_sec * cfg.fs)} samples")
    
    # 2. Load data for subject 1
    print("\n2. Loading Subject 1 data...")
    try:
        ds_T, meta_T = load_subject_data(1, file_suffix='T')
        print(f"   - Loaded X shape: {ds_T.X.shape}")
        print(f"   - Sampling rate: {ds_T.fs} Hz")
        print(f"   - Trials: {len(ds_T.y)}")
        print(f"   - Raw trial duration: {ds_T.X.shape[-1] / ds_T.fs:.2f}s ({ds_T.X.shape[-1]} samples)")
    except FileNotFoundError as e:
        print(f"   - Error: {e}")
        print("   - Please ensure data path is correct in config.py")
        return
    
    # 3. Apply preprocessing (without scaling to match training)
    print("\n3. Applying Preprocessing (no scaling):")
    preprocessor = Preprocessor(fs_target=cfg.fs, do_scaling=False)
    ds_proc = preprocessor.transform(ds_T)
    
    print(f"   - Preprocessed X shape: {ds_proc.X.shape}")
    print(f"   - Samples after preprocessing: {ds_proc.X.shape[-1]}")
    print(f"   - Trial duration: {ds_proc.X.shape[-1] / ds_proc.fs:.2f}s")
    
    # 4. Verify time window
    print("\n4. Time Window Verification:")
    expected_start = 0.5  # seconds
    expected_end = 2.5    # seconds
    expected_duration = expected_end - expected_start
    expected_samples = int(expected_duration * cfg.fs)
    
    print(f"   - Expected window: {expected_start}s to {expected_end}s")
    print(f"   - Expected duration: {expected_duration}s")
    print(f"   - Expected samples: {expected_samples}")
    print(f"   - Actual samples: {ds_proc.X.shape[-1]}")
    
    if ds_proc.X.shape[-1] == expected_samples:
        print("   ✓ PASS: Time window samples match expected!")
    else:
        print(f"   ✗ FAIL: Expected {expected_samples} samples, got {ds_proc.X.shape[-1]}")
    
    # 5. Filter to binary classes
    print("\n5. Binary Classification Check:")
    ds_filtered = filter_dataset(ds_proc, cfg.selected_labels)
    print(f"   - Classes used: {cfg.selected_labels}")
    print(f"   - Filtered X shape: {ds_filtered.X.shape}")
    print(f"   - Class distribution: {dict(zip(*np.unique(ds_filtered.y, return_counts=True)))}")
    
    print("\n" + "=" * 60)
    print("Verification Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
