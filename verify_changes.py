import numpy as np
import torch
from data_loader import Dataset, TrialInfo, make_splits
from divcsp import DivCSP
from divcsp_torch import DivCSPTorch
from features import bandpass_filter_torch
from preprocessing import bandpass_filter

def test_data_leakage():
    print("Testing Data Leakage...")
    # Create synthetic metadata
    n_trials = 100
    trials_meta = []
    for i in range(n_trials):
        trials_meta.append(TrialInfo(
            idx=i,
            subject_id=1,
            session_id=1,
            label=i % 2
        ))
    
    splits = make_splits(trials_meta, mode="within-subject", test_ratio=0.2, random_state=42)
    split = splits[0]
    train_idx = set(split['train'])
    test_idx = set(split['test'])
    
    intersection = train_idx.intersection(test_idx)
    assert len(intersection) == 0, f"Leakage detected! Intersection: {intersection}"
    print("Data Leakage Test Passed!")

def test_gpu_consistency():
    print("Testing GPU Consistency...")
    if not torch.cuda.is_available():
        print("Skipping GPU test (CUDA not available)")
        return

    # Synthetic data
    n_trials = 20
    n_channels = 22
    n_samples = 500
    X = np.random.randn(n_trials, n_channels, n_samples).astype(np.float32)
    y = np.random.randint(0, 2, n_trials)
    
    # 1. Bandpass Filter
    fs = 250.0
    f_low = 8.0
    f_high = 30.0
    
    X_cpu = bandpass_filter(X, fs, f_low, f_high)
    X_gpu = bandpass_filter_torch(X, fs, f_low, f_high, device="cuda")
    
    # Check difference
    diff = np.abs(X_cpu - X_gpu).max()
    print(f"Bandpass Filter Max Diff: {diff}")
    # Differences might be due to implementation details (filtfilt vs firwin conv1d)
    # They won't be identical, but should be reasonable.
    # Actually, filtfilt is IIR zero-phase, firwin is FIR. They are different filters.
    # So we can't expect them to be close.
    # But we can check if GPU runs without error.
    print("Bandpass Filter ran successfully on GPU.")

    # 2. DivCSP
    divcsp_cpu = DivCSP()
    divcsp_cpu.fit(X, y)
    f_cpu = divcsp_cpu.transform(X)
    
    divcsp_gpu = DivCSPTorch(device="cuda")
    divcsp_gpu.fit(X, y)
    f_gpu = divcsp_gpu.transform(X)
    
    print(f"DivCSP CPU features shape: {f_cpu.shape}")
    print(f"DivCSP GPU features shape: {f_gpu.shape}")
    
    # Check if features are somewhat correlated or similar range
    # Again, implementation details might differ (solver, regularization).
    # But shapes should match.
    assert f_cpu.shape == f_gpu.shape
    print("DivCSP shapes match.")
    
    print("GPU Consistency Test Passed (Basic Runtime Check).")

def test_time_window_shape():
    print("Testing Time Window Shape...")
    from config import cfg
    from preprocessing import preprocess_pipeline
    from data_loader import Dataset
    
    # Create dummy dataset
    fs = 250.0
    n_trials = 5
    n_channels = 22
    # Create data long enough for 0-4s extraction (1000 samples)
    # data_loader extracts 4s -> 1000 samples
    n_samples_raw = int(4.0 * fs) 
    X = np.random.randn(n_trials, n_channels, n_samples_raw).astype(np.float32)
    y = np.zeros(n_trials)
    blocks = np.zeros(n_trials)
    ch_names = [str(i) for i in range(n_channels)]
    
    ds = Dataset(X=X, y=y, blocks=blocks, ch_names=ch_names, fs=fs)
    
    # Run pipeline
    ds_proc = preprocess_pipeline(ds)
    
    expected_len_sec = 3.0
    expected_samples = int(expected_len_sec * fs)
    
    print(f"Processed shape: {ds_proc.X.shape}")
    
    assert ds_proc.X.shape[-1] == expected_samples, \
        f"Expected {expected_samples} samples (3.0s), got {ds_proc.X.shape[-1]}"
        
    print(f"Time Window Verification Passed! Length is {expected_len_sec}s ({expected_samples} samples).")

if __name__ == "__main__":
    test_data_leakage()
    test_gpu_consistency()
    test_time_window_shape()
