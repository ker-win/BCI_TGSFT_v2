from data_loader import load_subject_data
from config import cfg
from segmentation import generate_time_windows

def verify():
    print(f"Config trial_len_sec: {cfg.trial_len_sec}")
    
    # Load data for subject 1
    ds, meta = load_subject_data(1, file_suffix='T')
    
    print(f"Dataset X shape: {ds.X.shape}")
    print(f"Dataset fs: {ds.fs}")
    
    expected_samples = int(cfg.trial_len_sec * cfg.fs)
    print(f"Expected samples: {expected_samples}")
    
    if ds.X.shape[-1] == expected_samples:
        print("SUCCESS: Dataset X shape matches expected trial length.")
    else:
        print(f"FAILURE: Dataset X shape {ds.X.shape[-1]} does not match expected {expected_samples}.")

    # Check segmentation
    tws = generate_time_windows()
    print(f"Generated {len(tws)} time windows.")
    
    max_end_idx = max(t.end_idx for t in tws)
    print(f"Max time window end index: {max_end_idx}")
    
    if max_end_idx <= expected_samples:
        print("SUCCESS: Time windows fit within trial length.")
    else:
        print(f"FAILURE: Max time window end index {max_end_idx} exceeds trial length {expected_samples}.")

if __name__ == "__main__":
    verify()
