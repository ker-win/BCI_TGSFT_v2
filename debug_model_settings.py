# debug_model_settings.py
"""
Debug script to check saved model's time window settings.
"""
import pickle
import os

def check_model_settings(model_path):
    print(f"\n{'='*60}")
    print(f"Checking model: {model_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(model_path):
        print(f"  ERROR: Model file not found!")
        return
    
    with open(model_path, 'rb') as f:
        state = pickle.load(f)
    
    # Check preprocessor settings
    print("\n1. Preprocessor Settings:")
    print(f"   - t_start: {state.preprocessor.t_start}")
    print(f"   - t_end: {state.preprocessor.t_end}")
    print(f"   - fs_target: {state.preprocessor.fs_target}")
    
    expected_samples = int((state.preprocessor.t_end - state.preprocessor.t_start) * state.preprocessor.fs_target)
    print(f"   - Expected samples after crop: {expected_samples}")
    
    # Check time windows
    print("\n2. Time Windows (first 5):")
    for tw in state.time_windows[:5]:
        print(f"   - ID={tw.id}: start={tw.start_idx}, end={tw.end_idx}, len={tw.length_sec}s")
    print(f"   ... Total: {len(state.time_windows)} time windows")
    
    # Check max end index
    max_end = max(tw.end_idx for tw in state.time_windows)
    print(f"\n3. Max time window end_idx: {max_end}")
    
    if max_end > expected_samples:
        print(f"   WARNING: Max end_idx ({max_end}) > expected samples ({expected_samples})!")
        print(f"   This will cause slicing issues!")
    else:
        print(f"   OK: Max end_idx fits within expected samples.")
    
    # Check freq bands
    print(f"\n4. Frequency Bands: {len(state.freq_bands)} bands")
    
    # Check ensemble
    print(f"\n5. Ensemble Members: {len(state.ensemble)}")
    
if __name__ == "__main__":
    model_dir = "models"
    for sub in range(1, 10):
        model_path = os.path.join(model_dir, f"subject_{sub}_model.pkl")
        check_model_settings(model_path)
