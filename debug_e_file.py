# debug_e_file.py
"""
Debug script to check E file data for failed subjects.
"""
import numpy as np
from data_loader import load_subject_data
from preprocessing import Preprocessor

def check_e_file(subject_id):
    print(f"\n{'='*60}")
    print(f"Checking Subject {subject_id} E file")
    print(f"{'='*60}")
    
    try:
        dataset, _ = load_subject_data(subject_id, file_suffix='E')
        print(f"1. Raw loaded data:")
        print(f"   - X shape: {dataset.X.shape}")
        print(f"   - fs: {dataset.fs}")
        print(f"   - n_trials: {len(dataset.y)}")
        print(f"   - samples: {dataset.X.shape[-1]}")
        print(f"   - duration: {dataset.X.shape[-1] / dataset.fs:.2f}s")
        
        # Filter to classes 0, 1
        from data_loader import filter_dataset
        ds_filtered = filter_dataset(dataset, [0, 1])
        print(f"\n2. After filtering to classes [0, 1]:")
        print(f"   - X shape: {ds_filtered.X.shape}")
        print(f"   - n_trials: {len(ds_filtered.y)}")
        
        # Apply preprocessor with saved model settings
        print(f"\n3. Applying preprocessor (t_start=0.5, t_end=1.5):")
        preprocessor = Preprocessor(fs_target=250.0, do_scaling=False, t_start=0.5, t_end=1.5)
        ds_proc = preprocessor.transform(ds_filtered)
        print(f"   - X shape after transform: {ds_proc.X.shape}")
        print(f"   - samples: {ds_proc.X.shape[-1]}")
        
        if ds_proc.X.shape[-1] < 34:
            print(f"   WARNING: Samples ({ds_proc.X.shape[-1]}) < 34 (filtfilt padlen requirement)!")
        else:
            print(f"   OK: Samples >= 34")
            
        return True
        
    except FileNotFoundError as e:
        print(f"   ERROR: {e}")
        return False
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Check the failing subjects
    for sub in [1, 7, 8, 9]:
        check_e_file(sub)
