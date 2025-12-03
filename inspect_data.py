import numpy as np
import os
from config import cfg

def inspect(subject_id=1):
    file_path = os.path.join(cfg.data_path, f'A0{subject_id}E.npz')
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    data = np.load(file_path)
    events_type = data['etyp'].T.ravel()
    unique_events = np.unique(events_type)
    print(f"Unique events in {file_path}: {unique_events}")
    
    # Check for MI events
    mi_events = [769, 770, 771, 772]
    found_mi = [e for e in mi_events if e in unique_events]
    print(f"Found MI events: {found_mi}")
    
    if 783 in unique_events:
        print("Found event 783 (Unknown/Cue unknown).")

if __name__ == "__main__":
    inspect()
