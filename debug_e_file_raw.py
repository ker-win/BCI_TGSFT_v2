# debug_e_file_raw.py
"""
Debug script to check raw E file structure.
"""
import numpy as np
import os

data_path = "C:\\Users\\EEG_Dataset\\bcidatasetIV2a-master"

for sub in [1, 7, 8, 9]:
    file_path = os.path.join(data_path, f'A0{sub}E.npz')
    
    print(f"\n{'='*60}")
    print(f"Subject {sub} E file: {file_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(file_path):
        print(f"  File not found!")
        continue
        
    data = np.load(file_path)
    
    print(f"Keys in file: {list(data.keys())}")
    
    # Check event types
    etyp = data['etyp'].T.ravel()
    unique_types = np.unique(etyp)
    print(f"\nUnique event types: {unique_types}")
    
    # Count occurrences
    for et in [768, 769, 770, 771, 772, 32766, 783]:
        count = np.sum(etyp == et)
        if count > 0:
            print(f"  Event {et}: {count} occurrences")
    
    # Check if there's a separate labels file
    labels_file = os.path.join(data_path, f'A0{sub}E_labels.npz')
    if os.path.exists(labels_file):
        print(f"\nLabels file found: {labels_file}")
    else:
        labels_file_mat = os.path.join(data_path, f'true_labels', f'A0{sub}E.mat')
        if os.path.exists(labels_file_mat):
            print(f"\nLabels MAT file found: {labels_file_mat}")
        else:
            print(f"\nNo separate labels file found.")
            
    # Also check for other potential label file locations
    for pattern in ['labels', 'true_labels', 'Labels', 'True_labels']:
        label_dir = os.path.join(data_path, pattern)
        if os.path.exists(label_dir):
            print(f"Found label directory: {label_dir}")
            files = os.listdir(label_dir)
            print(f"  Files: {files[:5]}...")
