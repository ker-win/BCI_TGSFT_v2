# data_loader.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List
import numpy as np
import os
from config import cfg

@dataclass
class Dataset:
    X: np.ndarray          # (n_trials, n_channels, n_samples)
    y: np.ndarray          # (n_trials,)
    blocks: np.ndarray     # (n_trials,) block index
    ch_names: list         # channel names
    fs: float              # sampling rate

def load_subject_data(subject_id: int, root_dir: str = None) -> Dataset:
    """
    Loads raw data for a single subject from BCI Competition IV 2a dataset.
    
    Args:
        subject_id: Subject ID (1-9)
        root_dir: Path to the dataset directory. If None, uses cfg.data_path.
        
    Returns:
        Dataset object containing X, y, blocks, ch_names, fs.
    """
    if root_dir is None:
        root_dir = cfg.data_path
        
    file_path = os.path.join(root_dir, f'A0{subject_id}T.npz')
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
        
    data = np.load(file_path)
    
    # Extract data
    # raw: (n_channels, n_samples) -> transpose to match (n_samples, n_channels) if needed, 
    # but usually we want (n_channels, n_samples) for MNE/processing
    # load_data.py does: self.raw = self.data['s'].T -> (22, n_samples)
    raw_eeg = data['s'].T
    events_type = data['etyp'].T.ravel()
    events_position = data['epos'].T.ravel()
    events_duration = data['edur'].T.ravel()
    
    # 22 EEG channels (usually first 22)
    # BCI IV 2a has 22 EEG + 3 EOG. We usually take first 22.
    # load_data.py takes channel 7 by default or list of channels.
    # We will take all 22 EEG channels.
    n_eeg_channels = 22
    raw_eeg = raw_eeg[:n_eeg_channels, :]
    
    # Channel names for BCI IV 2a (standard 10-20 system)
    ch_names = [
        'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 
        'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 
        'P2', 'POz'
    ]
    
    # Define events
    # 769: left, 770: right, 771: foot, 772: tongue
    mi_types = {769: 0, 770: 1, 771: 2, 772: 3}
    
    # Trial extraction
    # Start of trial code: 768
    start_trial_code = 768
    start_trial_indices = np.where(events_type == start_trial_code)[0]
    
    trials = []
    labels = []
    block_ids = []
    
    # Block/Run detection
    # Event 32766 indicates start of a new run
    run_start_indices = np.where(events_type == 32766)[0]
    run_start_positions = events_position[run_start_indices]
    
    # If no run events, assume 1 block
    if len(run_start_positions) == 0:
        run_start_positions = [0]
    
    # Parameters for extraction
    # Paper uses 2-5s after cue? Or 0.5-3.5s?
    # load_data.py uses tmin=0.5, tmax=3.5 (3 seconds)
    # We will extract a generous window and let preprocessing crop it later.
    # Let's extract 0 to 4s relative to cue to be safe.
    # Note: 768 is start of trial. Cue is usually shortly after?
    # In BCI IV 2a:
    # t=0: fixation cross
    # t=2s: cue (769-772)
    # t=3.25s: start of motor imagery
    # t=6s: end of motor imagery
    # Wait, load_data.py logic:
    # idxs = [i for i, x in enumerate(starttrial_events[0]) if x]
    # type_e = self.events_type[0, index+1] (Cue is next event)
    # start = self.events_position[0, index]
    # stop = start + self.events_duration[0, index]
    # This seems to extract the whole trial duration defined in 'edur'.
    
    # Let's follow a standard approach:
    # Find cue events (769-772) directly.
    # Extract [0, 4] seconds relative to cue.
    
    fs = 250.0
    
    for i, etype in enumerate(events_type):
        if etype in mi_types:
            label = mi_types[etype]
            pos = events_position[i]
            
            # Determine block ID
            # Find the last run start position that is <= current pos
            current_block = 0
            for b_idx, run_pos in enumerate(run_start_positions):
                if pos >= run_pos:
                    current_block = b_idx
            
            # Extract epoch
            # We want enough data to crop later.
            # Let's take 0s to 4s relative to cue.
            # (Paper says 2-5s after stimulus? We need to be careful with timing)
            # If we extract 0-4s, we cover the MI period.
            t_start_sample = int(pos)
            t_end_sample = int(pos + 4.0 * fs)
            
            if t_end_sample <= raw_eeg.shape[1]:
                trial_data = raw_eeg[:, t_start_sample:t_end_sample]
                trials.append(trial_data)
                labels.append(label)
                block_ids.append(current_block)
                
    X = np.array(trials)  # (n_trials, n_channels, n_samples)
    y = np.array(labels)
    blocks = np.array(block_ids)
    
    return Dataset(X=X, y=y, blocks=blocks, ch_names=ch_names, fs=fs)

def split_train_test_by_blocks(dataset: Dataset, test_block: int) -> Tuple[Dataset, Dataset]:
    """
    Splits the dataset into train and test sets based on block ID (LOBO).
    """
    mask_test = dataset.blocks == test_block
    mask_train = ~mask_test

    def subset(mask):
        return Dataset(
            X=dataset.X[mask],
            y=dataset.y[mask],
            blocks=dataset.blocks[mask],
            ch_names=dataset.ch_names,
            fs=dataset.fs,
        )

    return subset(mask_train), subset(mask_test)

def filter_dataset(dataset: Dataset, labels: List[int]) -> Dataset:
    """
    Filters the dataset to keep only the specified labels.
    """
    mask = np.isin(dataset.y, labels)
    return Dataset(
        X=dataset.X[mask],
        y=dataset.y[mask],
        blocks=dataset.blocks[mask],
        ch_names=dataset.ch_names,
        fs=dataset.fs,
    )
