# preprocessing.py
from __future__ import annotations
import numpy as np
from scipy.signal import butter, filtfilt, resample
from config import cfg
from data_loader import Dataset

def bandpass_filter(
    X: np.ndarray,
    fs: float,
    f_low: float,
    f_high: float,
    order: int = 5,
) -> np.ndarray:
    """
    Zero-phase bandpass filter (Butterworth + filtfilt).
    
    Args:
        X: (n_trials, n_channels, n_samples)
        fs: Sampling frequency
        f_low: Low cutoff frequency
        f_high: High cutoff frequency
        order: Filter order
        
    Returns:
        Filtered data with same shape as X.
    """
    nyq = 0.5 * fs
    b, a = butter(order, [f_low / nyq, f_high / nyq], btype="band")
    Xf = filtfilt(b, a, X, axis=-1)
    return Xf.astype(np.float32)

def resample_to_fs(dataset: Dataset, fs_new: float) -> Dataset:
    """
    Resamples the dataset to a new sampling frequency.
    """
    if abs(dataset.fs - fs_new) < 1e-5:
        return dataset

    n_trials, n_ch, n_samples = dataset.X.shape
    n_samples_new = int(n_samples * fs_new / dataset.fs)
    
    # resample along the last axis (time)
    X_rs = resample(dataset.X, n_samples_new, axis=-1)
    
    return Dataset(
        X=X_rs.astype(np.float32),
        y=dataset.y,
        blocks=dataset.blocks,
        ch_names=dataset.ch_names,
        fs=fs_new,
    )

def crop_trials(dataset: Dataset, t_start: float, t_end: float) -> Dataset:
    """
    Crops trials to a specific time window.
    
    Args:
        dataset: Input dataset
        t_start: Start time in seconds (relative to current trial start)
        t_end: End time in seconds
    """
    fs = dataset.fs
    start_idx = int(round(t_start * fs))
    end_idx = int(round(t_end * fs))
    
    # Ensure indices are within bounds
    n_samples = dataset.X.shape[-1]
    if start_idx < 0: start_idx = 0
    if end_idx > n_samples: end_idx = n_samples
    
    X_crop = dataset.X[..., start_idx:end_idx]
    
    return Dataset(
        X=X_crop,
        y=dataset.y,
        blocks=dataset.blocks,
        ch_names=dataset.ch_names,
        fs=dataset.fs,
    )

def preprocess_pipeline(dataset: Dataset) -> Dataset:
    """
    Main preprocessing pipeline:
    1. Resample to cfg.fs (250Hz)
    2. Crop to desired trial length (e.g. 0 to 1.5s relative to cue? Or 2-3.5s?)
       
       Note: data_loader extracts 0-4s relative to cue.
       Paper says: "time window of 2-5s after the cue".
       Wait, if we extracted 0-4s, we have [0, 4]s.
       If we want 1.5s length, maybe we take [2.0, 3.5]?
       Or [0.5, 2.0]?
       
       Let's assume we want to capture the MI activity.
       MI usually starts shortly after cue.
       Let's crop from 0.5s to 0.5 + trial_len_sec.
       
       Actually, let's make it configurable or fixed to a reasonable window.
       If we assume the user wants to reproduce the paper, we should look at what they did.
       Paper: "time segment of length 3s (2-5s)".
       But our config has `trial_len_sec = 1.5`.
       Maybe we should extract a 3s window first, then the segmentation logic handles the sliding windows?
       
       The segmentation logic in `segmentation.py` takes `cfg.trial_len_sec` as the total length to slide over.
       So if `trial_len_sec` is 1.5s, we should crop a 1.5s chunk where MI is most prominent.
       Usually 0.5s to 2.0s or 1.0s to 2.5s post-cue.
       Let's pick 0.5s to 0.5 + cfg.trial_len_sec.
    """
    ds = resample_to_fs(dataset, cfg.fs)
    
    # Crop to the main analysis window
    # We'll start at 0.5s post-cue to avoid cue-related VEPs
    t_start = 0.5
    t_end = t_start + cfg.trial_len_sec
    
    ds = crop_trials(ds, t_start=t_start, t_end=t_end)
    
    return ds
