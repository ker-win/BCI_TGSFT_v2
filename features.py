# features.py
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np

from config import cfg
from data_loader import Dataset
from preprocessing import bandpass_filter
from segmentation import (
    ChannelGroup, FreqBand, TimeWindow, SFTSSpec,
)
from divcsp import DivCSP, DivCSPParams

def precompute_freq_bands(
    dataset: Dataset,
    freq_bands: List[FreqBand],
) -> Dict[int, np.ndarray]:
    """
    Pre-computes bandpass filtered data for all frequency bands.
    Returns dict: fb_id -> X_fband (n_trials, n_channels, n_samples)
    """
    X = dataset.X
    fs = dataset.fs
    X_fband = {}
    for fb in freq_bands:
        # Check if we can reuse previous computation? 
        # For now, just compute.
        X_fband[fb.id] = bandpass_filter(X, fs, fb.f_low, fb.f_high)
    return X_fband

def get_sfts_data(
    X_fband: Dict[int, np.ndarray],
    sfts: SFTSSpec,
    channel_groups: List[ChannelGroup],
    time_windows: List[TimeWindow],
) -> np.ndarray:
    """
    Extracts data for a specific SFTS from precomputed frequency data.
    
    Returns: X_sfts, shape (n_trials, n_sel_channels, n_samples_window)
    """
    fb_X = X_fband[sfts.freq_band_id]  # (n_trials, n_channels, n_samples)

    # Find channel group and time window objects
    # Optim: Pass dicts instead of lists if this is slow
    cg = next(c for c in channel_groups if c.id == sfts.ch_group_id)
    tw = next(t for t in time_windows if t.id == sfts.time_window_id)

    # Slice channels
    X_c = fb_X[:, cg.ch_idx, :]                # (n_trials, n_sel_channels, n_samples)
    
    # Slice time
    # Ensure indices are valid
    start = max(0, tw.start_idx)
    end = min(X_c.shape[-1], tw.end_idx)
    
    X_ct = X_c[..., start:end]   # (n_trials, n_sel_channels, n_samples_window)
    return X_ct

def extract_divcsp_features_for_sfts(
    X_fband: Dict[int, np.ndarray],
    sfts: SFTSSpec,
    channel_groups: List[ChannelGroup],
    time_windows: List[TimeWindow],
    trial_idx: np.ndarray,
    y: np.ndarray,
    divcsp_params: DivCSPParams = None,
) -> Tuple[np.ndarray, DivCSPParams]:
    """
    Extracts features for a single SFTS using DivCSP.
    
    Args:
        trial_idx: Indices of trials to use (e.g. training set)
        y: Labels for all trials (used for fitting if params not provided)
    
    Returns:
        feats: (n_selected_trials, n_components)
        divcsp_params: Trained parameters
    """
    X_sfts_all = get_sfts_data(X_fband, sfts, channel_groups, time_windows)
    X_sel = X_sfts_all[trial_idx]
    y_sel = y[trial_idx]

    divcsp = DivCSP()
    if divcsp_params is not None:
        divcsp.set_params(divcsp_params)
    else:
        divcsp.fit(X_sel, y_sel)
        divcsp_params = divcsp.get_params()

    feats = divcsp.transform(X_sel)
    return feats, divcsp_params
