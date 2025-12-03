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
from divcsp_torch import DivCSPTorch
from scipy.signal import firwin
import torch
import torch.nn.functional as F

def bandpass_filter_torch(X_np: np.ndarray, fs: float, f_low: float, f_high: float, numtaps: int = 101, device: str = "cuda") -> np.ndarray:
    """
    Bandpass filter using PyTorch Conv1d.
    Args:
        X_np: (n_trials, n_channels, n_samples)
    """
    nyq = fs / 2.0
    # Design filter using scipy (FIR)
    taps = firwin(numtaps, [f_low / nyq, f_high / nyq], pass_zero=False)
    
    # Convert to tensor
    if not torch.cuda.is_available() and device == "cuda":
        device = "cpu"
        
    device_obj = torch.device(device)
    taps_tensor = torch.from_numpy(taps.astype(np.float32)).to(device_obj).view(1, 1, -1) # (out, in, kernel)
    
    X_tensor = torch.from_numpy(X_np.astype(np.float32)).to(device_obj) # (T, C, N)
    T, C, N = X_tensor.shape
    
    # Reshape for conv1d: (Batch, Channel, Time) -> We treat (T*C) as Batch, 1 Channel
    # Use reshape instead of view to handle non-contiguous tensors
    X_reshaped = X_tensor.reshape(T * C, 1, N)

    
    # Padding to keep size same (same padding)
    padding = numtaps // 2
    
    X_filtered = F.conv1d(X_reshaped, taps_tensor, padding=padding)
    
    # Reshape back
    X_out = X_filtered.view(T, C, -1)
    
    return X_out.detach().cpu().numpy()


def precompute_freq_bands(
    dataset: Dataset,
    freq_bands: List[FreqBand],
    use_gpu: bool = False,
    device: str = "cuda"
) -> Dict[int, np.ndarray]:
    """
    Pre-computes bandpass filtered data for all frequency bands.
    Returns dict: fb_id -> X_fband (n_trials, n_channels, n_samples)
    """
    X = dataset.X
    fs = dataset.fs
    X_fband = {}
    
    # Check if GPU is actually available
    if use_gpu and not torch.cuda.is_available():
        print("Warning: GPU requested but not available. Falling back to CPU.")
        use_gpu = False

    for fb in freq_bands:
        if use_gpu:
            X_fband[fb.id] = bandpass_filter_torch(X, fs, fb.f_low, fb.f_high, device=device)
        else:
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
    # Use DivCSPTorch if requested (need to pass config or param, but for now let's stick to DivCSP unless we change this function signature or global config)
    # Actually, we should allow using DivCSPTorch here if we want GPU acceleration in feature selection.
    # The user plan says "Update extract_divcsp_features_for_sfts to use DivCSPTorch when enabled".
    # We can check cfg.train_cfg.use_gpu if we add it, or pass it in.
    # For now, let's check a global flag or default to DivCSP (CPU) to avoid breaking changes unless we update call sites.
    # But wait, we want to use GPU.
    
    use_gpu = getattr(cfg.train_cfg, 'use_gpu', False)
    
    if use_gpu:
        divcsp = DivCSPTorch(device="cuda")
    else:
        divcsp = DivCSP()

    if divcsp_params is not None:
        divcsp.set_params(divcsp_params)
    else:
        divcsp.fit(X_sel, y_sel)
        divcsp_params = divcsp.get_params()


    feats = divcsp.transform(X_sel)
    return feats, divcsp_params
