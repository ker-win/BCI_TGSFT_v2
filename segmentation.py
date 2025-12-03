# segmentation.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np
from config import cfg

@dataclass
class TimeWindow:
    id: int
    start_idx: int
    end_idx: int
    length_sec: float

@dataclass
class FreqBand:
    id: int
    f_low: float
    f_high: float
    bandwidth: float

@dataclass
class ChannelGroup:
    id: int
    name: str
    ch_idx: List[int]   # channel indices

@dataclass
class SFTSSpec:
    id: int
    ch_group_id: int
    freq_band_id: int
    time_window_id: int

def generate_time_windows() -> List[TimeWindow]:
    fs = cfg.fs
    total_len_samples = int(cfg.trial_len_sec * fs)
    windows: List[TimeWindow] = []
    wid = 0

    for L_sec in cfg.time_cfg.lengths:
        L = int(round(L_sec * fs))
        step = int(round(L * (1.0 - cfg.time_cfg.overlap)))
        if step == 0:
            step = 1
            
        start = 0
        while True:
            end = start + L
            if end > total_len_samples:
                break
            windows.append(TimeWindow(id=wid, start_idx=start, end_idx=end, length_sec=L_sec))
            wid += 1
            start += step

    return windows

def generate_freq_bands() -> List[FreqBand]:
    bands: List[FreqBand] = []
    bid = 0
    for bw in cfg.freq_cfg.bandwidths:
        f = cfg.freq_cfg.min_freq
        step = bw * (1.0 - cfg.freq_cfg.overlap)
        if step <= 0:
            step = bw
            
        while True:
            f_low = f
            f_high = f_low + bw
            if f_high > cfg.freq_cfg.max_freq + 1e-6:
                break
            bands.append(FreqBand(id=bid, f_low=f_low, f_high=f_high, bandwidth=bw))
            bid += 1
            f += step
    return bands

def generate_channel_groups(ch_names: list) -> List[ChannelGroup]:
    """
    Generates channel groups.
    For BCI IV 2a (22 channels), we define some manual groups based on the paper or standard sensorimotor areas.
    """
    groups: List[ChannelGroup] = []
    
    # Helper to find indices
    def get_idx(names):
        return [ch_names.index(n) for n in names if n in ch_names]

    # 1. All channels
    groups.append(ChannelGroup(id=0, name="All", ch_idx=list(range(len(ch_names)))))
    
    # 2. C3 centered (Left Motor)
    c3_group = ['FC3', 'C5', 'C3', 'C1', 'CP3']
    groups.append(ChannelGroup(id=1, name="Left_Motor", ch_idx=get_idx(c3_group)))
    
    # 3. C4 centered (Right Motor)
    c4_group = ['FC4', 'C2', 'C4', 'C6', 'CP4']
    groups.append(ChannelGroup(id=2, name="Right_Motor", ch_idx=get_idx(c4_group)))
    
    # 4. Cz centered (Central)
    cz_group = ['FCz', 'C1', 'Cz', 'C2', 'CPz']
    groups.append(ChannelGroup(id=3, name="Central", ch_idx=get_idx(cz_group)))
    
    # 5. Frontal-Central
    fc_group = ['Fz', 'FC1', 'FCz', 'FC2']
    groups.append(ChannelGroup(id=4, name="Frontal_Central", ch_idx=get_idx(fc_group)))
    
    # 6. Central-Parietal
    cp_group = ['CP1', 'CPz', 'CP2', 'P1', 'Pz', 'P2']
    groups.append(ChannelGroup(id=5, name="Central_Parietal", ch_idx=get_idx(cp_group)))
    
    # Add more groups if needed to match the 10 groups mentioned in paper
    # ...
    
    # Re-index ids just in case
    for i, g in enumerate(groups):
        g.id = i
        
    return groups

def generate_sfts_specs(
    channel_groups: List[ChannelGroup],
    freq_bands: List[FreqBand],
    time_windows: List[TimeWindow],
) -> List[SFTSSpec]:
    specs: List[SFTSSpec] = []
    sid = 0
    for cg in channel_groups:
        for fb in freq_bands:
            for tw in time_windows:
                specs.append(SFTSSpec(
                    id=sid,
                    ch_group_id=cg.id,
                    freq_band_id=fb.id,
                    time_window_id=tw.id,
                ))
                sid += 1
    return specs
