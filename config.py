# config.py
from dataclasses import dataclass
from typing import List

@dataclass
class TimeWindowConfig:
    lengths: List[float]      # seconds, e.g. [0.2, 0.4, 0.7, 1.0, 1.25, 1.5]
    overlap: float            # e.g. 0.5

@dataclass
class FreqBandConfig:
    min_freq: float           # e.g. 4.0
    max_freq: float           # e.g. 40.0
    bandwidths: List[float]   # e.g. [4, 8, 16, 32]
    overlap: float            # e.g. 0.5

@dataclass
class DivCSPConfig:
    n_components: int = 4
    lambda_reg: float = 0.1
    max_iter: int = 100
    tol: float = 1e-6

@dataclass
class FeatureSelectionConfig:
    D: int = 5    # Number of SFTS added per step
    K: int = 3    # Number of top ensemble members to keep

@dataclass
class TrainingConfig:
    random_state: int = 42
    n_jobs: int = -1          # -1 = use all cores
    verbose: int = 1

@dataclass
class SVMConfig:
    max_iter: int = 10000
    dual: str = "auto"

@dataclass
class GlobalConfig:
    fs: float = 250.0
    trial_len_sec: float = 1.5  # Length of the trial to use
    
    # Data paths (can be modified)
    data_path: str = "C:\\Users\\EEG_Dataset\\bcidatasetIV2a-master"
    
    # Classes to use (default to binary: 0=Left, 1=Right)
    selected_labels: List[int] = None
    
    time_cfg: TimeWindowConfig = None
    freq_cfg: FreqBandConfig = None
    divcsp_cfg: DivCSPConfig = None
    fs_cfg: FeatureSelectionConfig = None
    train_cfg: TrainingConfig = None
    svm_cfg: SVMConfig = None

    def __post_init__(self):
        if self.selected_labels is None:
            self.selected_labels = [0, 1]  # Default to Left vs Right

        if self.time_cfg is None:
            self.time_cfg = TimeWindowConfig(
                lengths=[0.2, 0.4, 0.7, 1.0, 1.25, 1.5],
                overlap=0.5,
            )
        if self.freq_cfg is None:
            self.freq_cfg = FreqBandConfig(
                min_freq=4.0,
                max_freq=40.0,
                bandwidths=[4.0, 8.0, 16.0, 32.0],
                overlap=0.5,
            )
        if self.divcsp_cfg is None:
            self.divcsp_cfg = DivCSPConfig()
        if self.fs_cfg is None:
            self.fs_cfg = FeatureSelectionConfig()
        if self.train_cfg is None:
            self.train_cfg = TrainingConfig()
        if self.svm_cfg is None:
            self.svm_cfg = SVMConfig()

# Global configuration instance
cfg = GlobalConfig()
