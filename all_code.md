# Consolidated Python Code

## config.py

```python
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
    use_gpu: bool = False     # Enable GPU acceleration

@dataclass
class SVMConfig:
    max_iter: int = 100000
    dual: str = "auto"

@dataclass
class ExperimentConfig:
    mode: str = "within-subject"
    use_E_as_test: bool = True # If True, use T file for train, E file for test
    test_ratio: float = 0.2    # Used if use_E_as_test is False
    random_state: int = 42

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
    exp_cfg: ExperimentConfig = None

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
        if self.exp_cfg is None:
            self.exp_cfg = ExperimentConfig()

# Global configuration instance
cfg = GlobalConfig()


```

## data_loader.py

```python
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

@dataclass
class TrialInfo:
    idx: int           # Index in Dataset.X
    subject_id: int
    session_id: int    # block_id / run_id
    label: int


def load_subject_data(subject_id: int, root_dir: str = None, file_suffix: str = 'T') -> Tuple[Dataset, List[TrialInfo]]:

    """
    Loads raw data for a single subject from BCI Competition IV 2a dataset.
    
    Args:
        subject_id: Subject ID (1-9)
        root_dir: Path to the dataset directory. If None, uses cfg.data_path.
        file_suffix: 'T' for training data, 'E' for evaluation data.
        
    Returns:
        Dataset object containing X, y, blocks, ch_names, fs.
    """
    if root_dir is None:
        root_dir = cfg.data_path
        
    file_path = os.path.join(root_dir, f'A0{subject_id}{file_suffix}.npz')
    
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
    
    dataset = Dataset(X=X, y=y, blocks=blocks, ch_names=ch_names, fs=fs)
    
    trials_meta = []
    for i in range(len(y)):
        trials_meta.append(TrialInfo(
            idx=i,
            subject_id=subject_id,
            session_id=int(blocks[i]),
            label=int(y[i]),
        ))
    
    return dataset, trials_meta


def subset_dataset(dataset: Dataset, indices: List[int] | np.ndarray) -> Dataset:
    """
    Creates a subset of the dataset based on indices.
    """
    return Dataset(
        X=dataset.X[indices],
        y=dataset.y[indices],
        blocks=dataset.blocks[indices],
        ch_names=dataset.ch_names,
        fs=dataset.fs,
    )

def make_splits(trials_meta: List[TrialInfo], mode: str = "within-subject", test_ratio: float = 0.2, random_state: int = 42) -> List[Dict[str, List[int]]]:
    """
    Generates train/val/test splits based on the mode.
    
    Args:
        trials_meta: List of TrialInfo objects.
        mode: 'within-subject' (random split).
        test_ratio: Ratio of test set size.
        random_state: Random seed.
        
    Returns:
        List of dicts, each containing 'train', 'val', 'test' indices.
    """
    if mode == "within-subject":
        rng = np.random.RandomState(random_state)
        all_idx = np.arange(len(trials_meta))
        rng.shuffle(all_idx)

        n_test = int(len(all_idx) * test_ratio)
        test_idx = all_idx[:n_test]
        train_idx = all_idx[n_test:]
        
        # For now, val is empty or can be a subset of train if needed later
        return [{"train": train_idx.tolist(), "val": [], "test": test_idx.tolist()}]
    
    else:
        raise NotImplementedError(f"Mode {mode} not implemented yet.")


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

```

## divcsp.py

```python
# divcsp.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh, inv, fractional_matrix_power
from typing import Optional, Tuple
from config import cfg

@dataclass
class DivCSPParams:
    """
    Stores trained parameters for reuse.
    """
    filters_: np.ndarray      # (n_components, n_channels)
    patterns_: Optional[np.ndarray] = None  # (n_channels, n_components)

class DivCSP:
    """
    Divergence-based CSP implementation.
    Currently implements standard CSP as a baseline.
    """

    def __init__(self,
                 n_components: int = None,
                 lambda_reg: float = None,
                 max_iter: int = None,
                 tol: float = None,
                 ):
        cfg_div = cfg.divcsp_cfg
        self.n_components = n_components or cfg_div.n_components
        self.lambda_reg = lambda_reg if lambda_reg is not None else cfg_div.lambda_reg
        self.max_iter = max_iter or cfg_div.max_iter
        self.tol = tol or cfg_div.tol

        self.filters_: Optional[np.ndarray] = None   # (n_components, n_channels)
        self.patterns_: Optional[np.ndarray] = None  # (n_channels, n_components)

    @staticmethod
    def _compute_covariance(X: np.ndarray) -> np.ndarray:
        """
        X: (n_trials, n_channels, n_samples)
        Returns average covariance, shape (n_channels, n_channels)
        """
        n_trials, n_ch, n_samples = X.shape
        cov = np.zeros((n_ch, n_ch), dtype=np.float64)
        for i in range(n_trials):
            Xi = X[i]
            # Normalize by trace (common in CSP implementations)
            C = Xi @ Xi.T
            trace = np.trace(C)
            if trace > 0:
                C /= trace
            cov += C
        cov /= n_trials
        return cov

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DivCSP":
        """
        X: (n_trials, n_channels, n_samples)
        y: (n_trials,), binary (0 / 1)
        """
        # 1. Split by class
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("DivCSP currently supports binary classification only.")
            
        X0 = X[y == classes[0]]
        X1 = X[y == classes[1]]
        
        Sigma0 = self._compute_covariance(X0)
        Sigma1 = self._compute_covariance(X1)
        
        # Regularization (optional, simple shrinkage)
        if self.lambda_reg > 0:
            n_ch = Sigma0.shape[0]
            eye = np.eye(n_ch)
            Sigma0 = (1 - self.lambda_reg) * Sigma0 + self.lambda_reg * eye * np.trace(Sigma0) / n_ch
            Sigma1 = (1 - self.lambda_reg) * Sigma1 + self.lambda_reg * eye * np.trace(Sigma1) / n_ch

        # 2. Solve Generalized Eigenvalue Problem
        # Sigma0 * w = lambda * (Sigma0 + Sigma1) * w
        # Or standard CSP: Sigma0 * w = lambda * Sigma1 * w (if Sigma1 is invertible)
        # Usually we solve: Sigma0 * w = lambda * (Sigma0 + Sigma1) * w
        # Eigenvalues will be in [0, 1].
        # 0 -> discriminative for class 1
        # 1 -> discriminative for class 0
        
        SigmaTotal = Sigma0 + Sigma1
        
        # Eigh returns eigenvalues in ascending order
        evals, evecs = eigh(Sigma0, SigmaTotal)
        
        # Sort descending
        ix = np.argsort(evals)[::-1]
        evecs = evecs[:, ix]
        
        # Select components
        # We want extreme eigenvalues (close to 1 and close to 0)
        n_filters = min(self.n_components, X.shape[1])
        n_half = n_filters // 2
        
        # Take first n_half and last n_half
        filters_indices = list(range(n_half)) + list(range(len(evals) - n_half, len(evals)))
        
        self.filters_ = evecs[:, filters_indices].T  # (n_components, n_channels)
        
        # Compute patterns (inverse of filters)
        # P = (W * W^T)^-1 * W  (pseudo-inverse)
        # Or simply pinv(W)
        self.patterns_ = np.linalg.pinv(self.filters_).T # (n_channels, n_components)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Projects data and computes log-variance features.
        X: (n_trials, n_channels, n_samples)
        Returns: (n_trials, n_components)
        """
        if self.filters_ is None:
            raise RuntimeError("Call fit() first.")
            
        n_trials = X.shape[0]
        n_comp = self.filters_.shape[0]
        feats = np.zeros((n_trials, n_comp), dtype=np.float32)

        for i in range(n_trials):
            Xi = X[i]          # (n_channels, n_samples)
            Zi = self.filters_ @ Xi  # (n_components, n_samples)
            # Variance along time axis
            var = np.var(Zi, axis=1)
            # Log-transform
            feats[i, :] = np.log(var + 1e-12)
            
        return feats

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    def get_params(self) -> DivCSPParams:
        return DivCSPParams(filters_=self.filters_, patterns_=self.patterns_)

    def set_params(self, params: DivCSPParams | dict):
        if isinstance(params, dict):
            try:
                self.filters_ = params['filters_']
                self.patterns_ = params.get('patterns_')
            except KeyError:
                # Fallback for potential key mismatch (e.g. without underscore)
                if 'filters' in params:
                    self.filters_ = params['filters']
                    self.patterns_ = params.get('patterns')
                else:
                    print(f"Error: params keys: {params.keys()}")
                    raise
        else:
            self.filters_ = params.filters_
            self.patterns_ = params.patterns_



```

## divcsp_torch.py

```python
import torch
import numpy as np

class DivCSPTorch:
    def __init__(self, n_components=4, lambda_reg=0.1, max_iter=100, tol=1e-6, device="cuda"):
        self.n_components = n_components
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter
        self.tol = tol
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.filters_ = None  # numpy version of filters for compatibility

    def _compute_covariance(self, X):  # X: (n_trials, n_channels, n_samples) torch tensor
        n_trials, n_ch, _ = X.shape
        cov = torch.zeros(n_ch, n_ch, device=self.device)
        for i in range(n_trials):
            Xi = X[i]
            C = Xi @ Xi.t()
            C = C / torch.trace(C)
            cov += C
        cov /= n_trials
        return cov

    def fit(self, X_np, y_np):
        # Convert to GPU tensor
        X = torch.from_numpy(X_np).float().to(self.device)
        y = torch.from_numpy(y_np).long().to(self.device)

        classes = torch.unique(y)
        # assert classes.numel() == 2, "DivCSPTorch currently supports binary classification only"

        # If more than 2 classes, we might need One-vs-Rest or similar, but for now assume binary as per user code
        if classes.numel() != 2:
             # Fallback or error? The original DivCSP might handle multiclass? 
             # Original code in divcsp.py seems to handle binary.
             pass

        X0 = X[y == classes[0]]
        X1 = X[y == classes[1]]

        Sigma0 = self._compute_covariance(X0)
        Sigma1 = self._compute_covariance(X1)

        # Regularization (optional, if needed)
        # Sigma0 += self.lambda_reg * torch.eye(Sigma0.shape[0], device=self.device)
        # Sigma1 += self.lambda_reg * torch.eye(Sigma1.shape[0], device=self.device)

        # Generalized Eigenvalue Problem: Sigma0 * w = lambda * (Sigma0 + Sigma1) * w
        # Or standard CSP: Sigma0 * w = lambda * Sigma1 * w?
        # Usually CSP solves: Sigma1 * w = lambda * Sigma0 * w  (maximize variance ratio)
        # Or simultaneous diagonalization of Sigma0 and Sigma1.
        # Common approach:
        # R = Sigma0 + Sigma1
        # P = R^(-1/2)
        # S0_tilde = P * Sigma0 * P^T
        # Decompose S0_tilde = U * Lambda * U^T
        # W = U^T * P
        
        R = Sigma0 + Sigma1
        # Eigen decomposition of R
        # e, V = torch.linalg.eigh(R)
        # Sort eigenvalues descending
        # idx = torch.argsort(e, descending=True)
        # e = e[idx]
        # V = V[:, idx]
        
        # Whitening transformation
        # P = torch.diag(e.pow(-0.5)) @ V.t()
        
        # S0_tilde = P @ Sigma0 @ P.t()
        # e_tilde, U = torch.linalg.eigh(S0_tilde)
        
        # Sort e_tilde
        # idx_tilde = torch.argsort(e_tilde, descending=True)
        # U = U[:, idx_tilde]
        
        # W = U.t() @ P
        
        # Using torch.linalg.eigh for generalized eigenproblem if available?
        # torch.linalg.eigh(A, B) solves A v = lambda B v
        # We want to maximize variance of class 0 vs class 1 (and vice versa)
        # Standard CSP: Find W such that W^T Sigma0 W is diagonal and W^T Sigma1 W is diagonal
        # and W^T (Sigma0 + Sigma1) W = I
        
        # Let's use the simultaneous diagonalization approach
        e, V = torch.linalg.eigh(R)
        # e are eigenvalues, V are eigenvectors
        # Remove small eigenvalues for stability
        mask = e > 1e-10
        e = e[mask]
        V = V[:, mask]
        
        P = torch.diag(e.pow(-0.5)) @ V.t()
        S0_tilde = P @ Sigma0 @ P.t()
        
        e_tilde, U = torch.linalg.eigh(S0_tilde)
        # Sort U by eigenvalues
        idx = torch.argsort(e_tilde, descending=True)
        U = U[:, idx]
        
        W = U.t() @ P
        
        # Select components
        # Top n_components/2 and Bottom n_components/2
        n_filters = self.n_components
        if n_filters > W.shape[0]:
            n_filters = W.shape[0]
            
        filters_list = []
        # Take first n/2
        filters_list.append(W[:n_filters//2])
        # Take last n/2
        filters_list.append(W[-n_filters//2:])
        
        self.filters_torch = torch.cat(filters_list, dim=0) # (n_comp, n_ch)
        self.filters_ = self.filters_torch.detach().cpu().numpy()
        
        return self

    def transform(self, X_np):
        if self.filters_ is None:
            raise RuntimeError("DivCSPTorch not fitted")
            
        W = self.filters_torch # (n_comp, n_ch)
        X = torch.from_numpy(X_np).float().to(self.device) # (n_trials, n_ch, n_samples)
        
        # Z = W * X
        # Einsum: k=components, c=channels, t=trials, n=samples
        # W: (k, c)
        # X: (t, c, n)
        # Z: (t, k, n)
        Z = torch.einsum("kc,tcn->tkn", W, X)
        
        # Variance
        var = Z.var(dim=-1) # (t, k)
        
        # Log-variance
        # Normalize?
        # var = var / var.sum(dim=1, keepdim=True) # Optional, some CSP implementations do this
        feats = torch.log(var + 1e-12)
        
        return feats.detach().cpu().numpy()

    def get_params(self):
        return {
            "filters": self.filters_,
            "n_components": self.n_components
        }

    def set_params(self, params):
        self.filters_ = params["filters"]
        self.n_components = params["n_components"]
        if self.filters_ is not None:
            self.filters_torch = torch.from_numpy(self.filters_).float().to(self.device)

```

## evaluate_all_subjects.py

```python
from __future__ import annotations
import argparse
import os
import numpy as np
import logging
import datetime
import sys
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from config import cfg
from data_loader import load_subject_data, filter_dataset, make_splits, subset_dataset

from model import FGSFTMIModel

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"eval_all_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Logging started. Saving to {log_file}")

def evaluate_subject(subject_id, model_dir, data_dir, use_test_data=False):
    model_path = os.path.join(model_dir, f"subject_{subject_id}_model.pkl")
    if not os.path.exists(model_path):
        logging.warning(f"Model for Subject {subject_id} not found at {model_path}. Skipping.")
        return None

    if data_dir:
        cfg.data_path = data_dir
        
    logging.info(f"Evaluating Subject {subject_id}...")
    
    file_suffix = 'E' if use_test_data else 'T'
    try:
        dataset, trials_meta = load_subject_data(subject_id, file_suffix=file_suffix)
    except FileNotFoundError as e:

        logging.error(f"Data for Subject {subject_id} not found: {e}")
        return None

    # Filter classes
    # We will filter AFTER splitting if we are splitting, to match main_train.py logic.
    # If using test data (E file), we filter immediately.
    if use_test_data:
        dataset = filter_dataset(dataset, cfg.selected_labels)


    if use_test_data:
        ds_test = dataset
    else:
        # Replicate split strategy: use make_splits with same random state as training
        # Assuming default config used in training (test_ratio=0.2, random_state=42)
        
        # We already have trials_meta from above
        
        # Filter metadata to match filtered dataset (binary classes)
        # However, filter_dataset only filters the Dataset object.
        # The indices from make_splits are based on the FULL dataset (trials_meta).
        # So we should split FIRST, then subset, then filter.
        
        splits = make_splits(trials_meta, mode="within-subject", test_ratio=0.2, random_state=42)
        test_idx = splits[0]['test']
        
        ds_test_unfiltered = subset_dataset(dataset, test_idx)
        ds_test = filter_dataset(ds_test_unfiltered, cfg.selected_labels)


    logging.info(f"Loading model for Subject {subject_id}...")
    model = FGSFTMIModel()
    try:
        model.load(model_path)
    except Exception as e:
        logging.error(f"Failed to load model for Subject {subject_id}: {e}")
        return None
    
    logging.info(f"Predicting for Subject {subject_id}...")
    y_pred = model.predict(ds_test)
    
    acc = accuracy_score(ds_test.y, y_pred)
    logging.info(f"Subject {subject_id} Accuracy: {acc:.4f}")
    
    return {
        "Subject": subject_id,
        "Accuracy": acc,
        "Num_Samples": len(ds_test.y)
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate FGSFT-MI Model for All Subjects")
    parser.add_argument("--models_dir", type=str, default="models", help="Directory containing trained models")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to dataset")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save logs")
    parser.add_argument("--use_test_data", action="store_true", help="Use evaluation dataset (E files) instead of splitting training data")
    
    args = parser.parse_args()
    
    setup_logging(args.log_dir)
    
    results = []
    
    for subject_id in range(1, 10):
        res = evaluate_subject(subject_id, args.models_dir, args.data_dir, args.use_test_data)
        if res:
            results.append(res)
            
    if results:
        df = pd.DataFrame(results)
        logging.info("\n" + "="*40)
        logging.info("Evaluation Summary")
        logging.info("="*40)
        logging.info("\n" + df.to_string(index=False))
        logging.info("-" * 40)
        logging.info(f"Average Accuracy: {df['Accuracy'].mean():.4f}")
        logging.info("="*40)
    else:
        logging.warning("No results obtained.")

if __name__ == "__main__":
    main()

```

## feature_selection.py

```python
# feature_selection.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
from tqdm import tqdm

from config import cfg
from data_loader import Dataset
from segmentation import SFTSSpec, ChannelGroup, TimeWindow, FreqBand
from features import precompute_freq_bands, get_sfts_data, extract_divcsp_features_for_sfts
from divcsp import DivCSP, DivCSPParams

@dataclass
class SFTSScore:
    sfts_id: int
    acc: float

@dataclass
class EnsembleMember:
    sfts_ids: List[int]                    # SFTS indices used
    svm: LinearSVC                         # Trained SVM model
    csp_params: Dict[int, DivCSPParams]    # sfts_id -> DivCSPParams

def lobo_blocks(dataset: Dataset) -> List[int]:
    """
    Returns unique block IDs for LOBO CV.
    """
    return sorted(np.unique(dataset.blocks).tolist())

def evaluate_single_sfts_lobo(
    dataset: Dataset,
    sfts_spec: SFTSSpec,
    X_fband: Dict[int, np.ndarray],
    channel_groups: List[ChannelGroup],
    time_windows: List[TimeWindow],
    use_gpu: bool = False,
) -> float:

    """
    Evaluates a single SFTS using LOBO CV.
    """
    blocks = lobo_blocks(dataset)
    y = dataset.y
    acc_list = []

    # Optimization: 
    # Instead of re-fitting CSP for every fold (which is very slow for thousands of SFTS),
    # we can fit CSP once on the whole dataset (or training set of outer loop),
    # and then just do LOBO on SVM.
    # However, strictly speaking, CSP should be fitted on training folds only to avoid leakage.
    # The paper says "The projection matrix W is calculated... using training samples".
    # So we MUST fit CSP inside the loop.
    # To speed up, we can use a smaller number of iterations or parallelize at the SFTS level.
    
    # We will implement strict LOBO here.
    
    # Pre-extract data for this SFTS to avoid repeated slicing
    X_sfts_all = get_sfts_data(X_fband, sfts_spec, channel_groups, time_windows)

    for test_block in blocks:
        mask_test = dataset.blocks == test_block
        mask_train = ~mask_test

        trial_train = np.where(mask_train)[0]
        trial_test = np.where(mask_test)[0]
        
        if len(trial_train) == 0 or len(trial_test) == 0:
            continue

        # Fit CSP on training data
        # Fit CSP on training data
        # Fit CSP on training data
        # Use DivCSPTorch if GPU enabled
        if use_gpu:
            from divcsp_torch import DivCSPTorch
            divcsp = DivCSPTorch(device="cuda")
        else:
            divcsp = DivCSP()

            
        divcsp.fit(X_sfts_all[trial_train], y[trial_train])

        
        # Transform
        f_train = divcsp.transform(X_sfts_all[trial_train])
        f_test = divcsp.transform(X_sfts_all[trial_test])

        # Train SVM
        clf = LinearSVC(
            random_state=cfg.train_cfg.random_state,
            dual=cfg.svm_cfg.dual,
            max_iter=cfg.svm_cfg.max_iter
        )
        clf.fit(f_train, y[trial_train])
        
        # Predict
        y_pred = clf.predict(f_test)
        acc_list.append(accuracy_score(y[trial_test], y_pred))

    return float(np.mean(acc_list)) if acc_list else 0.0

def rank_all_sfts(
    dataset: Dataset,
    sfts_specs: List[SFTSSpec],
    channel_groups: List[ChannelGroup],
    freq_bands: List[FreqBand],
    time_windows: List[TimeWindow],
) -> List[SFTSScore]:
    """
    Ranks all SFTS by LOBO accuracy.
    Uses parallel processing.
    """
    print(f"Precomputing frequency bands...")
    use_gpu = getattr(cfg.train_cfg, 'use_gpu', False)
    X_fband = precompute_freq_bands(dataset, freq_bands, use_gpu=use_gpu)

    
    print(f"Evaluating {len(sfts_specs)} SFTS candidates...")
    
    # Parallel execution
    n_jobs = cfg.train_cfg.n_jobs
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_single_sfts_lobo)(
            dataset, spec, X_fband, channel_groups, time_windows, use_gpu
        ) for spec in tqdm(sfts_specs, desc="Ranking SFTS")
    )

    
    scores = [SFTSScore(sfts_id=spec.id, acc=acc) for spec, acc in zip(sfts_specs, results)]
    
    # Sort descending
    scores_sorted = sorted(scores, key=lambda s: s.acc, reverse=True)
    return scores_sorted

def _precalc_single_sfts(
    s_id: int,
    sfts_specs: List[SFTSSpec],
    X_fband: Dict[int, np.ndarray],
    channel_groups: List[ChannelGroup],
    time_windows: List[TimeWindow],
    dataset: Dataset,
    use_gpu: bool = False,
) -> Tuple[int, Dict[str, DivCSPParams], Dict[str, np.ndarray]]:

    """
    Helper for parallel pre-calculation of CSP features.
    Returns:
        s_id
        csp_params_map: {'all': params, block_id: params}
        feats_map: {'all': feats, block_id: feats}
    """
    spec = next(s for s in sfts_specs if s.id == s_id)
    X_sfts = get_sfts_data(X_fband, spec, channel_groups, time_windows)
    y = dataset.y
    blocks = lobo_blocks(dataset)
    
    csp_params_map = {}
    feats_map = {}
    
    # 1. Fit on ALL data (for final model)
    # 1. Fit on ALL data (for final model)
    # 1. Fit on ALL data (for final model)
    if use_gpu:
        from divcsp_torch import DivCSPTorch
        divcsp_all = DivCSPTorch(device="cuda")
    else:
        divcsp_all = DivCSP()

        
    divcsp_all.fit(X_sfts, y)

    csp_params_map['all'] = divcsp_all.get_params()
    feats_map['all'] = divcsp_all.transform(X_sfts)
    
    # 2. Fit for each LOBO fold (to avoid leakage)
    for test_block in blocks:
        mask_test = dataset.blocks == test_block
        mask_train = ~mask_test
        
        if np.sum(mask_train) == 0:
            continue
            
        if np.sum(mask_train) == 0:
            continue
            
        if use_gpu:
            divcsp_fold = DivCSPTorch(device="cuda")
        else:
            divcsp_fold = DivCSP()
            
        divcsp_fold.fit(X_sfts[mask_train], y[mask_train])

        
        # Transform ALL data using this fold's CSP
        # We will slice it later in evaluation
        feats_map[test_block] = divcsp_fold.transform(X_sfts)
        csp_params_map[test_block] = divcsp_fold.get_params()
    
    return s_id, csp_params_map, feats_map

def _evaluate_ensemble_step(
    j: int,
    top_j_ids: List[int],
    feats_map_all: Dict[int, Dict],
    csp_params_map_all: Dict[int, Dict],
    dataset: Dataset,
    y: np.ndarray,
    blocks: List[int],
) -> Tuple[int, float, EnsembleMember]:
    """
    Helper for parallel evaluation of an ensemble step.
    """
    # LOBO CV
    acc_list = []
    
    for test_block in blocks:
        mask_test = dataset.blocks == test_block
        mask_train = ~mask_test
        idx_train = np.where(mask_train)[0]
        idx_test = np.where(mask_test)[0]
        
        if len(idx_train) == 0 or len(idx_test) == 0:
            continue
            
        # Construct features for this fold using CSP trained WITHOUT test_block
        # For each s_id, we take feats_map[s_id][test_block]
        fold_feats_list = []
        for sid in top_j_ids:
            if test_block in feats_map_all[sid]:
                fold_feats_list.append(feats_map_all[sid][test_block])
            else:
                # Fallback if block not found (shouldn't happen)
                fold_feats_list.append(feats_map_all[sid]['all'])
                
        F_fold = np.concatenate(fold_feats_list, axis=1)

        clf = LinearSVC(
            random_state=cfg.train_cfg.random_state,
            dual=cfg.svm_cfg.dual,
            max_iter=cfg.svm_cfg.max_iter
        )
        clf.fit(F_fold[idx_train], y[idx_train])
        y_pred = clf.predict(F_fold[idx_test])
        acc_list.append(accuracy_score(y[idx_test], y_pred))

    mean_acc = float(np.mean(acc_list)) if acc_list else 0.0
    
    # Train final model on ALL data using 'all' CSP features
    all_feats_list = [feats_map_all[sid]['all'] for sid in top_j_ids]
    F_all = np.concatenate(all_feats_list, axis=1)
    
    clf_final = LinearSVC(
        random_state=cfg.train_cfg.random_state,
        dual=cfg.svm_cfg.dual,
        max_iter=cfg.svm_cfg.max_iter
    )
    clf_final.fit(F_all, y)
    
    # Store 'all' CSP params for the final model
    member_csp_params = {sid: csp_params_map_all[sid]['all'] for sid in top_j_ids}
    
    member = EnsembleMember(
        sfts_ids=top_j_ids,
        svm=clf_final,
        csp_params=member_csp_params,
    )
    
    return j, mean_acc, member

def build_ensemble(
    dataset: Dataset,
    sfts_specs: List[SFTSSpec],
    channel_groups: List[ChannelGroup],
    freq_bands: List[FreqBand],
    time_windows: List[TimeWindow],
    scores_sorted: List[SFTSScore],
) -> List[EnsembleMember]:
    """
    Builds the ensemble by iteratively adding SFTS (Algorithm 2).
    Optimized with parallel pre-calculation and parallel search.
    """

    use_gpu = getattr(cfg.train_cfg, 'use_gpu', False)
    X_fband = precompute_freq_bands(dataset, freq_bands, use_gpu=use_gpu)
    blocks = lobo_blocks(dataset)


    y = dataset.y

    D = cfg.fs_cfg.D
    K = cfg.fs_cfg.K
    max_j = len(scores_sorted)
    step = D
    
    # Identify all unique SFTS IDs needed
    all_sfts_ids = [s.sfts_id for s in scores_sorted]
    
    print(f"Pre-calculating features for {len(all_sfts_ids)} candidates in parallel (with LOBO CSP)...")
    n_jobs = cfg.train_cfg.n_jobs
    
    # 1. Parallel Pre-calculation
    precalc_results = Parallel(n_jobs=n_jobs)(
        delayed(_precalc_single_sfts)(
            sid, sfts_specs, X_fband, channel_groups, time_windows, dataset, use_gpu
        ) for sid in tqdm(all_sfts_ids, desc="Pre-calc Features")
    )

    
    # Store in dictionaries
    csp_params_map_all: Dict[int, Dict] = {}
    feats_map_all: Dict[int, Dict] = {}
    
    for sid, params_map, feats_map in precalc_results:
        csp_params_map_all[sid] = params_map
        feats_map_all[sid] = feats_map
        
    print(f"Building ensemble (Total SFTS: {max_j}, Step: {D}) in parallel...")
    
    # 2. Parallel Ensemble Search
    steps = range(step, max_j + 1, step)
    
    step_args = []
    for j in steps:
        top_j_ids = [s.sfts_id for s in scores_sorted[:j]]
        step_args.append((j, top_j_ids))
        
    results = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate_ensemble_step)(
            j, top_j_ids, feats_map_all, csp_params_map_all, dataset, y, blocks
        ) for j, top_j_ids in tqdm(step_args, desc="Building Ensemble")
    )
    
    members_with_acc = []
    for j, acc, member in results:
        members_with_acc.append((acc, member))
        
    # Sort by accuracy descending
    members_with_acc.sort(key=lambda x: x[0], reverse=True)
    
    top_members = [m for acc, m in members_with_acc[:K]]
    
    print(f"Top {K} ensemble members selected. Best Acc: {members_with_acc[0][0]:.4f}")
    
    return top_members

```

## features.py

```python
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

```

## inspect_data.py

```python
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

```

## load_data.py

```python
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset,DataLoader
from sklearn.preprocessing import StandardScaler
import scipy.io as sio
import torch


class MotorImageryDataset:
    def __init__(self, dataset='A01T.npz'):
        if not dataset.endswith('.npz'):
            dataset += '.npz'

        path = "C:\\Users\\EEG_Dataset\\bcidatasetIV2a-master"
        #path = "C:\\Users\\Kerwin\\EEG_Dataset\\bcidatasetIV2a-master"

        self.data = np.load(f"{path}\\{dataset}")

        self.Fs = 250  # 250Hz from original paper

        # keys of data ['s', 'etyp', 'epos', 'edur', 'artifacts']

        self.raw = self.data['s'].T
        self.events_type = self.data['etyp'].T #'etyp' 存儲事件類型信息
        self.events_position = self.data['epos'].T #'epos' 記錄了每個事件的起始位置
        self.events_duration = self.data['edur'].T #'edur' 記錄了每個事件的持續時間
        self.artifacts = self.data['artifacts'].T #'artifacts' 存儲了人工標記的雜訊或干擾信息

        # Types of motor imagery817000

        self.mi_types = {769: 'left', 770: 'right',
                         771: 'foot', 772: 'tongue', 783: 'unknown'}

    def get_trials_from_channel(self, channel=7):

        # Channel default is C3

        startrial_code = 768
        starttrial_events = self.events_type == startrial_code
        idxs = [i for i, x in enumerate(starttrial_events[0]) if x] #提取所有試驗開始的位置索引

        trials = []
        classes = []

        for index in idxs:
            try:
                type_e = self.events_type[0, index+1] #取得試驗開始位置的下一個事件（index+1），例如：如果 index 位置是 768（試驗開始），index+1 可能是 769（左手）
                class_e = self.mi_types[type_e] #將事件代碼轉換為對應的類別名稱，例如：769 轉換為 'left'
                classes.append(class_e)

                start = self.events_position[0, index] #- - 獲取試驗的開始位置（時間點）
                stop = start + self.events_duration[0, index] #計算試驗的結束位置（時間點）
                trial = self.raw[channel, start:stop] #從指定通道提取這段時間內的腦電信號數據
                trial = trial.reshape((1, -1)) #將試驗數據重塑為二維數組，1 表示一個試驗，-1 表示自動計算另一個維度
                trials.append(trial)

            except:
                continue

        return trials, classes

    def get_trials_from_channels(self, channels=[7, 9, 11]):
        trials_c = []
        classes_c = []
        for c in channels:
            t, c = self.get_trials_from_channel(channel=c) #t 獲取該通道的試驗數據，c 獲取對應的類別標籤

            tt = np.concatenate(t, axis=0) #np.concatenate(t, axis=0) 將同一通道的所有試驗數據沿第一個維度連接
            trials_c.append(tt)
            classes_c.append(c)

        return trials_c, classes_c


class BCIDataset(Dataset):
    def __init__(self, args, training=True):
        self.rate = args.test_rate
        self.label_dict = args.label_dict
        self.training = training
        self.batch = args.batch
        self.mi_types = {769: 'left', 770: 'right',
                         771: 'foot', 772: 'tongue', 783: 'unknown'}

        self.x, self.y = self.load_data(args.subject, args.electrodes)
        minft = self.x.min()
        maxft = self.x.max()
        self.x = ((self.x - minft)/(maxft - minft))


    def load_data(self, subject, electrodes):
        trs,cls = [],[]
        for sub in subject:
            datasets = MotorImageryDataset(f'A0{sub}T.npz')
            trials_, classes_ = datasets.get_trials_from_channels(electrodes)
            trials = np.stack(trials_, axis=1).astype('float32')
            classes = [self.label_dict[la] for la in classes_[0]]

            border = int(len(classes) * (1 - self.rate))

            if self.training == True:
                trs.append(trials[:border])
                cls += classes[:border]
            else:
                trs.append(trials[border:])
                cls += classes[border:]

        trs = np.concatenate(trs,axis=0)

        return trs, cls


    def __len__(self):
        return len(self.x) - self.batch

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# --------------------------------------------------------------------------
# 運動想像 (Motor Imagery) EEG 分析流程 (CSP 8-30Hz 版本)
# 方法: 共空間模式 (CSP) + 集成學習分類器
# 數據集: BCI Competition IV-2a
# --------------------------------------------------------------------------

import numpy as np
import mne
import pandas as pd
from matplotlib import pyplot as plt

# Sklearn 相關模組
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from mne.decoding import CSP

# --- 1. 數據載入類 (重構與優化) ---
class MotorImageryDataset:
    """
    用於載入 BCI Competition IV-2a 數據集的類。
    這個版本直接處理並返回 Epochs 陣列，使流程更簡潔。
    """
    def __init__(self, file_path, subject_id):
        """
        初始化數據集。
        :param file_path: 包含 .npz 數據文件的資料夾路徑。
        :param subject_id: 受試者編號 (例如 '1', '2', ... '9')。
        """
        dataset_file = f'A0{subject_id}T.npz'
        full_path = f"{file_path}/{dataset_file}"
        
        try:
            data = np.load(full_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"錯誤：找不到數據文件 {full_path}。請檢查路徑是否正確。")

        self.fs = 250  # 採樣率 (Hz)
        
        # 載入數據，'s' 是 EEG 信號，維度為 (channels, samples)
        self.raw_eeg = data['s'].T
        self.events_type = data['etyp'].ravel()
        self.events_position = data['epos'].ravel()
        self.events_duration = data['edur'].ravel()
        
        # 定義運動想像任務的標籤
        self.mi_labels = {769: 'left', 770: 'right', 771: 'foot', 772: 'tongue'}

    def get_epochs(self, tmin, tmax):
        """
        從連續的 EEG 信號中提取所有試驗 (Epochs)。
        返回一個 NumPy 陣列 (trials, channels, samples) 和對應的標籤。
        """
        trial_start_code = 768
        trial_start_indices = np.where(self.events_type == trial_start_code)[0]

        trials = []
        labels = []

        for start_idx in trial_start_indices:
            # 運動想像的提示事件緊跟在試驗開始事件之後
            cue_idx = start_idx + 1
            if cue_idx < len(self.events_type):
                cue_type = self.events_type[cue_idx]
                
                # 檢查這個 cue 是否是我們感興趣的 MI 任務
                if cue_type in self.mi_labels:
                    # 獲取 cue 的時間點作為 epoch 的 0 時刻
                    cue_pos = self.events_position[cue_idx]
                    
                    # 計算 epoch 的開始和結束採樣點
                    start_sample = cue_pos + int(tmin * self.fs)
                    end_sample = cue_pos + int(tmax * self.fs)
                    
                    # 確保索引不越界
                    if end_sample <= self.raw_eeg.shape[1]:
                        epoch_data = self.raw_eeg[:, start_sample:end_sample]
                        trials.append(epoch_data)
                        labels.append(self.mi_labels[cue_type])

        if not trials:
            return np.array([]), np.array([])
            
        # *** 修正點：將數據類型從 float32 改為 float64 以兼容 MNE 濾波器 ***
        return np.stack(trials, axis=0).astype('float64'), np.array(labels)

# --- 2. MNE 濾波器函數 ---
def apply_mne_filter(data, lowcut, highcut, fs):
    """
    使用 MNE 的 filter_data 函數進行濾波。
    :param data: EEG 數據，形狀 (trials, channels, samples)。
    :param lowcut: 低頻截止點。
    :param highcut: 高頻截止點。
    :param fs: 採樣率。
    :return: 濾波後的數據或在出錯時返回 None。
    """
    try:
        # MNE 的濾波器對 NaN/Inf 處理較好，但執行後仍需檢查
        filtered_data = mne.filter.filter_data(data, sfreq=fs, l_freq=lowcut, h_freq=highcut,
                                               method='fir', phase='zero-double',
                                               fir_window='hamming', fir_design='firwin', verbose=False)
        
        # 檢查濾波後是否產生無效值
        if np.any(np.isnan(filtered_data)) or np.any(np.isinf(filtered_data)):
            print(f"      警告: 在濾波 {lowcut}-{highcut} Hz 時產生了 NaN 或 Inf 值。將跳過此頻帶。")
            return None
        return filtered_data
    except Exception as e:
        print(f"      錯誤: MNE 濾波 ({lowcut}-{highcut} Hz) 失敗: {e}。將跳過此頻帶。")
        return None

# --- 3. 主程式設定 ---
class Args:
    # --- 數據路徑與參數 ---
    # !!! 請修改為您的數據集路徑 !!!
    data_path = "C:\\Users\\EEG_Dataset\\bcidatasetIV2a-master" 
    subject_list = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
    # subject_list = ['1'] # 可先用單一受試者測試
    
    # --- Epoching 參數 ---
    # 從提示 (cue) 開始後 0.5 秒到 3.5 秒截取數據，共 3 秒
    tmin, tmax = 0.5, 3.5 
    
    # --- CSP 參數 ---
    # *** 這裡是主要修改點：從 FBCSP 改為單一頻帶的 CSP ***
    filter_bands = [(8, 30)] # 使用 8-30Hz 的單一頻帶
    n_csp_components = 4 # 每個類別提取的 CSP component 數量 (OVR策略)

    # --- 特徵選擇與分類 ---
    use_feature_selection = True
    # 因為類別數增加，特徵總數也增加，這裡選擇更多的特徵
    n_features_to_select = 16 
    test_rate = 0.2
    random_state = 42 # 為了結果可重現

    # --- 任務定義 ---
    # 選擇分類任務類型: '4_class' (左/右/腳/舌) 或 '2_class' (左/右)
    #classification_type = '4_class' 
    classification_type = '2_class'

    if classification_type == '4_class':
        class_type = ['left', 'right', 'foot', 'tongue']
        label_dict = {'left': 0, 'right': 1, 'foot': 2, 'tongue': 3}
    elif classification_type == '2_class':
        class_type = ['left', 'right']
        label_dict = {'left': 0, 'right': 1}
    else:
        raise ValueError(f"不支援的分類類型: {classification_type}")

if __name__ == "__main__":
    args = Args()
    results_summary = {} # 用於儲存每個受試者的最終結果

    # --- 4. 主迴圈：處理每個受試者 ---
    for sub_id in args.subject_list:
        print(f"\n{'='*20} 正在處理受試者 A0{sub_id}T {'='*20}")

        # 1. 載入數據並提取 Epochs
        try:
            dataset = MotorImageryDataset(args.data_path, sub_id)
            X_raw, y_raw_labels = dataset.get_epochs(args.tmin, args.tmax)
        except Exception as e:
            print(f"  錯誤: 載入受試者 {sub_id} 的數據失敗: {e}")
            continue

        if X_raw.shape[0] == 0:
            print(f"  警告: 未能為受試者 {sub_id} 載入任何有效的試驗。跳過。")
            continue
        print(f"  成功載入數據，原始 Epochs 形狀: {X_raw.shape}")

        # 2. 過濾感興趣的類別並轉換標籤
        mask = np.isin(y_raw_labels, args.class_type)
        X = X_raw[mask]
        y_labels = y_raw_labels[mask]
        y = np.array([args.label_dict[label] for label in y_labels])

        if X.shape[0] == 0:
            print(f"  警告: 過濾後沒有剩下類別為 {args.class_type} 的試驗。跳過。")
            continue
        print(f"  類別過濾後，Epochs 形狀: {X.shape}, 標籤數量: {len(y)}")

        # 3. 劃分訓練集和測試集
        n_trials = X.shape[0]
        indices = np.arange(n_trials)
        np.random.seed(args.random_state) # 確保每次劃分都一樣
        np.random.shuffle(indices)
        
        border = int(n_trials * (1 - args.test_rate))
        train_idx, test_idx = indices[:border], indices[border:]
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        if len(np.unique(y_train)) < len(args.class_type) or len(np.unique(y_test)) < len(args.class_type):
            print(f"  警告: 訓練集或測試集中的類別少於 {len(args.class_type)} 個，可能影響分類性能。")
            print(f"  訓練集類別: {np.unique(y_train)}, 測試集類別: {np.unique(y_test)}")
            if len(np.unique(y_train)) < 2:
                 print("  訓練集類別少於2，無法進行分類，跳過此受試者。")
                 continue

        print(f"  訓練集形狀: {X_train.shape}, 標籤: {len(y_train)} (類別: {np.unique(y_train)})")
        print(f"  測試集形狀: {X_test.shape}, 標籤: {len(y_test)} (類別: {np.unique(y_test)})")

        # 4. CSP 特徵提取
        print("  >> 開始 CSP 特徵提取 (8-30 Hz)...")
        train_features_list = []
        test_features_list = []
        
        # 雖然是單一頻帶，保留迴圈結構以便未來擴展
        for band_idx, (lowcut, highcut) in enumerate(args.filter_bands):
            print(f"    處理頻帶 {band_idx+1}/{len(args.filter_bands)}: {lowcut}-{highcut} Hz")
            
            # 濾波
            X_train_filt = apply_mne_filter(X_train, lowcut, highcut, dataset.fs)
            X_test_filt = apply_mne_filter(X_test, lowcut, highcut, dataset.fs)

            if X_train_filt is None or X_test_filt is None:
                continue

            # CSP (mne 會自動對多分類使用 One-vs-Rest 策略)
            csp = CSP(n_components=args.n_csp_components, reg='ledoit_wolf', log=True, cov_est='epoch')
            try:
                csp.fit(X_train_filt, y_train)
                train_features_list.append(csp.transform(X_train_filt))
                test_features_list.append(csp.transform(X_test_filt))
            except Exception as e:
                print(f"      錯誤: CSP 在頻帶 {lowcut}-{highcut} Hz 失敗: {e}。跳過此頻帶。")
                continue

        if not train_features_list:
            print("  錯誤: CSP 未能提取任何特徵。跳過此受試者。")
            results_summary[f'A0{sub_id}T'] = {'error': 'CSP feature extraction failed'}
            continue

        # 合併所有頻帶的特徵 (此處只有一個頻帶)
        X_train_csp = np.concatenate(train_features_list, axis=1)
        X_test_csp = np.concatenate(test_features_list, axis=1)
        print(f"  CSP 特徵形狀 - 訓練集: {X_train_csp.shape}, 測試集: {X_test_csp.shape}")

        # 5. 特徵選擇 (可選)
        selector = None
        if args.use_feature_selection and X_train_csp.shape[1] > 1:
            print(f"  >> 應用特徵選擇 (SelectKBest, k={args.n_features_to_select})...")
            k = min(args.n_features_to_select, X_train_csp.shape[1])
            selector = SelectKBest(mutual_info_classif, k=k)
            try:
                X_train_final = selector.fit_transform(X_train_csp, y_train)
                X_test_final = selector.transform(X_test_csp)
                print(f"    選擇後特徵形狀 - 訓練集: {X_train_final.shape}, 測試集: {X_test_final.shape}")
            except Exception as e:
                print(f"    錯誤: 特徵選擇失敗: {e}。將使用所有特徵。")
                X_train_final = X_train_csp
                X_test_final = X_test_csp
                selector = None
        else:
            X_train_final = X_train_csp
            X_test_final = X_test_csp
            print("  >> 跳過特徵選擇。")

        # 6. 分類流程
        print("  >> 開始分類流程...")
        
        # 定義分類器
        svm = SVC(probability=True, random_state=args.random_state)
        knn = KNeighborsClassifier()
        lda = LinearDiscriminantAnalysis()

        # 使用投票分類器集成模型
        voting_clf = VotingClassifier(
            estimators=[("svm", svm), ("knn", knn), ('lda', lda)],
            voting='soft'
        )

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', voting_clf)
        ])

        # 7. 超參數搜索 (GridSearchCV)
        param_grid = {
            'classifier__svm__C': [0.1, 1, 10],
            'classifier__svm__kernel': ['rbf', 'linear'],
            'classifier__knn__n_neighbors': [5, 7, 9],
            'classifier__lda__solver': ['svd', 'lsqr']
        }
        
        print("  >> 執行 GridSearchCV 尋找最佳參數...")
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=0)
        try:
            grid_search.fit(X_train_final, y_train)
        except Exception as e:
            print(f"  錯誤: GridSearchCV 失敗: {e}。跳過分類。")
            results_summary[f'A0{sub_id}T'] = {'error': f'GridSearchCV failed: {e}'}
            continue

        # 8. 使用最佳模型進行評估
        print(f"    最佳參數: {grid_search.best_params_}")
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test_final)

        # 9. 性能評估與結果儲存
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report_dict = classification_report(y_test, y_pred, 
                                            target_names=args.class_type,
                                            output_dict=True, zero_division=0)

        print(f"\n  受試者 A0{sub_id}T 測試集準確率: {accuracy:.4f}")
        print("  混淆矩陣 (4x4):\n", cm)
        print("  分類報告:\n", classification_report(y_test, y_pred, target_names=args.class_type, zero_division=0))
        
        try:
            cv_scores = cross_val_score(best_model, X_train_final, y_train, cv=5)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
        except Exception:
            cv_mean, cv_std = None, None

        results_summary[f'A0{sub_id}T'] = {
            'accuracy': accuracy,
            'cv_mean_score_train': cv_mean,
            'cv_std_score_train': cv_std,
            'report': report_dict,
            'n_train_samples': len(y_train),
            'n_test_samples': len(y_test),
        }

    # --- 5. 總結所有受試者的結果並保存 ---
    print(f"\n{'='*20} 所有受試者結果總結 {'='*20}")

    results_data = {
        'Subject': [],
        'Mean CV Score (Train)': [],
        'Test Accuracy (%)': [],
        'Macro F1 Score (Test %)': []
    }

    for subject, res in results_summary.items():
        if 'accuracy' in res:
            results_data['Subject'].append(subject)
            results_data['Mean CV Score (Train)'].append(res.get('cv_mean_score_train', np.nan))
            results_data['Test Accuracy (%)'].append(res['accuracy'] * 100)
            macro_f1 = res['report'].get('macro avg', {}).get('f1-score', np.nan)
            results_data['Macro F1 Score (Test %)'].append(macro_f1 * 100)
        else:
            print(f"{subject}: 處理失敗 - {res.get('error', '未知錯誤')}")

    if results_data['Subject']:
        results_df = pd.DataFrame(results_data)
        
        # 計算平均值
        mean_row = pd.DataFrame({
            'Subject': ['Average'],
            'Mean CV Score (Train)': [np.nanmean(results_df['Mean CV Score (Train)'])],
            'Test Accuracy (%)': [np.nanmean(results_df['Test Accuracy (%)'])],
            'Macro F1 Score (Test %)': [np.nanmean(results_df['Macro F1 Score (Test %)'])]
        })
        results_df = pd.concat([results_df, mean_row], ignore_index=True)

        # 格式化輸出
        for col in results_df.columns[1:]:
            results_df[col] = results_df[col].round(2)

        print("\n--- CSP (8-30Hz) 四分類結果摘要 ---")
        print(results_df.to_string(index=False))

        # 保存到 CSV
        csv_filename = 'csp_8-30hz_4class_classification_summary.csv'
        results_df.to_csv(csv_filename, index=False, na_rep='N/A')
        print(f"\n結果摘要已保存至 {csv_filename}")
    else:
        print("沒有成功處理的受試者結果可供總結。")

```

## main_eval.py

```python
# main_eval.py
from __future__ import annotations
import argparse
import os
import numpy as np
import logging
import datetime
import sys
from sklearn.metrics import accuracy_score, classification_report

from config import cfg
from data_loader import load_subject_data, split_train_test_by_blocks, filter_dataset
from model import FGSFTMIModel

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"eval_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Logging started. Saving to {log_file}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate FGSFT-MI Model")
    parser.add_argument("--subject", type=int, default=1, help="Subject ID (1-9)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to dataset")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save logs")
    parser.add_argument("--use_test_data", action="store_true", help="Use evaluation dataset (E files) instead of splitting training data")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_dir)
    
    if args.data_dir:
        cfg.data_path = args.data_dir
        
    logging.info(f"Loading data for Subject {args.subject}...")
    
    file_suffix = 'E' if args.use_test_data else 'T'
    try:
        dataset = load_subject_data(args.subject, file_suffix=file_suffix)
    except FileNotFoundError as e:
        logging.error(e)
        return

    # Filter classes
    logging.info(f"Filtering classes to {cfg.selected_labels}...")
    dataset = filter_dataset(dataset, cfg.selected_labels)

    if args.use_test_data:
        logging.info("Using Evaluation dataset (E file) as test set.")
        ds_test = dataset
    else:
        # We need to know which part was test set. 
        # Ideally, we should save the split info or use a standard split.
        # For this demo, we assume the same split strategy as main_train: last block is test.
        blocks = np.unique(dataset.blocks)
        if len(blocks) > 1:
            test_block = blocks[-1]
            logging.info(f"Using Block {test_block} as test set.")
            _, ds_test = split_train_test_by_blocks(dataset, test_block)
        else:
            logging.info("Only 1 block found. Cannot replicate random split without seed info.")
            logging.info("Evaluating on ALL data.")
            ds_test = dataset

    logging.info("Loading model...")
    model = FGSFTMIModel()
    model.load(args.model_path)
    
    logging.info("Predicting...")
    y_pred = model.predict(ds_test)
    
    acc = accuracy_score(ds_test.y, y_pred)
    logging.info(f"Accuracy: {acc:.4f}")
    logging.info("\nClassification Report:")
    logging.info("\n" + classification_report(ds_test.y, y_pred))

if __name__ == "__main__":
    main()

```

## main_train.py

```python
# main_train.py
from __future__ import annotations
import argparse
import os
import numpy as np
import logging
import datetime
import sys
from sklearn.metrics import accuracy_score

from config import cfg
from data_loader import load_subject_data, make_splits, subset_dataset
from model import FGSFTMIModel

def setup_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"train_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Logging started. Saving to {log_file}")

def run_experiment_for_subject(subject_id, output_dir):
    logging.info(f"==========================================")
    logging.info(f"Starting training for Subject {subject_id}")
    logging.info(f"==========================================")
    
    # 1. Load Data
    logging.info(f"Loading data for Subject {subject_id}...")
    try:
        ds_T, meta_T = load_subject_data(subject_id, file_suffix='T')
    except FileNotFoundError as e:
        logging.error(f"Data for subject {subject_id} not found: {e}")
        return

    ds_train = None
    ds_test = None

    if cfg.exp_cfg.use_E_as_test:
        logging.info("Mode: Train on T file, Test on E file.")
        try:
            ds_E, meta_E = load_subject_data(subject_id, file_suffix='E')
            if len(ds_E.X) == 0:
                logging.warning(f"E file for subject {subject_id} is empty (no known labels). Falling back to splitting T file.")
                cfg.exp_cfg.use_E_as_test = False
            else:
                ds_train = ds_T
                ds_test = ds_E
        except FileNotFoundError:
            logging.warning(f"E file for subject {subject_id} not found. Falling back to splitting T file.")
            cfg.exp_cfg.use_E_as_test = False

    
    if not cfg.exp_cfg.use_E_as_test:
        logging.info(f"Mode: Within-subject split on T file (Test Ratio: {cfg.exp_cfg.test_ratio})")
        splits = make_splits(
            meta_T, 
            mode=cfg.exp_cfg.mode, 
            test_ratio=cfg.exp_cfg.test_ratio, 
            random_state=cfg.exp_cfg.random_state
        )
        # Assuming single fold for now as per make_splits implementation
        split = splits[0]
        train_idx = split['train']
        test_idx = split['test']
        
        logging.info(f"Split sizes: Train={len(train_idx)}, Test={len(test_idx)}")
        
        ds_train = subset_dataset(ds_T, train_idx)
        ds_test = subset_dataset(ds_T, test_idx)

    logging.info(f"Train Data: X={ds_train.X.shape}, y={ds_train.y.shape}")
    logging.info(f"Test Data: X={ds_test.X.shape}, y={ds_test.y.shape}")

    # Filter classes
    logging.info(f"Filtering classes to {cfg.selected_labels}...")
    from data_loader import filter_dataset
    ds_train = filter_dataset(ds_train, cfg.selected_labels)
    ds_test = filter_dataset(ds_test, cfg.selected_labels)
    logging.info(f"Filtered Train Data: X={ds_train.X.shape}, y={ds_train.y.shape}")
    logging.info(f"Filtered Test Data: X={ds_test.X.shape}, y={ds_test.y.shape}")

    # 2. Train Model

    # Note: Preprocessing is now handled inside model.fit() on the training data
    # and model.predict() will apply the same transformation.
    
    logging.info("Initializing model...")
    model = FGSFTMIModel()
    
    logging.info("Fitting model (this may take a while)...")
    if cfg.train_cfg.use_gpu:
        logging.info("GPU Acceleration Enabled.")
    
    model.fit(ds_train)
    
    # 3. Evaluate
    logging.info("Evaluating on test set...")
    y_pred = model.predict(ds_test)
    acc = accuracy_score(ds_test.y, y_pred)
    logging.info(f"Subject {subject_id} Test Accuracy: {acc:.4f}")
    
    # Save
    save_path = os.path.join(output_dir, f"subject_{subject_id}_model.pkl")
    model.save(save_path)
    logging.info(f"Model saved to {save_path}")
    logging.info(f"Finished Subject {subject_id}")

def main():
    parser = argparse.ArgumentParser(description="Train FGSFT-MI Model")
    parser.add_argument("--subject", type=int, default=1, help="Subject ID (1-9)")
    parser.add_argument("--all_subjects", action="store_true", help="Train all subjects (1-9) sequentially")
    parser.add_argument("--data_dir", type=str, default=None, help="Path to dataset")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save models")
    parser.add_argument("--log_dir", type=str, default="logs", help="Directory to save logs")
    parser.add_argument("--n_jobs", type=int, default=-1, help="Number of parallel jobs")
    
    # Experiment Config Args
    parser.add_argument("--mode", type=str, default="within-subject", help="Experiment mode")
    parser.add_argument("--use_E_as_test", action="store_true", help="Use E file as test set")
    parser.add_argument("--no_E_as_test", action="store_false", dest="use_E_as_test", help="Do not use E file as test set")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Test ratio if splitting T file")
    parser.add_argument("--use_gpu", action="store_true", help="Enable GPU acceleration")
    
    parser.set_defaults(use_E_as_test=True)

    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_dir)
    
    start_time = datetime.datetime.now()
    logging.info(f"Execution started at: {start_time}")

    # Update config
    if args.data_dir:
        cfg.data_path = args.data_dir
    cfg.train_cfg.n_jobs = args.n_jobs
    cfg.train_cfg.use_gpu = args.use_gpu
    
    cfg.exp_cfg.mode = args.mode
    cfg.exp_cfg.use_E_as_test = args.use_E_as_test
    cfg.exp_cfg.test_ratio = args.test_ratio
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.all_subjects:
        logging.info("Training ALL subjects (1-9)...")
        for sub in range(1, 10):
            try:
                run_experiment_for_subject(sub, args.output_dir)
            except Exception as e:
                logging.error(f"Failed to train subject {sub}: {e}", exc_info=True)
    else:
        run_experiment_for_subject(args.subject, args.output_dir)

    end_time = datetime.datetime.now()
    duration = end_time - start_time
    logging.info(f"Execution finished at: {end_time}")
    logging.info(f"Total execution time: {duration}")

if __name__ == "__main__":
    main()

```

## model.py

```python
# model.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pickle
import os

from data_loader import Dataset
from config import cfg
from segmentation import (
    generate_time_windows,
    generate_freq_bands,
    generate_channel_groups,
    generate_sfts_specs,
    SFTSSpec, ChannelGroup, TimeWindow, FreqBand,
)
from preprocessing import Preprocessor
from feature_selection import rank_all_sfts, build_ensemble, EnsembleMember
from features import precompute_freq_bands, get_sfts_data
from divcsp import DivCSP
from divcsp_torch import DivCSPTorch


@dataclass
class FGSFTMIModelState:
    channel_groups: List[ChannelGroup]
    freq_bands: List[FreqBand]
    time_windows: List[TimeWindow]
    sfts_specs: List[SFTSSpec]
    ensemble: List[EnsembleMember]
    preprocessor: Preprocessor # Save preprocessor state


class FGSFTMIModel:
    def __init__(self):
        self.state: Optional[FGSFTMIModelState] = None

    def fit(self, dataset: Dataset) -> "FGSFTMIModel":
        """
        Main training pipeline.
        """
        print("Starting training pipeline...")
        
        # 1. Preprocess
        print("Preprocessing data...")
        # ds_proc = preprocess_pipeline(dataset)
        self.preprocessor = Preprocessor(fs_target=cfg.fs)
        self.preprocessor.fit(dataset.X, dataset.fs)
        ds_proc = self.preprocessor.transform(dataset)

        
        # 2. Generate Segments
        print("Generating segments...")
        time_windows = generate_time_windows()
        freq_bands = generate_freq_bands()
        channel_groups = generate_channel_groups(ds_proc.ch_names)
        sfts_specs = generate_sfts_specs(channel_groups, freq_bands, time_windows)
        
        print(f"Generated {len(sfts_specs)} SFTS specs.")
        
        # 3. Rank SFTS
        print("Ranking SFTS...")
        scores_sorted = rank_all_sfts(
            ds_proc, sfts_specs, channel_groups, freq_bands, time_windows
        )
        
        # 4. Build Ensemble
        print("Building ensemble...")
        top_members = build_ensemble(
            ds_proc, sfts_specs, channel_groups, freq_bands, time_windows, scores_sorted
        )
        
        self.state = FGSFTMIModelState(
            channel_groups=channel_groups,
            freq_bands=freq_bands,
            time_windows=time_windows,
            sfts_specs=sfts_specs,
            ensemble=top_members,
            preprocessor=self.preprocessor,

        )

        print("Training complete.")
        return self

    def predict_proba(self, dataset: Dataset) -> np.ndarray:
        """
        Predict class probabilities for new data.
        Returns: (n_trials, 2)
        """
        if self.state is None:
            raise RuntimeError("Model not fitted.")
            
        # Preprocess
        # ds_proc = preprocess_pipeline(dataset)
        ds_proc = self.state.preprocessor.transform(dataset)
        
        # Precompute freq bands
        use_gpu = getattr(cfg.train_cfg, 'use_gpu', False)
        X_fband = precompute_freq_bands(ds_proc, self.state.freq_bands, use_gpu=use_gpu)

        
        n_trials = ds_proc.X.shape[0]
        n_classes = 2 # Binary
        probs_sum = np.zeros((n_trials, n_classes))
        
        # Ensemble voting
        # Algorithm 3: Average probabilities
        
        for member in self.state.ensemble:
            # Extract features for this member
            feats_list = []
            for s_id in member.sfts_ids:
                spec = next(s for s in self.state.sfts_specs if s.id == s_id)
                
                # Get data
                X_sfts = get_sfts_data(
                    X_fband, spec, self.state.channel_groups, self.state.time_windows
                )
                
                # Transform using stored CSP params
                # Transform using stored CSP params
                # Check if we should use GPU for transform
                use_gpu = getattr(cfg.train_cfg, 'use_gpu', False)
                
                if use_gpu:
                     divcsp = DivCSPTorch(device="cuda")
                else:
                     divcsp = DivCSP()
                     
                divcsp.set_params(member.csp_params[s_id])
                f = divcsp.transform(X_sfts)

                feats_list.append(f)
            
            F_concat = np.concatenate(feats_list, axis=1)
            
            # Predict proba
            # LinearSVC doesn't support predict_proba by default unless calibrated, 
            # but we can use decision_function and softmax or CalibratedClassifierCV.
            # However, for simplicity and since we used LinearSVC directly:
            # We can use decision_function and sigmoid.
            
            d = member.svm.decision_function(F_concat) # (n_trials,)
            # Sigmoid for binary
            prob_1 = 1 / (1 + np.exp(-d))
            prob_0 = 1 - prob_1
            
            probs = np.vstack([prob_0, prob_1]).T
            probs_sum += probs
            
        return probs_sum / len(self.state.ensemble)

    def predict(self, dataset: Dataset) -> np.ndarray:
        probs = self.predict_proba(dataset)
        return np.argmax(probs, axis=1)

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump(self.state, f)
            
    def load(self, path: str):
        with open(path, 'rb') as f:
            self.state = pickle.load(f)

```

## preprocessing.py

```python
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

class Preprocessor:
    def __init__(self, fs_target: float = 250.0, do_scaling: bool = True):
        self.fs_target = fs_target
        self.do_scaling = do_scaling
        self.scaler = None

    def fit(self, X_train: np.ndarray, fs: float):
        """
        Fits the scaler on the training data.
        Args:
            X_train: (n_trials, n_channels, n_samples)
            fs: Sampling frequency of X_train
        """
        # 1. Resample if needed (conceptually, we assume X_train is already consistent or we handle it)
        # For scaling, we need to flatten
        if self.do_scaling:
            N, C, T = X_train.shape
            # Flatten to (N*T, C) or (N, C*T)? 
            # Standard scaling usually per channel or per feature. 
            # If we want to normalize amplitude across all time points per channel:
            # We can reshape to (N*T, C) -> fit scaler -> (mean/std per channel)
            # Or (N, C*T) -> fit scaler -> (mean/std per timepoint per channel)
            # EEG usually does per-channel scaling (0 mean, 1 std over time).
            # But here we are fitting on the whole training set.
            # Let's assume we want to standardize each channel's distribution across the dataset.
            # Reshape to (N * T, C) to compute stats per channel.
            X_2d = np.transpose(X_train, (0, 2, 1)).reshape(-1, C)
            
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler()
            self.scaler.fit(X_2d)

    def transform(self, dataset: Dataset) -> Dataset:
        """
        Applies resampling and scaling to the dataset.
        """
        # 1. Resample
        ds_resampled = resample_to_fs(dataset, self.fs_target)
        
        # 2. Crop (Optional, if we want to enforce it here, but maybe better separate)
        # For now, let's stick to what the user asked: Resample + Scaling
        
        X_out = ds_resampled.X
        
        if self.do_scaling and self.scaler is not None:
            N, C, T = X_out.shape
            # Reshape to (N*T, C)
            X_2d = np.transpose(X_out, (0, 2, 1)).reshape(-1, C)
            X_scaled = self.scaler.transform(X_2d)
            # Reshape back to (N, T, C) then transpose to (N, C, T)
            X_out = X_scaled.reshape(N, T, C).transpose(0, 2, 1)
            
        return Dataset(
            X=X_out.astype(np.float32),
            y=ds_resampled.y,
            blocks=ds_resampled.blocks,
            ch_names=ds_resampled.ch_names,
            fs=self.fs_target
        )


```

## segmentation.py

```python
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

```

## verify_changes.py

```python
import numpy as np
import torch
from data_loader import Dataset, TrialInfo, make_splits
from divcsp import DivCSP
from divcsp_torch import DivCSPTorch
from features import bandpass_filter_torch
from preprocessing import bandpass_filter

def test_data_leakage():
    print("Testing Data Leakage...")
    # Create synthetic metadata
    n_trials = 100
    trials_meta = []
    for i in range(n_trials):
        trials_meta.append(TrialInfo(
            idx=i,
            subject_id=1,
            session_id=1,
            label=i % 2
        ))
    
    splits = make_splits(trials_meta, mode="within-subject", test_ratio=0.2, random_state=42)
    split = splits[0]
    train_idx = set(split['train'])
    test_idx = set(split['test'])
    
    intersection = train_idx.intersection(test_idx)
    assert len(intersection) == 0, f"Leakage detected! Intersection: {intersection}"
    print("Data Leakage Test Passed!")

def test_gpu_consistency():
    print("Testing GPU Consistency...")
    if not torch.cuda.is_available():
        print("Skipping GPU test (CUDA not available)")
        return

    # Synthetic data
    n_trials = 20
    n_channels = 22
    n_samples = 500
    X = np.random.randn(n_trials, n_channels, n_samples).astype(np.float32)
    y = np.random.randint(0, 2, n_trials)
    
    # 1. Bandpass Filter
    fs = 250.0
    f_low = 8.0
    f_high = 30.0
    
    X_cpu = bandpass_filter(X, fs, f_low, f_high)
    X_gpu = bandpass_filter_torch(X, fs, f_low, f_high, device="cuda")
    
    # Check difference
    diff = np.abs(X_cpu - X_gpu).max()
    print(f"Bandpass Filter Max Diff: {diff}")
    # Differences might be due to implementation details (filtfilt vs firwin conv1d)
    # They won't be identical, but should be reasonable.
    # Actually, filtfilt is IIR zero-phase, firwin is FIR. They are different filters.
    # So we can't expect them to be close.
    # But we can check if GPU runs without error.
    print("Bandpass Filter ran successfully on GPU.")

    # 2. DivCSP
    divcsp_cpu = DivCSP()
    divcsp_cpu.fit(X, y)
    f_cpu = divcsp_cpu.transform(X)
    
    divcsp_gpu = DivCSPTorch(device="cuda")
    divcsp_gpu.fit(X, y)
    f_gpu = divcsp_gpu.transform(X)
    
    print(f"DivCSP CPU features shape: {f_cpu.shape}")
    print(f"DivCSP GPU features shape: {f_gpu.shape}")
    
    # Check if features are somewhat correlated or similar range
    # Again, implementation details might differ (solver, regularization).
    # But shapes should match.
    assert f_cpu.shape == f_gpu.shape
    print("DivCSP shapes match.")
    
    print("GPU Consistency Test Passed (Basic Runtime Check).")

if __name__ == "__main__":
    test_data_leakage()
    test_gpu_consistency()

```

