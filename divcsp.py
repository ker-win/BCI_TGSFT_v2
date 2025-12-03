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

    def set_params(self, params: DivCSPParams):
        self.filters_ = params.filters_
        self.patterns_ = params.patterns_
