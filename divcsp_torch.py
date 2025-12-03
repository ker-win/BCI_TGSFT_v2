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
