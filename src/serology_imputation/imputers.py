"""
imputers.py

Two imputers for end-to-end serology missing-value imputation:
1) IterativeImputer + RandomForestRegressor (classical ML baseline)
2) PyTorch AutoencoderImputer (deep learning)

Designed to work with NumPy arrays or pandas DataFrames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

try:
    import pandas as pd
except ImportError:  # pandas optional
    pd = None

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge

import torch
from torch import nn
import random


ArrayLike = Union[np.ndarray, "pd.DataFrame"]


def _to_numpy(X: ArrayLike) -> np.ndarray:
    if pd is not None and isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=float)
    return np.asarray(X, dtype=float)


def _restore_type(X_original: ArrayLike, X_np: np.ndarray) -> ArrayLike:
    if pd is not None and isinstance(X_original, pd.DataFrame):
        return pd.DataFrame(X_np, index=X_original.index, columns=X_original.columns)
    return X_np


def set_global_seed(seed: int, torch_threads: int = 1) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Safe-ish in Ray workers; may be locked in some environments
    try:
        torch.set_num_threads(torch_threads)
    except RuntimeError:
        pass



# ----------------------------
# 1) IterativeImputer + RF
# ----------------------------

@dataclass(frozen=True)
class IterativeRFConfig:
    max_iter: int = 20
    tol: float = 1e-2
    n_estimators: int = 50
    max_depth: Optional[int] = 10
    min_samples_leaf: int = 2
    max_features: str = "sqrt"
    # IMPORTANT for Ray: keep this 1 to avoid oversubscription
    n_jobs: int = 1


class IterativeRFImputer(BaseEstimator, TransformerMixin):
    """
    Iterative imputation where each feature with missing values is modeled
    using a RandomForestRegressor in a round-robin fashion.

    Note: IterativeImputer is stochastic; set random_state for reproducibility.
    """

    def __init__(self, seed: int = 42, config: IterativeRFConfig = IterativeRFConfig()):
        self.seed = seed
        self.config = config
        self.imputer_: Optional[IterativeImputer] = None

    def fit(self, X: ArrayLike, y=None):
        X_np = _to_numpy(X)

        rf = RandomForestRegressor(
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            min_samples_leaf=self.config.min_samples_leaf,
            max_features=self.config.max_features,
            n_jobs=self.config.n_jobs,
            random_state=self.seed,
        )

        self.imputer_ = IterativeImputer(
            estimator=rf,
            max_iter=self.config.max_iter,
            tol=self.config.tol,
            random_state=self.seed,
            imputation_order="random",
            skip_complete=True,
        )
        self.imputer_.fit(X_np)
        return self

    def transform(self, X: ArrayLike):
        if self.imputer_ is None:
            raise RuntimeError("IterativeRFImputer not fitted. Call fit() first.")
        X_np = _to_numpy(X)
        X_imp = self.imputer_.transform(X_np)
        X_imp = np.rint(X_imp)
        return _restore_type(X, X_imp)

    def fit_transform(self, X: ArrayLike, y=None):
        return self.fit(X).transform(X)


# ----------------------------
# 2) PyTorch Autoencoder Imputer
# ----------------------------

class NumericalAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


@dataclass(frozen=True)
class AutoencoderConfig:
    hidden_dim: int = 64
    latent_dim: int = 32
    epochs: int = 30
    batch_size: int = 256
    lr: float = 1e-3
    # Robust to outliers:
    loss: str = "smoothl1"  # "mse" or "smoothl1"
    # IMPORTANT for Ray: keep this 1 to avoid oversubscription
    torch_threads: int = 1


class AutoencoderImputer(BaseEstimator, TransformerMixin):
    """
    Masked-loss autoencoder imputer:
    - initialize missing values with column means
    - standardize
    - train AE with loss computed ONLY on observed entries
    - reconstruct and fill missing entries with reconstructions
    """

    def __init__(self, seed: int = 42, config: AutoencoderConfig = AutoencoderConfig()):
        self.seed = seed
        self.config = config

        self.scaler_ = StandardScaler()
        self.model_: Optional[NumericalAutoencoder] = None
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _make_loss(self):
        if self.config.loss.lower() == "mse":
            return nn.MSELoss(reduction="none")
        return nn.SmoothL1Loss(reduction="none")

    def fit(self, X: ArrayLike, y=None):
        set_global_seed(self.seed, torch_threads=self.config.torch_threads)

        X_np = _to_numpy(X)
        if X_np.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X_np.shape}")

        mask = ~np.isnan(X_np)  # observed entries
        col_means = np.nanmean(X_np, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)  # all-NaN columns -> 0

        X_init = np.where(mask, X_np, col_means)
        X_scaled = self.scaler_.fit_transform(X_init)

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=self.device_)
        mask_tensor = torch.tensor(mask, dtype=torch.bool, device=self.device_)

        n_samples, n_features = X_tensor.shape
        self.model_ = NumericalAutoencoder(
            input_dim=n_features,
            hidden_dim=self.config.hidden_dim,
            latent_dim=self.config.latent_dim,
        ).to(self.device_)

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.config.lr)
        criterion = self._make_loss()

        self.model_.train()
        for _epoch in range(self.config.epochs):
            perm = torch.randperm(n_samples, device=self.device_)
            for start in range(0, n_samples, self.config.batch_size):
                idx = perm[start : start + self.config.batch_size]
                batch_x = X_tensor[idx]
                batch_mask = mask_tensor[idx]

                optimizer.zero_grad(set_to_none=True)
                recon = self.model_(batch_x)

                loss_matrix = criterion(recon, batch_x)
                loss = loss_matrix[batch_mask].mean()

                loss.backward()
                optimizer.step()

        return self

    def transform(self, X: ArrayLike):
        if self.model_ is None:
            raise RuntimeError("AutoencoderImputer not fitted. Call fit() first.")

        X_np = _to_numpy(X)
        mask = ~np.isnan(X_np)

        col_means = np.nanmean(X_np, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)

        X_init = np.where(mask, X_np, col_means)
        X_scaled = self.scaler_.transform(X_init)

        X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=self.device_)

        self.model_.eval()
        with torch.no_grad():
            recon = self.model_(X_tensor).cpu().numpy()

        X_recon = self.scaler_.inverse_transform(recon)
        X_out = np.where(mask, X_np, X_recon)
        X_out = np.rint(X_out)
        return _restore_type(X, X_out)

    def fit_transform(self, X: ArrayLike, y=None):
        return self.fit(X).transform(X)


# ----------------------------
# 1b) IterativeImputer + BayesianRidge
# ----------------------------

@dataclass(frozen=True)
class IterativeBayesRidgeConfig:
    max_iter: int = 20
    tol: float = 1e-2
    # You can change this if you want; "ascending"/"descending" are also fine
    imputation_order: str = "random"
    skip_complete: bool = True


class IterativeBayesRidgeImputer(BaseEstimator, TransformerMixin):
    """
    Iterative imputation with BayesianRidge (fast baseline).
    IterativeImputer is stochastic due to feature ordering and sampling; random_state controls that.
    """

    def __init__(self, seed: int = 42, config: IterativeBayesRidgeConfig = IterativeBayesRidgeConfig()):
        self.seed = seed
        self.config = config
        self.imputer_: Optional[IterativeImputer] = None

    def fit(self, X: ArrayLike, y=None):
        X_np = _to_numpy(X)

        br = BayesianRidge()

        self.imputer_ = IterativeImputer(
            estimator=br,
            max_iter=self.config.max_iter,
            tol=self.config.tol,
            random_state=self.seed,
            imputation_order=self.config.imputation_order,
            skip_complete=self.config.skip_complete,
        )
        self.imputer_.fit(X_np)
        return self

    def transform(self, X: ArrayLike):
        if self.imputer_ is None:
            raise RuntimeError("IterativeBayesRidgeImputer not fitted. Call fit() first.")
        X_np = _to_numpy(X)
        X_imp = self.imputer_.transform(X_np)

        # Keep consistent with your other imputers (0/1/2 integer style)
        X_imp = np.rint(X_imp)
        return _restore_type(X, X_imp)

    def fit_transform(self, X: ArrayLike, y=None):
        return self.fit(X).transform(X)
