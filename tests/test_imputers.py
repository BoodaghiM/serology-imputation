import numpy as np
import pytest

from serology_imputation.imputers import IterativeRFImputer, AutoencoderImputer


@pytest.mark.parametrize("imputer_cls", [IterativeRFImputer, AutoencoderImputer])
def test_imputer_fills_nans(imputer_cls):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 6)).astype(float)

    # introduce missingness
    mask = rng.random(X.shape) < 0.2
    X[mask] = np.nan

    imp = imputer_cls(seed=1)
    X_imp = imp.fit_transform(X)

    assert X_imp.shape == X.shape
    assert np.isnan(X_imp).sum() == 0


def test_iterative_rf_deterministic_with_seed():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(150, 6)).astype(float)
    X[rng.random(X.shape) < 0.15] = np.nan

    imp1 = IterativeRFImputer(seed=123)
    imp2 = IterativeRFImputer(seed=123)

    X1 = imp1.fit_transform(X)
    X2 = imp2.fit_transform(X)

    # RF + IterativeImputer should be deterministic given fixed seed
    assert np.allclose(X1, X2, atol=1e-6)

