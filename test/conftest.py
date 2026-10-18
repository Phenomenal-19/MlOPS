"""Shared fixtures: small synthetic patient data so tests need no network or files."""
import numpy as np
import pandas as pd
import pytest

from src.features import FEATURE_COLUMNS


@pytest.fixture(scope="session")
def raw_frame() -> pd.DataFrame:
    """Frame shaped like the raw Cleveland file (includes `num`, '?' placeholders)."""
    return pd.DataFrame({
        "age": [63, 67, 67, 41, 41, 41],
        "sex": [1, 1, 1, 0, 0, 0],
        "cp": [1, 4, 4, 2, 2, 2],
        "trestbps": [145, 160, 120, 130, 130, 130],
        "chol": [233, 286, 229, 204, 204, 204],
        "fbs": [1, 0, 0, 0, 0, 0],
        "restecg": [2, 2, 2, 2, 2, 2],
        "thalach": [150, 108, 129, 172, 172, 172],
        "exang": [0, 1, 1, 0, 0, 0],
        "oldpeak": [2.3, 1.5, 2.6, 1.4, 1.4, 1.4],
        "slope": [3, 2, 2, 1, 1, 1],
        "ca": ["0.0", "3.0", "?", "0.0", "0.0", "0.0"],  # last two rows duplicate row 3
        "thal": ["6.0", "3.0", "7.0", "?", "?", "?"],
        "num": [0, 2, 1, 0, 0, 0],
    })


@pytest.fixture(scope="session")
def synthetic_data() -> tuple[pd.DataFrame, pd.Series]:
    """150 synthetic patients whose risk depends on a few features, with some NaNs."""
    rng = np.random.default_rng(0)
    n = 150
    X = pd.DataFrame({
        "age": rng.integers(30, 78, n).astype(float),
        "trestbps": rng.normal(131, 17, n).round(),
        "chol": rng.normal(246, 51, n).round(),
        "thalach": rng.normal(150, 22, n).round(),
        "oldpeak": rng.gamma(1.2, 0.9, n).round(1),
        "sex": rng.integers(0, 2, n).astype(float),
        "cp": rng.integers(1, 5, n).astype(float),
        "fbs": rng.integers(0, 2, n).astype(float),
        "restecg": rng.integers(0, 3, n).astype(float),
        "exang": rng.integers(0, 2, n).astype(float),
        "slope": rng.integers(1, 4, n).astype(float),
        "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3.0, 6.0, 7.0], n),
    })[FEATURE_COLUMNS]
    score = 0.05 * (X["age"] - 54) - 0.04 * (X["thalach"] - 150) + 0.9 * X["exang"] + 0.8 * X["ca"]
    y = pd.Series((score + rng.normal(0, 1, n) > 0.3).astype(int), name="target")
    X.loc[X.index[:4], "ca"] = np.nan
    X.loc[X.index[4:6], "thal"] = np.nan
    return X, y
