import numpy as np
import pandas as pd

from src.features import FEATURE_COLUMNS, build_preprocessor


def _dense(arr):
    return arr.toarray() if hasattr(arr, "toarray") else np.asarray(arr)


def test_feature_columns_has_all_13_inputs():
    assert len(FEATURE_COLUMNS) == 13
    assert len(set(FEATURE_COLUMNS)) == 13


def test_preprocessor_outputs_no_nans_even_with_missing_input(synthetic_data):
    X, _ = synthetic_data
    assert X.isna().any().any()  # fixture really contains missing values
    out = _dense(build_preprocessor().fit_transform(X))
    assert not np.isnan(out).any()
    assert out.shape[0] == len(X)


def test_numeric_columns_are_standardised(synthetic_data):
    X, _ = synthetic_data
    out = _dense(build_preprocessor().fit_transform(X))
    numeric = out[:, :5]  # the five numeric columns come first
    assert np.allclose(numeric.mean(axis=0), 0, atol=1e-8)
    assert np.allclose(numeric.std(axis=0), 1, atol=1e-6)


def test_unseen_category_is_ignored_not_an_error(synthetic_data):
    X, _ = synthetic_data
    pre = build_preprocessor().fit(X)
    unseen = X.head(2).copy()
    unseen["cp"] = 99.0
    assert pre.transform(unseen).shape[0] == 2


def test_preprocessing_is_deterministic(synthetic_data):
    X, _ = synthetic_data
    a = _dense(build_preprocessor().fit_transform(X))
    b = _dense(build_preprocessor().fit_transform(X))
    assert np.array_equal(a, b)


def test_all_missing_row_is_imputed_from_training_statistics(synthetic_data):
    X, _ = synthetic_data
    pre = build_preprocessor().fit(X)
    row = pd.DataFrame([{c: np.nan for c in FEATURE_COLUMNS}])
    out = _dense(pre.transform(row))
    assert not np.isnan(out).any()
    # median imputation then scaling puts the numeric part near zero
    assert np.all(np.abs(out[0, :5]) < 0.5)
