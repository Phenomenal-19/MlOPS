import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.features import FEATURE_COLUMNS
from src.models import train

METRIC_KEYS = {"accuracy", "precision", "recall", "f1", "roc_auc"}


def test_at_least_two_candidate_models_with_grids():
    assert len(train.CANDIDATES) >= 2
    for model, grid in train.CANDIDATES.values():
        assert hasattr(model, "fit")
        assert grid
        assert all(k.startswith("model__") for k in grid)


def test_make_pipeline_has_preprocess_then_model():
    pipe = train.make_pipeline(LogisticRegression())
    assert [name for name, _ in pipe.steps] == ["preprocess", "model"]


def test_load_split_is_stratified_and_reproducible(tmp_path, synthetic_data):
    X, y = synthetic_data
    path = tmp_path / "d.csv"
    pd.concat([X, y], axis=1).to_csv(path, index=False)
    first = train.load_split(path)
    second = train.load_split(path)
    X_tr, X_te, y_tr, y_te = first
    assert len(X_te) == pytest.approx(0.2 * len(X), abs=1)
    assert list(X_tr.columns) == FEATURE_COLUMNS
    assert abs(y_tr.mean() - y_te.mean()) < 0.1
    pd.testing.assert_frame_equal(first[0], second[0])


def test_evaluate_candidate_returns_valid_metrics(synthetic_data):
    X, y = synthetic_data
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)
    res = train.evaluate_candidate(
        "lr", LogisticRegression(max_iter=500), {"model__C": [0.1, 1.0]},
        X_tr, X_te, y_tr, y_te)
    assert set(res["cv_mean"]) == METRIC_KEYS
    assert set(res["test"]) == METRIC_KEYS
    for value in (*res["cv_mean"].values(), *res["test"].values()):
        assert 0.0 <= value <= 1.0
    assert res["best_params"]["C"] in (0.1, 1.0)
    # signal is planted in the data, so the model must beat chance comfortably
    assert res["cv_mean"]["roc_auc"] > 0.7
    assert len(res["grid_results"]) == 2


def test_test_metrics_returns_all_metrics(synthetic_data):
    X, y = synthetic_data
    pipe = train.make_pipeline(LogisticRegression(max_iter=500)).fit(X, y)
    scores = train.test_metrics(pipe, X, y)
    assert set(scores) == METRIC_KEYS
    assert scores["roc_auc"] > 0.8  # in-sample fit on planted signal
