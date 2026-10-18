import json
from pathlib import Path

import joblib
import pytest
from sklearn.linear_model import LogisticRegression

from src.features import FEATURE_COLUMNS
from src.models.train import make_pipeline
from src.predict import LABELS, load_model, predict


@pytest.fixture(scope="module")
def model(synthetic_data):
    X, y = synthetic_data
    return make_pipeline(LogisticRegression(max_iter=500)).fit(X, y)


def _record(X, i=0) -> dict:
    return {k: (None if v != v else float(v)) for k, v in X.iloc[i].items()}


def test_predict_response_schema(model, synthetic_data):
    X, _ = synthetic_data
    out = predict(model, [_record(X, 10)])
    assert len(out) == 1
    r = out[0]
    assert set(r) == {"prediction", "label", "confidence", "risk_probability"}
    assert r["prediction"] in (0, 1)
    assert r["label"] == LABELS[r["prediction"]]


def test_confidence_is_probability_of_predicted_class(model, synthetic_data):
    X, _ = synthetic_data
    for r in predict(model, [_record(X, i) for i in range(20)]):
        assert 0.5 <= r["confidence"] <= 1.0
        expected = r["risk_probability"] if r["prediction"] else 1 - r["risk_probability"]
        assert r["confidence"] == pytest.approx(expected, abs=1e-3)


def test_batch_prediction_matches_row_by_row(model, synthetic_data):
    X, _ = synthetic_data
    records = [_record(X, i) for i in range(5)]
    batch = predict(model, records)
    single = [predict(model, [r])[0] for r in records]
    assert batch == single


def test_missing_and_unknown_fields_are_tolerated(model, synthetic_data):
    X, _ = synthetic_data
    rec = _record(X, 3)
    del rec["ca"], rec["thal"]
    rec["not_a_feature"] = "ignored"
    out = predict(model, [rec])
    assert 0.0 <= out[0]["risk_probability"] <= 1.0


def test_sample_input_file_matches_the_feature_schema():
    sample = json.loads(Path("sample_input.json").read_text())
    assert set(sample) == set(FEATURE_COLUMNS)


def test_load_model_round_trip(tmp_path, model, synthetic_data):
    X, _ = synthetic_data
    path = tmp_path / "m.joblib"
    joblib.dump(model, path)
    loaded = load_model(path)
    rec = _record(X, 7)
    assert predict(loaded, [rec]) == predict(model, [rec])
