import json
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from src.models.train import make_pipeline

SAMPLE = json.loads(Path("sample_input.json").read_text())


@pytest.fixture(scope="module")
def client(tmp_path_factory, synthetic_data):
    X, y = synthetic_data
    model_dir = tmp_path_factory.mktemp("model")
    joblib.dump(make_pipeline(LogisticRegression(max_iter=500)).fit(X, y),
                model_dir / "model.joblib")
    (model_dir / "metadata.json").write_text(json.dumps({"model": "unit-test-model"}))

    import src.api.app as app_module
    import src.predict as predict_module

    predict_module.MODEL_PATH = model_dir / "model.joblib"
    app_module.MODEL_PATH = model_dir / "model.joblib"
    app_module.METADATA_PATH = model_dir / "metadata.json"
    with TestClient(app_module.app) as c:  # runs the lifespan (model load)
        yield c


def test_health_reports_model_loaded(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_root_describes_service_and_model(client):
    body = client.get("/").json()
    assert body["model"] == "unit-test-model"
    assert body["docs"] == "/docs"


def test_predict_returns_prediction_and_confidence(client):
    r = client.post("/predict", json=SAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"prediction", "label", "confidence", "risk_probability", "model"}
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1.0
    assert 0.0 <= body["risk_probability"] <= 1.0
    assert "X-Request-ID" in r.headers


def test_predict_accepts_missing_optional_fields(client):
    payload = {k: v for k, v in SAMPLE.items() if k not in ("ca", "thal")}
    assert client.post("/predict", json=payload).status_code == 200


@pytest.mark.parametrize("field,value", [
    ("age", 500), ("sex", 2), ("cp", 9), ("trestbps", 5), ("chol", -1), ("fbs", 0.5),
    ("restecg", 7), ("thalach", 0), ("exang", 3), ("oldpeak", -2), ("slope", 0),
    ("ca", 9), ("thal", 5), ("age", "old"),
])
def test_invalid_values_are_rejected_with_422(client, field, value):
    r = client.post("/predict", json={**SAMPLE, field: value})
    assert r.status_code == 422


def test_missing_required_field_is_422(client):
    payload = {k: v for k, v in SAMPLE.items() if k != "age"}
    assert client.post("/predict", json=payload).status_code == 422


def test_unknown_field_is_rejected(client):
    assert client.post("/predict", json={**SAMPLE, "bmi": 22}).status_code == 422


def test_requests_are_logged(client, caplog):
    with caplog.at_level("INFO", logger="heart-api"):
        client.post("/predict", json=SAMPLE)
    text = caplog.text
    assert "POST /predict status=200" in text
    assert "prediction=" in text
