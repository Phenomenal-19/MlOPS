import json
import logging
from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY
from sklearn.linear_model import LogisticRegression

from src.api.observability import JsonFormatter
from src.models.train import make_pipeline

SAMPLE = json.loads(Path("sample_input.json").read_text())


@pytest.fixture(scope="module")
def client(tmp_path_factory, synthetic_data):
    X, y = synthetic_data
    model_dir = tmp_path_factory.mktemp("metrics-model")
    joblib.dump(make_pipeline(LogisticRegression(max_iter=500)).fit(X, y),
                model_dir / "model.joblib")
    (model_dir / "metadata.json").write_text(json.dumps({"model": "metrics-test-model"}))

    import src.api.app as app_module
    import src.predict as predict_module

    predict_module.MODEL_PATH = model_dir / "model.joblib"
    app_module.MODEL_PATH = model_dir / "model.joblib"
    app_module.METADATA_PATH = model_dir / "metadata.json"
    with TestClient(app_module.app) as c:
        yield c


def _value(name: str, **labels) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


def test_metrics_endpoint_exposes_prometheus_text(client):
    client.post("/predict", json=SAMPLE)
    r = client.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")
    for name in ("api_http_requests_total", "api_http_request_duration_seconds_bucket",
                 "api_predictions_total", "api_prediction_confidence_bucket",
                 "api_model_info"):
        assert name in r.text


def test_request_counter_tracks_status_codes(client):
    ok = _value("api_http_requests_total", method="POST", path="/predict", status="200")
    bad = _value("api_http_requests_total", method="POST", path="/predict", status="422")
    client.post("/predict", json=SAMPLE)
    client.post("/predict", json={"age": 500})
    assert _value("api_http_requests_total", method="POST", path="/predict",
                  status="200") == ok + 1
    assert _value("api_http_requests_total", method="POST", path="/predict",
                  status="422") == bad + 1


def test_prediction_counter_and_confidence_histogram_increase(client):
    before = sum(_value("api_predictions_total", label=lbl)
                 for lbl in ("heart disease", "no heart disease"))
    conf_before = _value("api_prediction_confidence_count")
    client.post("/predict", json=SAMPLE)
    after = sum(_value("api_predictions_total", label=lbl)
                for lbl in ("heart disease", "no heart disease"))
    assert after == before + 1
    assert _value("api_prediction_confidence_count") == conf_before + 1


def test_unknown_urls_share_one_label_so_cardinality_stays_bounded(client):
    before = _value("api_http_requests_total", method="GET", path="unmatched", status="404")
    client.get("/nothing/here/1")
    client.get("/nothing/here/2")
    assert _value("api_http_requests_total", method="GET", path="unmatched",
                  status="404") == before + 2
    assert "nothing/here" not in client.get("/metrics").text


def test_metrics_scrapes_are_not_counted_as_api_traffic(client):
    client.get("/metrics")
    assert "path=\"/metrics\"" not in client.get("/metrics").text


def test_latency_histogram_records_observations(client):
    before = _value("api_http_request_duration_seconds_count", method="POST", path="/predict")
    client.post("/predict", json=SAMPLE)
    assert _value("api_http_request_duration_seconds_count", method="POST",
                  path="/predict") == before + 1


def test_model_info_gauge_carries_the_model_name(client):
    assert _value("api_model_info", model="metrics-test-model") == 1.0


def test_probe_endpoints_log_at_debug_not_info(client, caplog):
    with caplog.at_level(logging.INFO, logger="heart-api"):
        client.get("/health")
        client.get("/metrics")
    assert "GET /health" not in caplog.text
    assert "GET /metrics" not in caplog.text


def test_json_formatter_emits_parsable_structured_records():
    record = logging.LogRecord("heart-api", logging.INFO, __file__, 1,
                               "req=%s %s", ("abc", "POST /predict"), None)
    record.request_id, record.status, record.latency_ms = "abc", 200, 12.3
    out = json.loads(JsonFormatter().format(record))
    assert out["level"] == "INFO"
    assert out["msg"] == "req=abc POST /predict"
    assert (out["request_id"], out["status"], out["latency_ms"]) == ("abc", 200, 12.3)
    assert "ts" in out


def test_json_formatter_includes_exception_text():
    try:
        raise ValueError("boom")
    except ValueError:
        import sys
        record = logging.LogRecord("heart-api", logging.ERROR, __file__, 1, "failed", (),
                                   sys.exc_info())
    assert "ValueError: boom" in json.loads(JsonFormatter().format(record))["exc"]
