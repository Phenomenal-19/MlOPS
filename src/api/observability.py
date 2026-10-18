"""Prometheus metrics and logging setup for the API."""
import json
import logging
import os

from prometheus_client import Counter, Gauge, Histogram

# Label values are bounded on purpose (route templates, small enums) so the number
# of time series cannot grow with traffic.
HTTP_REQUESTS = Counter(
    "api_http_requests_total", "HTTP requests handled", ["method", "path", "status"])
HTTP_LATENCY = Histogram(
    "api_http_request_duration_seconds", "HTTP request latency in seconds", ["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0))
PREDICTIONS = Counter(
    "api_predictions_total", "Predictions served, by predicted class", ["label"])
CONFIDENCE = Histogram(
    "api_prediction_confidence", "Confidence (probability of the predicted class)",
    buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0))
MODEL_INFO = Gauge("api_model_info", "Loaded model (value is always 1)", ["model"])

# Endpoints that are polled constantly; their access logs go to DEBUG so real
# traffic stays visible, and /metrics scrapes do not count as API traffic.
QUIET_PATHS = {"/health", "/metrics"}
UNMATCHED = "unmatched"

_EXTRA_FIELDS = ("request_id", "method", "path", "status", "latency_ms", "prediction",
                 "confidence")


class JsonFormatter(logging.Formatter):
    """One JSON object per line, ready for log collectors."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update({k: getattr(record, k) for k in _EXTRA_FIELDS if hasattr(record, k)})
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging() -> None:
    handler = logging.StreamHandler()
    if os.getenv("LOG_FORMAT", "text").lower() == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), handlers=[handler], force=True)
