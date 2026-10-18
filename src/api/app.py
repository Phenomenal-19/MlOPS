"""FastAPI service exposing the packaged heart-disease model.

Endpoints
  GET  /health   liveness + whether the model is loaded
  GET  /         service / model info
  POST /predict  one patient (JSON) -> prediction, label, confidence, risk_probability
"""
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.predict import MODEL_PATH, load_model, predict

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("heart-api")

METADATA_PATH = Path(os.getenv("METADATA_PATH", Path(MODEL_PATH).with_name("metadata.json")))


class Patient(BaseModel):
    """The 13 Cleveland features. `ca` and `thal` may be omitted (imputed by the model)."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"example": {
            "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
            "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
            "ca": 0, "thal": 6}},
    )

    age: float = Field(ge=1, le=120, description="Age in years")
    sex: float = Field(description="1 = male, 0 = female")
    cp: float = Field(
        description="Chest pain: 1 typical, 2 atypical, 3 non-anginal, 4 asymptomatic")
    trestbps: float = Field(ge=50, le=260, description="Resting blood pressure (mm Hg)")
    chol: float = Field(ge=80, le=700, description="Serum cholesterol (mg/dl)")
    fbs: float = Field(description="Fasting blood sugar > 120 mg/dl (1 = true)")
    restecg: float = Field(description="Resting ECG result: 0, 1 or 2")
    thalach: float = Field(ge=40, le=250, description="Maximum heart rate achieved")
    exang: float = Field(description="Exercise-induced angina (1 = yes)")
    oldpeak: float = Field(ge=0, le=10, description="ST depression induced by exercise")
    slope: float = Field(description="Slope of the peak exercise ST segment: 1, 2 or 3")
    ca: float | None = Field(
        default=None, description="Major vessels coloured by fluoroscopy (0-4)")
    thal: float | None = Field(
        default=None, description="Thalassemia: 3 normal, 6 fixed, 7 reversible")

    @field_validator("sex", "fbs", "exang")
    @classmethod
    def _binary(cls, v: float) -> float:
        if v not in (0, 1):
            raise ValueError("must be 0 or 1")
        return v

    @field_validator("cp")
    @classmethod
    def _cp(cls, v: float) -> float:
        if v not in (1, 2, 3, 4):
            raise ValueError("must be 1, 2, 3 or 4")
        return v

    @field_validator("restecg")
    @classmethod
    def _restecg(cls, v: float) -> float:
        if v not in (0, 1, 2):
            raise ValueError("must be 0, 1 or 2")
        return v

    @field_validator("slope")
    @classmethod
    def _slope(cls, v: float) -> float:
        if v not in (1, 2, 3):
            raise ValueError("must be 1, 2 or 3")
        return v

    @field_validator("ca")
    @classmethod
    def _ca(cls, v: float | None) -> float | None:
        if v is not None and v not in (0, 1, 2, 3, 4):
            raise ValueError("must be 0-4")
        return v

    @field_validator("thal")
    @classmethod
    def _thal(cls, v: float | None) -> float | None:
        if v is not None and v not in (3, 6, 7):
            raise ValueError("must be 3, 6 or 7")
        return v


class Prediction(BaseModel):
    prediction: int = Field(description="1 = heart disease predicted, 0 = not predicted")
    label: str
    confidence: float = Field(description="Probability of the predicted class")
    risk_probability: float = Field(description="Probability of heart disease")
    model: str


def _model_name() -> str:
    try:
        return json.loads(METADATA_PATH.read_text())["model"]
    except (OSError, KeyError, ValueError):
        return "unknown"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model(Path(MODEL_PATH))
    app.state.model_name = _model_name()
    logger.info("Loaded model '%s' from %s", app.state.model_name, MODEL_PATH)
    yield


app = FastAPI(
    title="Heart Disease Risk API",
    description="Predicts the presence of heart disease from 13 clinical features.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = uuid.uuid4().hex[:8]
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("req=%s %s %s unhandled error", request_id, request.method,
                         request.url.path)
        raise
    ms = (time.perf_counter() - start) * 1000
    logger.info("req=%s %s %s status=%s latency_ms=%.1f", request_id, request.method,
                request.url.path, response.status_code, ms)
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok", "model_loaded": hasattr(request.app.state, "model")}


@app.get("/")
def info(request: Request) -> dict:
    return {"service": app.title, "version": app.version,
            "model": request.app.state.model_name, "docs": "/docs"}


@app.post("/predict", response_model=Prediction)
def predict_endpoint(patient: Patient, request: Request) -> Prediction:
    try:
        result = predict(request.app.state.model, [patient.model_dump()])[0]
    except Exception as exc:  # model/preprocessing failure, not a client error
        logger.exception("prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
    logger.info("prediction=%s confidence=%.3f", result["prediction"], result["confidence"])
    return Prediction(**result, model=request.app.state.model_name)
