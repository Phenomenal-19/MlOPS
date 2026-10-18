# Model-serving image for the heart-disease API.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first so this layer is cached until requirements change.
COPY requirements-serve.txt .
RUN pip install -r requirements-serve.txt

# Only what inference needs: no training code, MLflow or raw data.
COPY src/__init__.py src/__init__.py
COPY src/data/__init__.py src/data/__init__.py
COPY src/data/preprocess.py src/data/preprocess.py
COPY src/features.py src/predict.py src/
COPY src/api src/api
COPY models/model.joblib models/metadata.json models/

RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

ENV MODEL_PATH=/app/models/model.joblib \
    LOG_LEVEL=INFO

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
