#!/usr/bin/env bash
# One-command runner for a Linux machine / virtual lab (no copy-paste needed).
#
#   bash scripts/lab_run.sh          # ML pipeline + tests + local API check   (Python only)
#   bash scripts/lab_run.sh docker   # build and test the Docker image       (needs Docker)
#   bash scripts/lab_run.sh k8s      # Kubernetes + monitoring (kind)        (needs Docker)
#   bash scripts/lab_run.sh all      # everything above
set -euo pipefail

STAGE="${1:-ml}"
cd "$(dirname "$0")/.."
VENV=".venv"
API_PID=""

say() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mERROR: %s\033[0m\n' "$*" >&2; exit 1; }
cleanup() { [ -n "$API_PID" ] && kill "$API_PID" 2>/dev/null || true; }
trap cleanup EXIT

is_py312() { "$1" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)' 2>/dev/null; }

setup_env() {
  say "Python 3.12 environment ($VENV)"
  if [ -x "$VENV/bin/python" ] && is_py312 "$VENV/bin/python"; then
    echo "reusing $VENV"
  else
    rm -rf "$VENV"
    local found=""
    for c in python3.12 python3 python; do
      if command -v "$c" >/dev/null 2>&1 && is_py312 "$c"; then found="$c"; break; fi
    done
    if [ -n "$found" ] && "$found" -m venv "$VENV" >/dev/null 2>&1; then
      echo "created venv with $found"
    elif [ -n "$found" ] && rm -rf "$VENV" && "$found" -m venv --without-pip "$VENV" >/dev/null 2>&1 \
         && curl -fsSL https://bootstrap.pypa.io/get-pip.py | "$VENV/bin/python" - -q; then
      echo "created venv with $found (pip bootstrapped; the venv module lacked ensurepip)"
    elif command -v conda >/dev/null 2>&1; then
      rm -rf "$VENV"
      echo "creating a conda environment with Python 3.12 (a few minutes)"
      conda create -y -q -p "$VENV" python=3.12
    else
      die "No usable Python 3.12 (venv or conda) found."
    fi
  fi
  PY="$VENV/bin/python"
  "$PY" --version
  "$PY" -m pip install -q --upgrade pip
  say "Installing requirements (2-4 minutes)"
  "$PY" -m pip install -q -r requirements-dev.txt
}

run_ml() {
  export MLFLOW_DISABLE_AGENT_HINT=1 PYTHONWARNINGS=ignore
  say "1/7 Dataset (download if the network allows, else use the committed cleaned CSV)"
  if "$PY" -m src.data.download; then
    "$PY" -m src.data.preprocess
  else
    echo "download failed - using data/processed/heart_clean.csv from the repo"
  fi
  say "2/7 EDA figures"
  "$PY" -m src.eda.run_eda
  say "3/7 Train, tune and track three models in MLflow (1-2 minutes)"
  "$PY" -m src.models.train
  say "4/7 Package the best model"
  "$PY" -m src.models.package
  "$PY" -m src.models.quality_gate
  say "5/7 Sample prediction"
  "$PY" -m src.predict --input sample_input.json
  say "6/7 Lint and tests (1-2 minutes)"
  "$VENV/bin/ruff" check .
  "$PY" -m pytest -q
  say "7/7 Local API check"
  "$PY" -m uvicorn src.api.app:app --port 8000 >/tmp/heart_api.log 2>&1 &
  API_PID=$!
  for _ in $(seq 1 30); do curl -sf localhost:8000/health >/dev/null && break; sleep 1; done
  curl -sf localhost:8000/health || { cat /tmp/heart_api.log; die "API did not start"; }
  echo
  curl -sf -X POST localhost:8000/predict -H "Content-Type: application/json" -d @sample_input.json
  echo
  kill "$API_PID" 2>/dev/null || true
  API_PID=""
  say "ML PIPELINE, TESTS AND API VERIFIED"
  echo "To browse MLflow:  $VENV/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000"
}

run_docker() {
  docker info >/dev/null 2>&1 || die "Docker is not available to this user (try 'docker ps')."
  bash scripts/docker_smoke_test.sh
}

run_k8s() {
  docker info >/dev/null 2>&1 || die "Docker is not available to this user (try 'docker ps')."
  export PATH="$HOME/.local/bin:$PATH"
  bash scripts/k8s_deploy.sh
  bash scripts/k8s_monitoring.sh
}

case "$STAGE" in
  ml)     setup_env; run_ml ;;
  docker) run_docker ;;
  k8s)    run_k8s ;;
  all)    setup_env; run_ml; run_docker; run_k8s ;;
  *)      die "Unknown stage '$STAGE'. Use: ml | docker | k8s | all" ;;
esac
