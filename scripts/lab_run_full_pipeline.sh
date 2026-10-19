#!/usr/bin/env bash
# Run the ENTIRE project on a Linux machine / virtual lab that has Docker - one command.
#
#   bash scripts/lab_run_full_pipeline.sh            # run everything (about 15-25 minutes)
#   bash scripts/lab_run_full_pipeline.sh teardown   # stop MLflow UI and remove the cluster
#   NO_UI=1 bash scripts/lab_run_full_pipeline.sh    # do not start the MLflow UI at the end
#
# Stages: preflight -> ML pipeline + lint + tests + API check -> Docker build and test
#         -> Kubernetes (kind + ingress + 2 replicas) -> Prometheus + Grafana -> traffic -> MLflow UI
# (CI/CD runs on GitHub when you push; see the Actions tab.)
# Everything is written to lab_full_run.log as well as the screen.
set -Eeuo pipefail

cd "$(dirname "$0")/.."
MODE="${1:-run}"
LOG="lab_full_run.log"
UI_PORT=5000
PIDFILE=".mlflow_ui.pid"
RESULTS=()
START=$SECONDS

blue() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
green() { printf '\033[1;32m%s\033[0m\n' "$*"; }
red() { printf '\033[1;31m%s\033[0m\n' "$*" >&2; }

stop_ui() {
  if [ -f "$PIDFILE" ]; then
    kill "$(cat "$PIDFILE")" 2>/dev/null || true
    rm -f "$PIDFILE"
    echo "MLflow UI stopped"
  fi
}

if [ "$MODE" = "teardown" ]; then
  blue "Teardown"
  stop_ui
  bash scripts/k8s_teardown.sh || true
  docker rm -f heart-api heart-api-smoke >/dev/null 2>&1 || true
  green "Teardown finished: MLflow UI stopped and the cluster removed if it existed. (.venv is kept.)"
  exit 0
fi
[ "$MODE" = "run" ] || { red "Unknown mode '$MODE'. Use: run | teardown"; exit 2; }

exec > >(tee -a "$LOG") 2>&1
echo "Full pipeline run started: $(date)"

summary() {
  blue "Summary ($(( (SECONDS - START) / 60 )) min $(( (SECONDS - START) % 60 )) s)"
  printf '  %s\n' "${RESULTS[@]}"
}

# step "title" command...   Runs the command; stops the whole run with a summary if it fails.
step() {
  local title="$1"; shift
  blue "$title"
  local t0=$SECONDS
  if "$@"; then
    RESULTS+=("PASS  $(( SECONDS - t0 ))s  $title")
  else
    RESULTS+=("FAIL  $(( SECONDS - t0 ))s  $title")
    summary
    red "Stopped at: $title   (full log: $LOG)"
    exit 1
  fi
}

port_in_use() { command -v ss >/dev/null 2>&1 && ss -ltn "sport = :$1" 2>/dev/null | grep -q LISTEN; }

preflight() {
  local bad=0
  docker info >/dev/null 2>&1 || { red "Docker is not usable by this user (try: docker ps)"; bad=1; }
  command -v curl >/dev/null 2>&1 || { red "curl is not installed"; bad=1; }
  for url in https://pypi.org/simple/pip/ \
             https://archive.ics.uci.edu/ \
             https://kind.sigs.k8s.io/dl/v0.27.0/kind-linux-amd64 \
             https://dl.k8s.io/release/v1.32.2/bin/linux/amd64/kubectl.sha256 \
             https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.12.1/deploy/static/provider/kind/deploy.yaml; do
    if curl -fsSL -m 20 -r 0-0 -o /dev/null "$url"; then
      echo "reachable: $url"
    else
      red "Cannot reach $url (network or proxy blocked?)"
      bad=1
    fi
  done
  for p in 8080 8000; do
    if port_in_use "$p"; then
      red "Port $p is already in use; free it and rerun (the cluster needs 8080, the API check 8000)"
      bad=1
    fi
  done
  if port_in_use "$UI_PORT"; then echo "note: port $UI_PORT is busy, so the MLflow UI will not be started"; fi
  local free_gb
  free_gb=$(df -Pk . | awk 'NR==2 {printf "%d", $4 / 1048576}')
  if [ "$free_gb" -lt 6 ]; then echo "warning: only ${free_gb} GB free; the images need about 5 GB"; fi
  return "$bad"
}

cluster_status() {
  export PATH="$HOME/.local/bin:$PATH"
  kubectl -n heart-mlops get deployment,pods,svc,ingress
  kubectl -n monitoring get pods,svc,ingress
}

start_ui() {
  if port_in_use "$UI_PORT" || [ "${NO_UI:-0}" = "1" ]; then
    echo "MLflow UI not started"
    return 0
  fi
  stop_ui
  nohup .venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 \
        --port "$UI_PORT" >/tmp/mlflow_ui.log 2>&1 &
  echo $! > "$PIDFILE"
  for _ in $(seq 1 30); do curl -sf "http://127.0.0.1:$UI_PORT/health" >/dev/null && break; sleep 1; done
  curl -sf "http://127.0.0.1:$UI_PORT/health" >/dev/null && echo "MLflow UI running on port $UI_PORT"
}

step "1/8 Preflight checks (Docker, network, ports)" preflight
step "2/8 ML pipeline: data, EDA, training with MLflow, packaging, lint, 87 tests, API check" \
     bash scripts/lab_run.sh ml
step "3/8 Docker: build the image, run the container, verify the API" bash scripts/lab_run.sh docker
step "4/8 Kubernetes: kind cluster, ingress, 2 API replicas, verification" bash scripts/k8s_deploy.sh
step "5/8 Monitoring: Prometheus + Grafana, targets, dashboard, JSON logs" bash scripts/k8s_monitoring.sh
step "6/8 Extra traffic so the Grafana panels fill up" bash scripts/generate_traffic.sh 300 http://127.0.0.1:8080
step "7/8 Final cluster status" cluster_status
step "8/8 MLflow UI (experiment tracking)" start_ui

summary
green "
EVERYTHING VERIFIED. Open these in the lab's browser:
  http://localhost:8080/docs                 API through the ingress (POST /predict -> Try it out)
  http://grafana.localhost:8080              Grafana dashboard 'Heart Disease API' (login admin / heart-grafana)
  http://prometheus.localhost:8080/targets   Prometheus targets (both API pods UP)
  http://127.0.0.1:$UI_PORT                       MLflow experiments (if started above)

Log of this run: $LOG
When finished:   bash scripts/lab_run_full_pipeline.sh teardown"
