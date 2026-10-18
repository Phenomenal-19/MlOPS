#!/usr/bin/env bash
# Build the serving image, run it, and verify /health and /predict.
# Works in WSL, on Linux and in CI. Exits non-zero on any failure.
set -euo pipefail

IMAGE="${IMAGE:-heart-api:latest}"
NAME="${NAME:-heart-api-smoke}"
PORT="${PORT:-8000}"
cd "$(dirname "$0")/.."

DOCKER=docker
$DOCKER info >/dev/null 2>&1 || DOCKER="sudo docker"

cleanup() { $DOCKER rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT
cleanup

echo "==> Building image $IMAGE"
$DOCKER build -t "$IMAGE" .

echo "==> Image size"
$DOCKER images "$IMAGE" --format '{{.Repository}}:{{.Tag}}  {{.Size}}'

echo "==> Starting container on port $PORT"
$DOCKER run -d --name "$NAME" -p "$PORT:8000" "$IMAGE" >/dev/null

echo "==> Waiting for /health"
for _ in $(seq 1 30); do
  if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; then break; fi
  sleep 1
done
curl -sf "http://127.0.0.1:$PORT/health" || { echo "Health check failed"; $DOCKER logs "$NAME"; exit 1; }
echo

echo "==> POST /predict with sample_input.json"
RESPONSE=$(curl -sf -X POST "http://127.0.0.1:$PORT/predict" \
  -H "Content-Type: application/json" -d @sample_input.json)
echo "$RESPONSE"
python3 - "$RESPONSE" <<'PY'
import json, sys
r = json.loads(sys.argv[1])
assert set(r) >= {"prediction", "confidence", "label"}, r
assert r["prediction"] in (0, 1), r
assert 0.5 <= r["confidence"] <= 1.0, r
print("response schema OK")
PY

echo "==> Invalid input must be rejected with 422"
CODE=$(curl -s -o /dev/null -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/predict" \
  -H "Content-Type: application/json" -d '{"age": 500}')
[ "$CODE" = "422" ] || { echo "Expected 422, got $CODE"; exit 1; }
echo "got 422 as expected"

echo "==> Container runs as non-root"
USER_ID=$($DOCKER exec "$NAME" id -u)
[ "$USER_ID" != "0" ] || { echo "container runs as root"; exit 1; }
echo "uid=$USER_ID"

echo "==> Container logs"
$DOCKER logs "$NAME" 2>&1 | tail -n 15

echo "==> ALL DOCKER CHECKS PASSED"
