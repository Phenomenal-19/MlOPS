#!/usr/bin/env bash
# Deploy the heart-disease API to a local kind cluster and verify it end to end.
# Needs Docker. Installs kind/kubectl (no sudo) if they are missing.
# Result: http://localhost:8080 -> ingress-nginx -> Service -> 2 API pods.
set -euo pipefail

CLUSTER="${CLUSTER:-heart-mlops}"
IMAGE="${IMAGE:-heart-api:latest}"
NS="heart-mlops"
BASE="http://127.0.0.1:8080"
INGRESS_MANIFEST="https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.12.1/deploy/static/provider/kind/deploy.yaml"

cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

diagnostics() {
  echo "!!! Deployment failed - cluster state follows" >&2
  kubectl get pods,svc,ingress -A 2>&1 | tail -n 40 >&2 || true
  kubectl -n "$NS" describe pods 2>&1 | tail -n 60 >&2 || true
  kubectl -n "$NS" logs -l app=heart-api --tail=30 >&2 2>&1 || true
}
trap diagnostics ERR

docker info >/dev/null 2>&1 || { echo "Docker is not usable for this user" >&2; exit 1; }
command -v kind >/dev/null && command -v kubectl >/dev/null || bash scripts/k8s_setup.sh

echo "==> Image $IMAGE"
docker build -q -t "$IMAGE" .   # cached layers make this fast; picks up code changes

echo "==> Cluster '$CLUSTER'"
if kind get clusters 2>/dev/null | grep -qx "$CLUSTER"; then
  echo "already exists"
else
  kind create cluster --name "$CLUSTER" --config k8s/kind/cluster.yaml --wait 120s
fi
kubectl config use-context "kind-$CLUSTER" >/dev/null

echo "==> Loading image into the cluster"
kind load docker-image "$IMAGE" --name "$CLUSTER"

echo "==> Ingress controller (ingress-nginx)"
if ! kubectl get ns ingress-nginx >/dev/null 2>&1; then
  kubectl apply -f "$INGRESS_MANIFEST"
fi
kubectl -n ingress-nginx wait --for=condition=ready pod \
  -l app.kubernetes.io/component=controller --timeout=240s

echo "==> Applying manifests"
EXISTED=false
kubectl -n "$NS" get deployment heart-api >/dev/null 2>&1 && EXISTED=true
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml -f k8s/ingress.yaml
# On a re-run the manifest is unchanged, so restart to pick up a freshly loaded image.
if [ "$EXISTED" = true ]; then
  kubectl -n "$NS" rollout restart deployment/heart-api >/dev/null
fi
kubectl -n "$NS" rollout status deployment/heart-api --timeout=180s
kubectl -n "$NS" wait --for=condition=ready pod -l app=heart-api --timeout=120s

echo "==> Verifying through the ingress at $BASE"
for _ in $(seq 1 30); do
  curl -sf "$BASE/health" >/dev/null && break
  sleep 2
done
curl -sf "$BASE/health"; echo
RESPONSE=$(curl -sf -X POST "$BASE/predict" -H "Content-Type: application/json" -d @sample_input.json)
echo "$RESPONSE"
python3 - "$RESPONSE" <<'PY'
import json, sys
r = json.loads(sys.argv[1])
assert r["prediction"] in (0, 1) and 0.5 <= r["confidence"] <= 1.0, r
print("prediction schema OK")
PY
CODE=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/predict" \
  -H "Content-Type: application/json" -d '{"age": 500}')
[ "$CODE" = "422" ] || { echo "Expected 422 for invalid input, got $CODE" >&2; exit 1; }
echo "invalid input rejected with 422"

echo "==> Sending 20 requests to show the Service spreading load across pods"
for _ in $(seq 1 20); do
  curl -sf -X POST "$BASE/predict" -H "Content-Type: application/json" \
    -d @sample_input.json >/dev/null
done

echo "==> Cluster state"
kubectl -n "$NS" get deployment,pods,svc,ingress -o wide

echo "==> POST /predict requests handled by each running pod"
for pod in $(kubectl -n "$NS" get pods -l app=heart-api \
    --field-selector=status.phase=Running -o name); do
  count=$(kubectl -n "$NS" logs "$pod" | grep -c "POST /predict" || true)
  echo "$pod: $count"
done

echo "==> KUBERNETES DEPLOYMENT VERIFIED"
