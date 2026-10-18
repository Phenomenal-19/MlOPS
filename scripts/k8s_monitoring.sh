#!/usr/bin/env bash
# Deploy Prometheus + Grafana next to the API and verify the whole monitoring path.
# Run scripts/k8s_deploy.sh first (it creates the cluster and the API).
#
#   Grafana     http://grafana.localhost:8080      (dashboard is the home page)
#   Prometheus  http://prometheus.localhost:8080
#   Grafana admin login (demo only): admin / heart-grafana
set -euo pipefail

BASE="http://127.0.0.1:8080"
GRAFANA_USER="admin"
GRAFANA_PASSWORD="heart-grafana"

cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"

diagnostics() {
  echo "!!! Monitoring setup failed - cluster state follows" >&2
  kubectl -n monitoring get pods,svc,ingress 2>&1 | tail -n 20 >&2 || true
  kubectl -n monitoring describe pods 2>&1 | tail -n 50 >&2 || true
  kubectl -n monitoring logs deploy/prometheus --tail=30 >&2 2>&1 || true
  kubectl -n monitoring logs deploy/grafana --tail=30 >&2 2>&1 || true
}
trap diagnostics ERR

kubectl -n heart-mlops get deployment heart-api >/dev/null 2>&1 \
  || { echo "heart-api is not deployed. Run: bash scripts/k8s_deploy.sh" >&2; exit 1; }

echo "==> Deploying Prometheus and Grafana"
kubectl apply -k k8s/monitoring
kubectl -n monitoring rollout status deployment/prometheus --timeout=240s
kubectl -n monitoring rollout status deployment/grafana --timeout=240s

prom() { curl -sf -H "Host: prometheus.localhost" "$BASE/api/v1/query" --data-urlencode "query=$1" -G; }

echo "==> Prometheus is discovering both API pods"
for _ in $(seq 1 40); do
  UP=$(prom 'count(up{job="heart-api"} == 1)' 2>/dev/null \
       | python3 -c "import sys,json; r=json.load(sys.stdin)['data']['result']; print(r[0]['value'][1] if r else 0)" 2>/dev/null || echo 0)
  [ "$UP" = "2" ] && break
  sleep 3
done
echo "heart-api targets up: $UP"
[ "$UP" = "2" ] || { echo "Expected 2 healthy targets, saw $UP" >&2; exit 1; }

echo "==> Generating traffic"
bash scripts/generate_traffic.sh 60 "$BASE"

echo "==> Waiting for Prometheus to scrape the new metrics"
TOTAL=0
for _ in $(seq 1 30); do
  TOTAL=$(prom 'sum(api_predictions_total)' \
          | python3 -c "import sys,json; r=json.load(sys.stdin)['data']['result']; print(int(float(r[0]['value'][1])) if r else 0)" 2>/dev/null || echo 0)
  [ "$TOTAL" -ge 40 ] && break
  sleep 3
done
echo "predictions recorded by Prometheus: $TOTAL"
[ "$TOTAL" -ge 40 ] || { echo "Expected at least 40 predictions, saw $TOTAL" >&2; exit 1; }

echo "==> Per-pod request rate (shows load balancing)"
prom 'sum by (pod) (api_http_requests_total{path="/predict"})' | python3 -c "
import sys, json
rows = json.load(sys.stdin)['data']['result']
for r in sorted(rows, key=lambda r: r['metric'].get('pod', '')):
    print(' ', r['metric'].get('pod', '?'), '->', int(float(r['value'][1])), 'requests')
assert len(rows) == 2, 'expected traffic on both pods'
"

echo "==> Grafana health, datasource and dashboard"
curl -sf -H "Host: grafana.localhost" "$BASE/api/health"; echo
curl -sf -u "$GRAFANA_USER:$GRAFANA_PASSWORD" -H "Host: grafana.localhost" \
  "$BASE/api/datasources/uid/prometheus/health"; echo
curl -sf -H "Host: grafana.localhost" "$BASE/api/search?query=Heart" | python3 -c "
import sys, json
found = json.load(sys.stdin)
assert any(d['uid'] == 'heart-api' for d in found), found
print('dashboard provisioned:', [d['title'] for d in found])
"

echo "==> Structured request logs from an API pod (JSON)"
kubectl -n heart-mlops logs deploy/heart-api --tail=200 | grep '"path": "/predict"' | tail -n 3

kubectl -n monitoring get pods,svc,ingress
echo
echo "Open  http://grafana.localhost:8080      (dashboard: Heart Disease API)"
echo "Open  http://prometheus.localhost:8080/targets"
echo "==> MONITORING VERIFIED"
