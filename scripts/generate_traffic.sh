#!/usr/bin/env bash
# Send a realistic mix of requests so the dashboards have data.
# Usage: bash scripts/generate_traffic.sh [count] [base_url]
set -euo pipefail

COUNT="${1:-60}"
BASE="${2:-http://127.0.0.1:8080}"

LOW='{"age":45,"sex":0,"cp":2,"trestbps":118,"chol":210,"fbs":0,"restecg":0,"thalach":172,"exang":0,"oldpeak":0.0,"slope":1,"ca":0,"thal":3}'
HIGH='{"age":67,"sex":1,"cp":4,"trestbps":160,"chol":286,"fbs":0,"restecg":2,"thalach":108,"exang":1,"oldpeak":1.5,"slope":2,"ca":3,"thal":3}'
MID='{"age":58,"sex":1,"cp":3,"trestbps":140,"chol":245,"fbs":0,"restecg":1,"thalach":135,"exang":0,"oldpeak":1.2,"slope":2}'
BAD='{"age":500}'

echo "==> Sending $COUNT requests to $BASE"
for i in $(seq 1 "$COUNT"); do
  case $((i % 10)) in
    0)       body="$BAD" ;;     # 10% invalid -> 422
    1|2|3)   body="$LOW" ;;
    4|5|6|7) body="$HIGH" ;;
    *)       body="$MID" ;;
  esac
  curl -s -o /dev/null -X POST "$BASE/predict" -H "Content-Type: application/json" -d "$body"
  sleep 0.1
done
echo "==> Done"
