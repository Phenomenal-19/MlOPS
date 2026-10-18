#!/usr/bin/env bash
# Remove the local kind cluster created by k8s_deploy.sh.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
kind delete cluster --name "${CLUSTER:-heart-mlops}"
