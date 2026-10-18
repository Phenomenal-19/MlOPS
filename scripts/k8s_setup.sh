#!/usr/bin/env bash
# Install kind and kubectl into ~/.local/bin (no sudo needed). Safe to re-run.
set -euo pipefail

KIND_VERSION="${KIND_VERSION:-v0.27.0}"
KUBECTL_VERSION="${KUBECTL_VERSION:-v1.32.2}"
BIN="$HOME/.local/bin"
mkdir -p "$BIN"
export PATH="$BIN:$PATH"

if ! command -v kind >/dev/null 2>&1; then
  echo "==> Installing kind $KIND_VERSION"
  curl -fsSL -o "$BIN/kind" "https://kind.sigs.k8s.io/dl/$KIND_VERSION/kind-linux-amd64"
  chmod +x "$BIN/kind"
fi

if ! command -v kubectl >/dev/null 2>&1; then
  echo "==> Installing kubectl $KUBECTL_VERSION"
  curl -fsSL -o "$BIN/kubectl" "https://dl.k8s.io/release/$KUBECTL_VERSION/bin/linux/amd64/kubectl"
  chmod +x "$BIN/kubectl"
fi

kind version
kubectl version --client
case ":$PATH:" in *":$BIN:"*) ;; *) echo "NOTE: add $BIN to PATH (e.g. in ~/.bashrc)";; esac
