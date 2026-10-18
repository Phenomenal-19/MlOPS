#!/usr/bin/env bash
# One-time setup of Docker Engine inside WSL (Ubuntu). Run with:  bash scripts/wsl_setup.sh
# It asks for your Ubuntu password once (sudo).
set -euo pipefail

echo "==> Installing Docker Engine"
sudo apt-get update
sudo apt-get install -y docker.io curl

echo "==> Letting user '$USER' run docker without sudo"
sudo usermod -aG docker "$USER"

echo "==> Starting the Docker daemon"
if [ -d /run/systemd/system ]; then
  sudo systemctl enable --now docker
else
  sudo service docker start
fi

sleep 2
sudo docker --version
sudo docker run --rm hello-world | head -n 5
echo
echo "Docker is ready. If a later 'docker' command says permission denied,"
echo "close this WSL window and open a new one (group membership refreshes)."
