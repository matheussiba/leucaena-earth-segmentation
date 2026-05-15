#!/usr/bin/env bash
# Build (if needed) and open an interactive shell in the segmentation container with GPU.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

echo "==> Building image (uses cache when unchanged)..."
docker compose build

echo "==> Starting shell (GPU via compose deploy.resources)..."
exec docker compose run --rm segmentation bash
