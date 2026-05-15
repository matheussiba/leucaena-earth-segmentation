#!/usr/bin/env bash
# Optional: mount Google Drive (G:) in WSL. Not needed if data is on C: (see .env.example).
# Run once per WSL session:  bash scripts/wsl-mount-gdrive.sh

set -euo pipefail

MNT=/mnt/g
if mountpoint -q "$MNT" 2>/dev/null; then
  echo "Already mounted: $MNT"
else
  sudo mkdir -p "$MNT"
  sudo mount -t drvfs 'G:' "$MNT"
  echo "Mounted G: at $MNT"
fi
