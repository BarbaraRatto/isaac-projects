#!/usr/bin/env bash
# Frozen map/PCA snapshots are already included; no terrain generation or PCA fit.
set -euo pipefail
GENERAL_MAP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GENERAL_MAP_ISAACLAB="${GENERAL_MAP_ISAACLAB:-/home/isaac/isaaclab/2.3.0/isaaclab.sh}"
exec "$GENERAL_MAP_ISAACLAB" -p "$GENERAL_MAP_DIR/train.py" \
    --headless --device cuda:0 --agent-device cpu --timesteps 600000 "$@"
