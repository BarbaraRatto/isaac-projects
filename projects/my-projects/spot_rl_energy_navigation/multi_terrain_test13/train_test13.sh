#!/usr/bin/env bash
# Frozen map/PCA snapshots are already included; no terrain generation or PCA fit.
set -euo pipefail
TEST13_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEST13_ISAACLAB="${TEST13_ISAACLAB:-/home/isaac/isaaclab/2.3.0/isaaclab.sh}"
exec "$TEST13_ISAACLAB" -p "$TEST13_DIR/train.py" \
    --headless --device cuda:0 --agent-device cpu --timesteps 100000 "$@"
