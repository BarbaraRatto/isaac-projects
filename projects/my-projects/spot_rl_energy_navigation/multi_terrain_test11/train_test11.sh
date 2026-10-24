#!/usr/bin/env bash
# Frozen map/PCA snapshots are already included; no terrain generation or PCA fit.
set -euo pipefail
TEST11_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEST11_ISAACLAB="${TEST11_ISAACLAB:-/home/isaac/isaaclab/2.3.0/isaaclab.sh}"
exec "$TEST11_ISAACLAB" -p "$TEST11_DIR/train.py" \
    --headless --device cuda:0 --agent-device cpu --timesteps 100000 "$@"
