#!/usr/bin/env bash
# Frozen map/PCA snapshots are already included; no terrain generation or PCA fit.
set -euo pipefail
TEST12_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEST12_ISAACLAB="${TEST12_ISAACLAB:-/home/isaac/isaaclab/2.3.0/isaaclab.sh}"
exec "$TEST12_ISAACLAB" -p "$TEST12_DIR/train.py" \
    --headless --device cuda:0 --agent-device cpu --timesteps 500000 "$@"
