#!/usr/bin/env bash
set -euo pipefail
LAB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAB_USD_PYTHON="${LAB_USD_PYTHON:-/home/isaac/isaaclab/2.3.0/_isaac_sim/python.sh}"
exec "$LAB_USD_PYTHON" "$LAB_DIR/create_terrains.py" "$@"
