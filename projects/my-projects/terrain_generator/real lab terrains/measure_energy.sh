#!/usr/bin/env bash
set -euo pipefail
LAB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAB_WRAPPER="${GENERAL_MAP_ISAACLAB:-/home/isaac/isaaclab/2.3.0/isaaclab.sh}"
exec "$LAB_WRAPPER" -p "$LAB_DIR/measure_energy.py" --device cuda:0 "$@"
