#!/usr/bin/env bash
# Run the paired straight-line baseline for the selected thesis missions.
set -euo pipefail
EXP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXP_LAUNCHER="${GENERAL_MAP_ISAACLAB:-/home/isaac/isaaclab/2.3.0/isaaclab.sh}"
if [[ "${TERM:-dumb}" == "dumb" ]]; then export TERM=xterm; fi
exec "$EXP_LAUNCHER" -p "$EXP_DIR/create_exp.py" --device cuda:0 "$@"
