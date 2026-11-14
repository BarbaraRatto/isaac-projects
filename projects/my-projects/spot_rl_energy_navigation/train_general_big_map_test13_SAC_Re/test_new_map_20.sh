#!/usr/bin/env bash
# Run the fixed twenty-mission suite on the new evaluation map.
set -euo pipefail
LAB_EVAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAB_EVAL_LAUNCHER="${GENERAL_MAP_ISAACLAB:-/home/isaac/isaaclab/2.3.0/isaaclab.sh}"
if [[ "${TERM:-dumb}" == "dumb" ]]; then export TERM=xterm; fi
exec "$LAB_EVAL_LAUNCHER" -p "$LAB_EVAL_DIR/new_map_eval/run.py" --device cuda:0 "$@"
