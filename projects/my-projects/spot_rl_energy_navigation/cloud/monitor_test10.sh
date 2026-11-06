#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -d "$PROJECT/spot_rl_energy_navigation/cloud" ]]; then
    PROJECT="$(cd "$PROJECT/../.." && pwd)"
fi
if [[ $# != 1 || "${1:-}" == --help ]]; then
    echo 'Usage: bash monitor_test10.sh PERSISTENT_RUNS_ROOT'
    echo 'Example: watch -n 10 bash monitor_test10.sh /persistent/test10-runs'
    exit 0
fi
[[ -f "$1/latest_run.txt" ]] || { echo 'No run started with start_test10.sh in this directory.'; exit 0; }
RUN_DIR="$(cat "$1/latest_run.txt")"
exec python3 "$PROJECT/spot_rl_energy_navigation/train_general_big_map_test10_SAC/status.py" --run-dir "$RUN_DIR"
