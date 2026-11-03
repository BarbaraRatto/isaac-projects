#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -d "$PROJECT/spot_rl_energy_navigation/cloud" ]]; then
    PROJECT="$(cd "$PROJECT/../.." && pwd)"
fi
if [[ $# != 1 || "${1:-}" == --help ]]; then
    echo 'Usage: bash monitor_test9.sh PERSISTENT_RUNS_ROOT'
    echo 'Example: watch -n 10 bash monitor_test9.sh /persistent/test9-runs'
    exit 0
fi
[[ -f "$1/latest_run.txt" ]] || { echo 'No run started with start_test9.sh in this directory.'; exit 0; }
RUN_DIR="$(cat "$1/latest_run.txt")"
exec python3 "$PROJECT/spot_rl_energy_navigation/train_general_big_map_test9_SAC/status.py" --run-dir "$RUN_DIR"
