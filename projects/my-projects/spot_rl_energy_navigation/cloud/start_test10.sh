#!/usr/bin/env bash
# RUNS_ROOT must be on storage retained by your cloud provider.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -d "$PROJECT/spot_rl_energy_navigation/cloud" ]]; then
    PROJECT="$(cd "$PROJECT/../.." && pwd)"
fi
if [[ $# == 0 || "${1:-}" == --help ]]; then
    echo 'Usage: bash start_test10.sh PERSISTENT_RUNS_ROOT [training arguments...]'
    echo 'Fresh run: bash start_test10.sh /persistent/test10-runs'
    echo 'Resume:    bash start_test10.sh /persistent/test10-runs --resume /persistent/test10-runs/OLD_RUN'
    echo 'Default: 128 robots, DINO batch 128, simulator/RGB/DINO/maps/replay/SAC on CUDA.'
    echo 'Fresh launch calibrates fixed visual scalers first; resume uses saved scalers.'
    exit 0
fi
RUNS_ROOT="$1"
shift
for arg in "$@"; do
    case "$arg" in
        --run-dir|--run-dir=*) echo 'Use the first argument to choose storage; this script creates a NEW run directory.' >&2; exit 2 ;;
    esac
done
[[ -f "$PROJECT/spot_rl_energy_navigation/cloud/env.sh" ]] || {
    echo 'Run bash setup_test10.sh first.' >&2; exit 1;
}
mkdir -p "$RUNS_ROOT"
RUNS_ROOT="$(cd "$RUNS_ROOT" && pwd)"
RUN_DIR="$RUNS_ROOT/general_map_test10_sac_$(date +%Y%m%d_%H%M%S)_$$"
printf '%s\n' "$RUN_DIR" > "$RUNS_ROOT/.latest_run.$$.tmp"
mv "$RUNS_ROOT/.latest_run.$$.tmp" "$RUNS_ROOT/latest_run.txt"
echo "[TEST10] Run: $RUN_DIR"
echo '[TEST10] Keep the WHOLE run directory, especially recovery/. Storage persistence is provided by your cloud platform.'
exec bash "$PROJECT/spot_rl_energy_navigation/cloud/run.sh" train_general_big_map_test10_SAC \
    --run-dir "$RUN_DIR" "$@"
