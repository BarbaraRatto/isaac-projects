#!/usr/bin/env bash
# Run an experiment launcher, or any Isaac Lab Python script, in the cloud env.
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == --help || $# == 0 ]]; then
    echo "Usage: bash cloud/run.sh EXPERIMENT [training arguments...]"
    echo "       bash cloud/run.sh --python PATH_TO_SCRIPT [arguments...]"
    exit 0
fi
[[ -f "$HERE/env.sh" ]] || { echo "Run bash $HERE/setup.sh first." >&2; exit 1; }
# shellcheck disable=SC1091
source "$HERE/env.sh"
if [[ "$1" == --python ]]; then
    shift
    [[ $# -gt 0 ]] || { echo 'Missing Python script.' >&2; exit 1; }
    exec "$GENERAL_MAP_ISAACLAB" -p "$@"
fi
experiment="$1"
shift
[[ "$experiment" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo 'Invalid experiment name.' >&2; exit 1; }
launcher="$HERE/../$experiment/train.sh"
[[ -f "$launcher" ]] || { echo "Launcher missing: $launcher" >&2; exit 1; }
exec bash "$launcher" "$@"
