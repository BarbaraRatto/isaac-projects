#!/usr/bin/env bash
# Separate simulator processes keep map changes and sensor teardown isolated.
set -euo pipefail
TEST11_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEST11_ISAACLAB="${TEST11_ISAACLAB:-/home/isaac/isaaclab/2.3.0/isaaclab.sh}"
for arg in "$@"; do
    case "$arg" in
        --layout|--layout=*|--output|--output=*)
            echo 'Non usare --layout o --output: il launcher seleziona le mappe e crea risultati separati.' >&2
            exit 2 ;;
    esac
done
for layout in rocks_asphalt__right_rocks rocks_asphalt__right_asphalt; do
    echo "[TEST11] Valutazione sequenziale: $layout"
    "$TEST11_ISAACLAB" -p "$TEST11_DIR/play.py" \
        --device cuda:0 --agent-device cpu --layout "$layout" "$@"
done
