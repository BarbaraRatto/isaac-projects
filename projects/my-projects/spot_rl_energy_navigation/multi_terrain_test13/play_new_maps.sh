#!/usr/bin/env bash
# Evaluate each new board in its own simulator process, one after another.
set -euo pipefail
TEST13_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEST13_ISAACLAB="${TEST13_ISAACLAB:-/home/isaac/isaaclab/2.3.0/isaaclab.sh}"
for arg in "$@"; do
    case "$arg" in
        --layout|--layout=*|--output|--output=*)
            echo 'Il launcher seleziona le tre mappe e crea risultati separati: omettere --layout e --output.' >&2
            exit 2 ;;
    esac
done
for layout in new_map_1 new_map_2 new_map_3; do
    echo "[TEST13] Valutazione sequenziale: $layout"
    "$TEST13_ISAACLAB" -p "$TEST13_DIR/play.py" \
        --device cuda:0 --agent-device cpu --layout "$layout" "$@"
done
