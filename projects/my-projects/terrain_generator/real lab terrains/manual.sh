#!/usr/bin/env bash
set -euo pipefail
LAB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$LAB_DIR/../.." && pwd)"
LAB_MATERIAL="${1:-comparison}"
if [[ $# -gt 0 ]]; then shift; fi
case "$LAB_MATERIAL" in
    grey|green) LAB_MAP="${LAB_MATERIAL}_flat.usda"; LAB_Y=0 ;;
    comparison) LAB_MAP=comparison.usda; LAB_Y=-1.5 ;;
    *) echo 'Usage: bash manual.sh [grey|green|comparison] [--speed 0.9 ...]' >&2; exit 2 ;;
esac
LAB_WRAPPER="${GENERAL_MAP_ISAACLAB:-/home/isaac/isaaclab/2.3.0/isaaclab.sh}"
[[ -f "$LAB_DIR/maps/$LAB_MAP" ]] || { echo 'Generate terrains with create_terrains.py first.' >&2; exit 1; }
exec "$LAB_WRAPPER" -p "$PROJECT/spot_rl_energy_navigation/manual_terrain.py" \
    --terrain "$LAB_DIR/maps/$LAB_MAP" --surface-drag-manifest "$LAB_DIR/maps/manifest.json" \
    --spawn 0 "$LAB_Y" .75 --device cuda:0 --speed .9 "$@"
