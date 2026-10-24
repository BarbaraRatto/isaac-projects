#!/usr/bin/env bash
# Generate the separate 10 cm rocks variant, then open keyboard driving.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
isaaclab=/home/isaac/isaaclab/2.3.0/isaaclab.sh
usd_python=/home/isaac/isaaclab/2.3.0/_isaac_sim/python.sh
source_usd=terrain_generator/real_terrains_rocks_10cm.usd
boards=terrain_generator/manual_asphalt_rocks_10cm
regenerate=0
if [[ "${1:-}" == "--regenerate" ]]; then
    regenerate=1
    shift
fi

if [[ "$regenerate" == 1 || ! -f "$source_usd" || terrain_generator/create_real_terrains_rocks_10cm.py -nt "$source_usd" ]]; then
    echo '[TERRAIN] Generazione della variante aggiornata (senza finestra)...'
    "$isaaclab" -p terrain_generator/create_real_terrains_rocks_10cm.py
    regenerate=1
fi
if [[ "$regenerate" == 1 || ! -f "$boards/rocks_right.usd" || ! -f "$boards/rocks_left.usd" || terrain_generator/create_manual_asphalt_rocks.py -nt "$boards/rocks_right.usd" ]]; then
    echo '[TERRAIN] Creazione mappe piccole con bordi a quota zero...'
    "$usd_python" terrain_generator/create_manual_asphalt_rocks.py \
        --source "$source_usd" --output-dir "$boards" --overwrite \
        --rocks-edge-width 0.02 --rocks-flat-border 0.02
fi
echo '[TERRAIN] Apertura guida manuale...'
exec "$isaaclab" -p spot_rl_energy_navigation/manual_terrain.py \
    --device cuda:0 --speed 0.9 --terrain "$boards/rocks_right.usd" "$@"
