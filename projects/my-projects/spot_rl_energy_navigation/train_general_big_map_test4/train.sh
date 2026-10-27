#!/usr/bin/env bash
# Calibrate on the frozen big map once, then start training in a fresh process.
set -euo pipefail
GENERAL_MAP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${GENERAL_MAP_ISAACLAB:-}" && -f "$GENERAL_MAP_DIR/../cloud/env.sh" ]]; then
    # shellcheck disable=SC1091
    source "$GENERAL_MAP_DIR/../cloud/env.sh"
fi
GENERAL_MAP_ISAACLAB="${GENERAL_MAP_ISAACLAB:-/home/isaac/isaaclab/2.3.0/isaaclab.sh}"
GENERAL_MAP_PREP_ARGS=(--headless --device cuda:0 --reuse)
GENERAL_MAP_SKIP_PREP=0
# Forward shared simulator/DINO options, preserving the original training CLI.
GENERAL_MAP_ARGS=("$@")
for ((i=0; i<${#GENERAL_MAP_ARGS[@]}; i++)); do
    arg="${GENERAL_MAP_ARGS[i]}"
    case "$arg" in
        --resume|--resume=*|--help|-h) GENERAL_MAP_SKIP_PREP=1 ;;
        --pca)
            ((i+=1))
            GENERAL_MAP_PREP_ARGS+=(--output "${GENERAL_MAP_ARGS[i]:?Missing --pca value}") ;;
        --pca=*) GENERAL_MAP_PREP_ARGS+=(--output "${arg#*=}") ;;
        --device|--dino-model|--dino-batch-size|--torch-threads|--rendering_mode|--kit_args)
            ((i+=1))
            GENERAL_MAP_PREP_ARGS+=("$arg" "${GENERAL_MAP_ARGS[i]:?Missing option value}") ;;
        --device=*|--dino-model=*|--dino-batch-size=*|--torch-threads=*|--rendering_mode=*|--kit_args=*)
            GENERAL_MAP_PREP_ARGS+=("$arg") ;;
    esac
done
if [[ "$GENERAL_MAP_SKIP_PREP" == 0 ]]; then
    echo '[TEST4] Checking/calibrating PCA on all seven terrains before RL training.'
    "$GENERAL_MAP_ISAACLAB" -p "$GENERAL_MAP_DIR/prepare_pca.py" "${GENERAL_MAP_PREP_ARGS[@]}"
fi
echo '[TEST4] Starting RL training.'
exec "$GENERAL_MAP_ISAACLAB" -p "$GENERAL_MAP_DIR/train.py" \
    --headless --device cuda:0 --agent-device cpu --timesteps 300000 "$@"
