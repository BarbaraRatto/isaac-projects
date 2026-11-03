#!/usr/bin/env bash
# Validate prepared assets; do not regenerate the map or refit PCA.
set -euo pipefail
GENERAL_MAP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${GENERAL_MAP_ISAACLAB:-}" && -f "$GENERAL_MAP_DIR/../cloud/env.sh" ]]; then
    source "$GENERAL_MAP_DIR/../cloud/env.sh"
fi
GENERAL_MAP_ISAACLAB="${GENERAL_MAP_ISAACLAB:-/home/isaac/isaaclab/2.3.0/isaaclab.sh}"
GENERAL_MAP_CHECK_ARGS=()
GENERAL_MAP_ARGS=("$@")
GENERAL_MAP_SKIP_CHECK=0
for ((i=0; i<${#GENERAL_MAP_ARGS[@]}; i++)); do
    arg="${GENERAL_MAP_ARGS[i]}"
    case "$arg" in
        --resume|--resume=*|--help|-h) GENERAL_MAP_SKIP_CHECK=1 ;;
        --pca|--dino-model)
            ((i+=1))
            GENERAL_MAP_CHECK_ARGS+=("$arg" "${GENERAL_MAP_ARGS[i]:?Missing option value}") ;;
        --pca=*|--dino-model=*) GENERAL_MAP_CHECK_ARGS+=("$arg") ;;
    esac
done
if [[ "$GENERAL_MAP_SKIP_CHECK" == 0 ]]; then
    "$GENERAL_MAP_ISAACLAB" -p "$GENERAL_MAP_DIR/check_setup.py" "${GENERAL_MAP_CHECK_ARGS[@]}"
fi
echo '[TEST9_SAC] SAC from scratch unless --resume is supplied. Use --run-dir on persistent cloud storage.'
exec "$GENERAL_MAP_ISAACLAB" -p "$GENERAL_MAP_DIR/train.py" \
    --headless --device cuda:0 --agent-device cpu "$@"
