#!/usr/bin/env bash
# First fit PCA if missing, then train on the two small mirrored maps.
set -euo pipefail
TEST10_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEST10_ISAACLAB="${TEST10_ISAACLAB:-/home/isaac/isaaclab/2.3.0/isaaclab.sh}"
TEST10_PCA="$TEST10_DIR/artifacts/pca32.npz"
TEST10_DEVICE="cuda:0"
TEST10_DINO_ARGS=()
TEST10_RESUME=0
TEST10_ARGS=("$@")
for ((i=0; i<${#TEST10_ARGS[@]}; i++)); do
    case "${TEST10_ARGS[i]}" in
        --resume|--resume=*) TEST10_RESUME=1 ;;
        --pca) TEST10_PCA="${TEST10_ARGS[i+1]:?Missing --pca value}" ;;
        --pca=*) TEST10_PCA="${TEST10_ARGS[i]#*=}" ;;
        --device) TEST10_DEVICE="${TEST10_ARGS[i+1]:?Missing --device value}" ;;
        --device=*) TEST10_DEVICE="${TEST10_ARGS[i]#*=}" ;;
        --dino-model) TEST10_DINO_ARGS=(--dino-model "${TEST10_ARGS[i+1]:?Missing --dino-model value}") ;;
        --dino-model=*) TEST10_DINO_ARGS=(--dino-model "${TEST10_ARGS[i]#*=}") ;;
    esac
done
if [[ "$TEST10_RESUME" == 0 && ! -f "$TEST10_PCA" ]]; then
    "$TEST10_ISAACLAB" -p "$TEST10_DIR/prepare_pca.py" --headless --device "$TEST10_DEVICE" \
        --output "$TEST10_PCA" "${TEST10_DINO_ARGS[@]}"
fi
exec "$TEST10_ISAACLAB" -p "$TEST10_DIR/train.py" \
    --headless --device "$TEST10_DEVICE" --agent-device cpu --pca "$TEST10_PCA" --timesteps 500000 "$@"
