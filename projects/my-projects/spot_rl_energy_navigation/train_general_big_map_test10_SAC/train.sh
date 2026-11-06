#!/usr/bin/env bash
# One command: immutable assets -> one-time final-cell calibration -> SAC.
set -euo pipefail
TEST10_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${GENERAL_MAP_ISAACLAB:-}" && -f "$TEST10_DIR/../cloud/env.sh" ]]; then
    source "$TEST10_DIR/../cloud/env.sh"
fi
GENERAL_MAP_ISAACLAB="${GENERAL_MAP_ISAACLAB:-/home/isaac/isaaclab/2.3.0/isaaclab.sh}"
TEST10_ARGS=("$@")
TEST10_RESUME=0
TEST10_HELP=0
TEST10_DEVICE=cuda:0
TEST10_CHECK=()
for ((i=0; i<${#TEST10_ARGS[@]}; i++)); do
    arg="${TEST10_ARGS[i]}"
    case "$arg" in
        --help|-h) TEST10_HELP=1 ;;
        --resume|--resume=*) TEST10_RESUME=1 ;;
        --device) ((i+=1)); TEST10_DEVICE="${TEST10_ARGS[i]:?Missing device}" ;;
        --device=*) TEST10_DEVICE="${arg#*=}" ;;
        --pca|--dino-model)
            ((i+=1)); TEST10_CHECK+=("$arg" "${TEST10_ARGS[i]:?Missing option value}") ;;
        --pca=*|--dino-model=*) TEST10_CHECK+=("$arg") ;;
    esac
done
if [[ "$TEST10_HELP" == 0 ]]; then
    "$GENERAL_MAP_ISAACLAB" -p "$TEST10_DIR/prepare_assets.py"
    if [[ "$TEST10_RESUME" == 0 ]]; then
        if [[ ! -f "$TEST10_DIR/artifacts/frozen_scalers.json" ]]; then
            echo '[TEST10] First launch: RGBD FINAL-CELL scaler calibration, no RL/PCA fitting.'
            flock -n "$TEST10_DIR/artifacts/calibration.lock" \
                "$GENERAL_MAP_ISAACLAB" -p "$TEST10_DIR/calibrate_scalers.py" \
                --headless --device "$TEST10_DEVICE" \
                --num-envs "${TEST10_CALIB_ENVS:-32}" --dino-batch-size 32 "${TEST10_CHECK[@]}"
        fi
        "$GENERAL_MAP_ISAACLAB" -p "$TEST10_DIR/check_setup.py" "${TEST10_CHECK[@]}"
    fi
fi
exec "$GENERAL_MAP_ISAACLAB" -p "$TEST10_DIR/train.py" \
    --headless --device cuda:0 --agent-device cuda:0 --map-backend cuda \
    --rgb-backend cuda --replay-device cuda:0 --dino-batch-size 128 \
    "${TEST10_ARGS[@]}"
