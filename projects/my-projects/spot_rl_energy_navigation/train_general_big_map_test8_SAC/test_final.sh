#!/usr/bin/env bash
# Visible evaluation of the completed 600k run on the shared stage-1 panel.
set -euo pipefail
TEST8_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEST8_CHECKPOINT="$TEST8_DIR/runs/general_map_test8_sac_20261031_010201_266860/policy_final.pt"
[[ -f "$TEST8_CHECKPOINT" ]] || { echo "Missing checkpoint: $TEST8_CHECKPOINT" >&2; exit 1; }
exec bash "$TEST8_DIR/test.sh" \
    --checkpoint "$TEST8_CHECKPOINT" --stage 1 \
    --num-envs 1 --episodes 9 --seed 20000 "$@"
