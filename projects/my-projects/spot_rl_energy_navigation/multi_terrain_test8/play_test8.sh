#!/usr/bin/env bash
# Test8 completed run: policy_final.zip, 20 episodes on each known map, GUI.
# Run with bash; extra arguments are forwarded to test_known_maps.py.
# Examples: --episodes 5, --headless, --dry-run, --checkpoint /path/policy_best.zip
set -euo pipefail

TEST8_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEST8_CHECKPOINT="$TEST8_DIR/runs/test8_sac_32env_20261020_152756_792002/policy_final.zip"

exec python3 "$TEST8_DIR/test_known_maps.py" \
    --checkpoint "$TEST8_CHECKPOINT" \
    --episodes 20 \
    "$@"
