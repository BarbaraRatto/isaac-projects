#!/usr/bin/env bash
# Visible evaluation of the completed 300k SAC continuation on stage 1.
# test.sh supplies one robot, nine episodes and seed 20000.
set -euo pipefail
TEST6_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$TEST6_DIR/test.sh" \
  --checkpoint "$TEST6_DIR/runs/general_map_test6_sac_20261029_001023_953670/policy_final.pt" \
  --stage 1 "$@"
