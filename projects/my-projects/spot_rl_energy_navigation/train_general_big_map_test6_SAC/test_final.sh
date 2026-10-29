#!/usr/bin/env bash
# Evaluate the completed 100k run. Stage 0 is the stage actually trained.
# Pass --stage 1 to compare with the nine mixed-terrain missions of test5.
set -euo pipefail
TEST6_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$TEST6_DIR/test.sh" \
  --checkpoint "$TEST6_DIR/runs/general_map_test6_sac_20261028_201519_133017/policy_final.pt" \
  --stage 0 "$@"
