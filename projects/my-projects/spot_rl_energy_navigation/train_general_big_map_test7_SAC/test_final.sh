#!/usr/bin/env bash
# Final 200k checkpoint; same stage-1 mission panel used for test6 comparisons.
set -euo pipefail
TEST7_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEST7_CHECKPOINT="$TEST7_DIR/runs/general_map_test7_sac_20261029_163507_402204/policy_final.pt"
[[ -f "$TEST7_CHECKPOINT" ]] || { echo "Missing checkpoint: $TEST7_CHECKPOINT" >&2; exit 1; }
exec bash "$TEST7_DIR/test.sh" \
    --checkpoint "$TEST7_CHECKPOINT" --stage 1 \
    --num-envs 1 --episodes 9 --seed 20000 "$@"
