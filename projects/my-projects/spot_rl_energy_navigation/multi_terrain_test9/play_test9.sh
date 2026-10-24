#!/usr/bin/env bash
# Policy finale del training test9: 100.000 transizioni, peso energia 0,03.
# 5 episodi per mappa, GUI; inoltra opzioni come --dry-run o --checkpoint.
set -euo pipefail

TEST9_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEST9_CHECKPOINT="$TEST9_DIR/runs/test9_sac_32env_20261021_152607_024312/policy_final.zip"

exec python3 "$TEST9_DIR/test_known_maps.py" \
    --checkpoint "$TEST9_CHECKPOINT" \
    --episodes 5 \
    "$@"
