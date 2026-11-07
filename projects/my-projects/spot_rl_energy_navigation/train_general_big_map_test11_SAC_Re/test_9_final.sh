#!/usr/bin/env bash
# Same nine missions as test_9.sh, using the completed 280k policy.
set -euo pipefail
TEST11_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEST11_FINAL="$TEST11_DIR/runs/general_map_test11_sac_re_20261107_024047_872269/policy_final.pt"
[[ -f "$TEST11_FINAL" ]] || { echo "Checkpoint non trovato: $TEST11_FINAL" >&2; exit 1; }
exec bash "$TEST11_DIR/test_9.sh" --checkpoint "$TEST11_FINAL" "$@"
