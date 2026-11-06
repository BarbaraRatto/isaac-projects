#!/usr/bin/env bash
# Fifteen sequential visible episodes using the newest completed final policy.
set -euo pipefail
LAB_TEST_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$LAB_TEST_DIR/test.sh" \
    --episodes 15 --num-envs 1 --mission-suite "$LAB_TEST_DIR/artifacts/missions_15.json" "$@"
