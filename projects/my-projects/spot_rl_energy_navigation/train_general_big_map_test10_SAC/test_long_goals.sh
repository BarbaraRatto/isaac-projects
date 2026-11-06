#!/usr/bin/env bash
# Deterministic evaluation only. Original policy/PCA/scalers remain frozen.
set -euo pipefail
TEST10_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TEST10_CHECKPOINT="$TEST10_DIR/checkpoint_trained_online/test10_relevant/policy_best_stage_1.pt"
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    echo 'Usage: bash test_long_goals.sh [play.py arguments...]'
    echo 'Defaults: imported best_stage_1, 9 episodes, random safe endpoints 12–24 m apart.'
    echo 'Examples: --headless; --distance-range 18 26; --checkpoint /path/policy_final.pt'
    echo 'The stage 4 selector is only for evaluation; the checkpoint was trained through stage 1.'
    exit 0
fi
echo '[TEST10] Long goals: default distance 12–24 m; 9 episodes; best_stage_1 checkpoint.'
echo '[TEST10] Evaluation only, beyond trained distances; saving trajectories for plots.'
exec bash "$TEST10_DIR/test.sh" \
    --checkpoint "$TEST10_CHECKPOINT" --stage 4 --suite free --distance-range 12 24 \
    --episodes 9 --seed 20000 "$@"
