#!/usr/bin/env bash
# Sequential paired simulation: original 300k, then 300k with 100k goal statistics.
set -euo pipefail
PAIR_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PAIR_EXTRA=()
if [[ $# -eq 1 && "$1" == '--headless' ]]; then
    PAIR_EXTRA=(--headless)
elif [[ $# -ne 0 ]]; then
    echo "Usage: bash $0 [--headless]" >&2
    exit 2
fi
PAIR_MODEL="$PAIR_DIR/runs/general_map_test6_sac_20261029_001023_953670/policy_final.pt"
PAIR_DONOR="$PAIR_DIR/runs/general_map_test6_sac_20261028_201519_133017/policy_final.pt"
[[ -f "$PAIR_MODEL" && -f "$PAIR_DONOR" ]] || { echo 'Missing checkpoint' >&2; exit 1; }
PAIR_OUTPUT="$(mktemp -d "$PAIR_DIR/results/goal_normalization_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
echo "Results: $PAIR_OUTPUT"
echo '[1/2] Original 300k model: nine stage-1 missions, seed 20000.'
bash "$PAIR_DIR/test.sh" --checkpoint "$PAIR_MODEL" --stage 1 \
    --output "$PAIR_OUTPUT/original_300k" "${PAIR_EXTRA[@]}"
echo '[2/2] Same actor and missions; only goal statistics from 100k.'
bash "$PAIR_DIR/test.sh" --checkpoint "$PAIR_MODEL" --stage 1 \
    --goal-normalizer-from "$PAIR_DONOR" --output "$PAIR_OUTPUT/goal_norm_100k" "${PAIR_EXTRA[@]}"
python3 "$PAIR_DIR/summarize_goal_pair.py" "$PAIR_OUTPUT"
