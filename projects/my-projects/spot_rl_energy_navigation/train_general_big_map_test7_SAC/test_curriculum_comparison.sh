#!/usr/bin/env bash
# Three new evaluations; reuse the completed final-model stage-1 evaluation.
set -euo pipefail
COMPARE_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
COMPARE_RUN="$COMPARE_DIR/runs/general_map_test7_sac_20261029_163507_402204"
COMPARE_OLD="$COMPARE_DIR/results/evaluation_20261029_233158_931651"
COMPARE_EXTRA=()
if [[ $# -eq 1 && "$1" == '--headless' ]]; then
    COMPARE_EXTRA=(--headless)
elif [[ $# -ne 0 ]]; then
    echo "Usage: bash $0 [--headless]" >&2
    exit 2
fi
for file in "$COMPARE_RUN/policy_150017.pt" "$COMPARE_RUN/policy_final.pt"; do
    [[ -f "$file" ]] || { echo "Missing checkpoint: $file" >&2; exit 1; }
done
python3 "$COMPARE_DIR/summarize_curriculum_comparison.py" --validate-reuse "$COMPARE_OLD"
COMPARE_OUTPUT="$(mktemp -d "$COMPARE_DIR/results/curriculum_comparison_$(date +%Y%m%d_%H%M%S)_XXXXXX")"
echo "Comparison results: $COMPARE_OUTPUT"
cp -a "$COMPARE_OLD" "$COMPARE_OUTPUT/200k_stage1"
echo '[1/3] Checkpoint 150017 — stage 0, nine missions.'
bash "$COMPARE_DIR/test.sh" --checkpoint "$COMPARE_RUN/policy_150017.pt" \
    --stage 0 --output "$COMPARE_OUTPUT/150k_stage0" "${COMPARE_EXTRA[@]}"
echo '[2/3] Checkpoint 200000 — same nine stage-0 missions.'
bash "$COMPARE_DIR/test.sh" --checkpoint "$COMPARE_RUN/policy_final.pt" \
    --stage 0 --output "$COMPARE_OUTPUT/200k_stage0" "${COMPARE_EXTRA[@]}"
echo '[3/3] Checkpoint 150017 — same stage-1 missions as the saved final evaluation.'
bash "$COMPARE_DIR/test.sh" --checkpoint "$COMPARE_RUN/policy_150017.pt" \
    --stage 1 --output "$COMPARE_OUTPUT/150k_stage1" "${COMPARE_EXTRA[@]}"
python3 "$COMPARE_DIR/summarize_curriculum_comparison.py" "$COMPARE_OUTPUT"
