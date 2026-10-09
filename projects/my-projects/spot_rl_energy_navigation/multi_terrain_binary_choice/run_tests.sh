#!/usr/bin/env bash
# Run evaluations only. Invoke from the my-projects repository root.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
    echo "Usage: bash spot_rl_energy_navigation/multi_terrain_binary_choice/run_tests.sh CHECKPOINT [diagnostic|full]" >&2
    exit 2
fi

checkpoint=$1
suite=${2:-diagnostic}
if [[ ! -f "$checkpoint" ]]; then
    echo "Checkpoint not found: $checkpoint" >&2
    exit 2
fi
if [[ "$suite" != diagnostic && "$suite" != full ]]; then
    echo "Suite must be diagnostic or full" >&2
    exit 2
fi

root=spot_rl_energy_navigation/multi_terrain_binary_choice
out_dir=$root/results/$(basename "$(dirname "$checkpoint")")
mkdir -p "$out_dir"

run_one() {
    local layout=$1 geometry=$2 episodes=$3
    local output=$out_dir/${layout}_${geometry}_${episodes}episodes.csv
    if [[ -e "$output" ]]; then
        local actual
        actual=$(python3 -c 'import csv,sys; print(sum(1 for _ in csv.DictReader(open(sys.argv[1], newline="", encoding="utf-8"))))' "$output")
        if [[ "$actual" != "$episodes" ]]; then
            echo "Incomplete output ($actual/$episodes): $output" >&2
            exit 2
        fi
        echo "[TEST] reusing completed evaluation: $output"
        return
    fi
    echo "[TEST] layout=$layout geometry=$geometry episodes=$episodes"
    /home/isaac/isaaclab/2.3.0/isaaclab.sh -p "$root/play.py" \
        --headless --checkpoint "$checkpoint" --layout "$layout" \
        --geometry "$geometry" --episodes "$episodes" --output "$output"
}

# First establish whether the final policy can navigate a map seen in training.
run_one asphalt_rocks__right_rocks centered 10
run_one asphalt_rocks__mixed centered 10

if [[ "$suite" == full ]]; then
    # Remaining held-out arrangements, including asymmetric start/goal positions.
    run_one asphalt_rocks__mixed short_costly 20
    for layout in ramp_gravel__mixed obstacles_gravel__mixed stairs_rocks__mixed; do
        run_one "$layout" centered 20
        run_one "$layout" short_costly 20
    done
fi

python3 "$root/analyze_tests.py" "$out_dir"
