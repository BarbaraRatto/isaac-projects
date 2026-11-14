#!/usr/bin/env bash
# Plot already recorded results; never starts the simulator or changes the policy.
set -euo pipefail
LAB_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAB_PYTHON="${GENERAL_MAP_ANALYSIS_PYTHON:-/home/isaac/isaaclab/2.3.0/_isaac_sim/python.sh}"
if [[ "${1:-}" == --help ]]; then
    echo 'Usage: bash analyze_results.sh [results/evaluation_FOLDER] [analyze_paths.py options]'
    echo 'Default: most recently modified completed evaluation folder; writes its analysis/ subfolder.'
    exit 0
fi
if [[ $# -gt 0 && "$1" != --* ]]; then
    LAB_RESULTS="$1"; shift
else
    LAB_RESULTS="$("$LAB_PYTHON" -c 'import sys; from pathlib import Path; p=list(Path(sys.argv[1]).glob("evaluation_*/summary.json")); print(max(p,key=lambda x:x.stat().st_mtime).parent) if p else sys.exit("No completed test results found")' "$LAB_DIR/results")"
fi
exec "$LAB_PYTHON" "$LAB_DIR/analyze_paths.py" "$LAB_RESULTS" "$@"
