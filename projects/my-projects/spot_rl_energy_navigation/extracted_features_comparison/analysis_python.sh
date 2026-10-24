#!/usr/bin/env bash
# Run offline scripts in the isolated environment; no SimulationApp or GPU.
set -euo pipefail
COMPARISON_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -x "$COMPARISON_DIR/.venv/bin/python" ]]; then
    echo "Run first: bash $COMPARISON_DIR/setup_analysis.sh" >&2
    exit 1
fi
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH
export MPLCONFIGDIR="$COMPARISON_DIR/.cache/matplotlib"
export OPENBLAS_NUM_THREADS=2
export OMP_NUM_THREADS=2
exec "$COMPARISON_DIR/.venv/bin/python" "$@"
