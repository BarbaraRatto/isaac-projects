#!/usr/bin/env bash
# Prepare only the local CPU-analysis environment. Does not start Isaac or DINO.
set -euo pipefail
COMPARISON_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PIP_CACHE_DIR="$COMPARISON_DIR/.cache/pip"
python3 -m venv "$COMPARISON_DIR/.venv"
"$COMPARISON_DIR/.venv/bin/python" -m pip install \
    -r "$COMPARISON_DIR/requirements-analysis.txt"
"$COMPARISON_DIR/.venv/bin/python" -c \
    'import numpy, scipy, sklearn, matplotlib, cv2; print("Local CPU analysis environment: OK")'
