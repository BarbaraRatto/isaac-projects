#!/usr/bin/env bash
# Resolve the last completed/interrupted run in this shell, then add exactly 1M.
# No user terminal variable is needed; start_test13_re.sh writes a NEW destination.
set -euo pipefail
LAB_PROJECT="${1:-/root/spot-test13-sac-re-cloud}"
LAB_RUNS_ROOT="${2:-/root/test13-re-runs}"
[[ $# -le 2 ]] || { echo 'Usage: bash continue_test13_1m.sh [PROJECT] [RUNS_ROOT]' >&2; exit 2; }
[[ -f "$LAB_PROJECT/spot_rl_energy_navigation/cloud/env.sh" ]] || {
    echo 'Installed test13 env.sh not found; use the original project directory.' >&2; exit 1;
}
[[ -f "$LAB_RUNS_ROOT/latest_run.txt" ]] || { echo 'latest_run.txt not found in runs root.' >&2; exit 1; }
LAB_PREVIOUS_RUN="$(cat "$LAB_RUNS_ROOT/latest_run.txt")"
[[ -n "$LAB_PREVIOUS_RUN" && "$LAB_PREVIOUS_RUN" == /* ]] || { echo 'Invalid previous run path.' >&2; exit 1; }
LAB_TARGET="$(python3 - "$LAB_PREVIOUS_RUN" "$LAB_RUNS_ROOT" <<'PY'
import json,sys
from pathlib import Path
run=Path(sys.argv[1]).resolve()
if not run.is_relative_to(Path(sys.argv[2]).resolve()):
    raise SystemExit('Previous run is outside the requested runs root.')
summary=run/'summary.json'
if not summary.is_file():
    raise SystemExit('Previous run has no summary.json: wait for completion or interrupt cleanly and wait for Saved.')
saved=json.loads(summary.read_text())
recovery=run/'recovery_status.json'
if not recovery.is_file() or not (run/'recovery').is_dir():
    raise SystemExit('Complete recovery missing. Do not resume from an ordinary best checkpoint.')
status=json.loads(recovery.read_text())
transitions=saved['transitions']
if type(transitions) is not int or transitions<1 or status['transitions']!=transitions:
    raise SystemExit('Summary and last complete recovery do not agree. Inspect the run before continuing.')
if not (run/status['checkpoint']).is_file():
    raise SystemExit('Recovery policy file missing. Keep the entire recovery directory.')
print(transitions+1_000_000)
PY
)"
echo "[TEST13 CONTINUE] Parent: $LAB_PREVIOUS_RUN"
echo "[TEST13 CONTINUE] Additional navigation transitions: 1,000,000; TOTAL target: $LAB_TARGET"
echo '[TEST13 CONTINUE] Equal episode probability for bands 0–3; evaluation at stage 3 every 250,000 new transitions.'
# Require the update before creating a new latest_run pointer.
python3 - "$LAB_PROJECT/spot_rl_energy_navigation/train_general_big_map_test13_SAC_Re/train.py" <<'PY'
from pathlib import Path
import sys
if "'--mission-distribution'" not in Path(sys.argv[1]).read_text():
    raise SystemExit('Mixed-distribution update is not installed. Run install.py --apply first.')
PY
cd "$LAB_PROJECT"
exec bash "$LAB_PROJECT/start_test13_re.sh" "$LAB_RUNS_ROOT" \
    --resume "$LAB_PREVIOUS_RUN" \
    --mission-distribution uniform --eval-every 250000 --timesteps "$LAB_TARGET" \
    --device cuda:0 --agent-device cuda:0 \
    --num-envs 128 --dino-batch-size 128 --gradient-steps 16 \
    --map-backend cuda --rgb-backend cuda --replay-device cuda:0
