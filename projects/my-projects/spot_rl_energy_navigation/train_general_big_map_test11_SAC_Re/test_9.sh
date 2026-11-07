#!/usr/bin/env bash
# Nine new, fixed missions; read-only evaluation, one visible robot.
set -euo pipefail
LAB_TEST_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAB_RUNS_ROOT="$LAB_TEST_DIR/runs"
LAB_ARGS=()
LAB_EXPLICIT_CHECKPOINT=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --runs-root)
            [[ $# -ge 2 ]] || { echo 'Missing --runs-root value.' >&2; exit 2; }
            LAB_RUNS_ROOT="$2"; shift 2 ;;
        --runs-root=*) LAB_RUNS_ROOT="${1#*=}"; shift ;;
        --help|-h)
            echo 'Usage: bash test_9.sh [--checkpoint /path/policy_best.pt] [--runs-root /root/test11-re-runs] [play.py options]'
            echo '2 grey-grey near + 2 at 6m + 2 at 9m + 2 at 12m + 1 grey-green.'
            echo 'Default: policy_best.pt from latest completed run, 1 visible robot, saved trajectories.'
            exit 0 ;;
        --checkpoint|--checkpoint=*) LAB_EXPLICIT_CHECKPOINT=1; LAB_ARGS+=("$1"); shift ;;
        *) LAB_ARGS+=("$1"); shift ;;
    esac
done
LAB_CHECKPOINT_ARGS=()
if [[ "$LAB_EXPLICIT_CHECKPOINT" == 0 ]]; then
    LAB_BEST="$(python3 "$LAB_TEST_DIR/select_checkpoint.py" --runs-root "$LAB_RUNS_ROOT")"
    LAB_CHECKPOINT_ARGS=(--checkpoint "$LAB_BEST")
    echo "[TEST11_RE_9] Selected policy: $LAB_BEST"
fi
exec bash "$LAB_TEST_DIR/test.sh" --num-envs 1 --episodes 9 --seed 231207 \
    --mission-suite "$LAB_TEST_DIR/artifacts/missions_9.json" \
    "${LAB_CHECKPOINT_ARGS[@]}" "${LAB_ARGS[@]}"
