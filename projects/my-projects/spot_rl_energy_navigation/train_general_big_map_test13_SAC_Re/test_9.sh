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
            echo 'Usage: bash test_9.sh [--checkpoint /path/policy_best.pt] [--runs-root /root/test13-re-runs] [play.py options]'
            echo '2 grey-grey near + 2 at 6m + 2 at 9m + 2 at 12m + 1 grey-grey with ice on the direct segment.'
            echo 'Default: policy_best.pt from latest completed run, 1 visible robot, saved trajectories.'
            exit 0 ;;
        --checkpoint|--checkpoint=*) LAB_EXPLICIT_CHECKPOINT=1; LAB_ARGS+=("$1"); shift ;;
        *) LAB_ARGS+=("$1"); shift ;;
    esac
done
LAB_CHECKPOINT_ARGS=()
if [[ "$LAB_EXPLICIT_CHECKPOINT" == 0 ]]; then
    LAB_FINAL=""
    if [[ -f "$LAB_RUNS_ROOT/latest_run.txt" ]]; then
        LAB_LATEST="$(cat "$LAB_RUNS_ROOT/latest_run.txt")"
        [[ -f "$LAB_LATEST/policy_final.pt" ]] || {
            echo 'Latest run has no policy_final.pt yet. Wait for completion or supply --checkpoint.' >&2; exit 1;
        }
        LAB_FINAL="$LAB_LATEST/policy_final.pt"
    else
        for candidate in "$LAB_RUNS_ROOT"/*/policy_final.pt; do
            [[ -f "$candidate" ]] || continue
            if [[ -z "$LAB_FINAL" || "$candidate" -nt "$LAB_FINAL" ]]; then LAB_FINAL="$candidate"; fi
        done
    fi
    [[ -n "$LAB_FINAL" ]] || { echo 'No completed run found. Supply --checkpoint or --runs-root.' >&2; exit 1; }
    LAB_BEST="$(dirname -- "$LAB_FINAL")/policy_best.pt"
    [[ -f "$LAB_BEST" ]] || { echo "Missing checkpoint: $LAB_BEST" >&2; exit 1; }
    LAB_CHECKPOINT_ARGS=(--checkpoint "$LAB_BEST")
    echo "[TEST13_RE_9] Selected policy: $LAB_BEST"
fi
exec bash "$LAB_TEST_DIR/test.sh" --num-envs 1 --episodes 9 --seed 231313 \
    --mission-suite "$LAB_TEST_DIR/artifacts/missions_9.json" \
    "${LAB_CHECKPOINT_ARGS[@]}" "${LAB_ARGS[@]}"
