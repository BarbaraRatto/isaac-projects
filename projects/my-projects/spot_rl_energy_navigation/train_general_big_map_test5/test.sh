#!/usr/bin/env bash
# Visible evaluation of the newest completed test5 run; no training or PCA fit.
set -euo pipefail
GENERAL_MAP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${GENERAL_MAP_ISAACLAB:-}" && -f "$GENERAL_MAP_DIR/../cloud/env.sh" ]]; then
    # shellcheck disable=SC1091
    source "$GENERAL_MAP_DIR/../cloud/env.sh"
fi
GENERAL_MAP_ISAACLAB="${GENERAL_MAP_ISAACLAB:-/home/isaac/isaaclab/2.3.0/isaaclab.sh}"
GENERAL_MAP_AUTO_CHECKPOINT=1
for arg in "$@"; do
    case "$arg" in
        --checkpoint|--checkpoint=*|--help|-h) GENERAL_MAP_AUTO_CHECKPOINT=0 ;;
    esac
done
GENERAL_MAP_CHECKPOINT_ARGS=()
if [[ "$GENERAL_MAP_AUTO_CHECKPOINT" == 1 ]]; then
    GENERAL_MAP_CHECKPOINT=""
    for candidate in "$GENERAL_MAP_DIR"/runs/*/policy_final.pt; do
        [[ -f "$candidate" ]] || continue
        if [[ -z "$GENERAL_MAP_CHECKPOINT" || "$candidate" -nt "$GENERAL_MAP_CHECKPOINT" ]]; then
            GENERAL_MAP_CHECKPOINT="$candidate"
        fi
    done
    if [[ -z "$GENERAL_MAP_CHECKPOINT" ]]; then
        echo 'Nessun policy_final.pt trovato nel test 5. Attendi la fine del training oppure indica --checkpoint /percorso/policy_latest.pt.' >&2
        exit 1
    fi
    GENERAL_MAP_CHECKPOINT_ARGS=(--checkpoint "$GENERAL_MAP_CHECKPOINT")
    echo "[TEST5] Modello selezionato: $GENERAL_MAP_CHECKPOINT"
fi
exec "$GENERAL_MAP_ISAACLAB" -p "$GENERAL_MAP_DIR/play.py" \
    --device cuda:0 --agent-device cpu --num-envs 1 --episodes 9 --seed 20000 \
    "${GENERAL_MAP_CHECKPOINT_ARGS[@]}" "$@"
