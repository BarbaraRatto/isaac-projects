#!/usr/bin/env bash
# Exact subset of the previous 15 missions, evaluated with policy_best.
set -euo pipefail
LAB_TEST_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LAB_SELECT_BEST=1
for arg in "$@"; do
    case "$arg" in --checkpoint|--checkpoint=*|--help|-h) LAB_SELECT_BEST=0 ;; esac
done
LAB_CHECKPOINT_ARGS=()
if [[ "$LAB_SELECT_BEST" == 1 ]]; then
    LAB_FINAL=""
    for candidate in "$LAB_TEST_DIR"/runs/*/policy_final.pt; do
        [[ -f "$candidate" ]] || continue
        if [[ -z "$LAB_FINAL" || "$candidate" -nt "$LAB_FINAL" ]]; then LAB_FINAL="$candidate"; fi
    done
    [[ -n "$LAB_FINAL" ]] || { echo 'Nessun training completato trovato.' >&2; exit 1; }
    LAB_BEST="$(dirname -- "$LAB_FINAL")/policy_best.pt"
    [[ -f "$LAB_BEST" ]] || { echo "Checkpoint best mancante: $LAB_BEST" >&2; exit 1; }
    LAB_CHECKPOINT_ARGS=(--checkpoint "$LAB_BEST")
    echo "[TEST9_BEST] Modello selezionato: $LAB_BEST"
fi
exec bash "$LAB_TEST_DIR/test.sh" --episodes 9 --num-envs 1 \
    --mission-suite "$LAB_TEST_DIR/artifacts/missions_9_best.json" "${LAB_CHECKPOINT_ARGS[@]}" "$@"
