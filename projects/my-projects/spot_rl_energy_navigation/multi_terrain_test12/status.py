"""Read-only progress report; standard Python, no Isaac Sim, torch or GPU required."""
import argparse
import csv
import io
import json
from collections import Counter
from pathlib import Path
import time


def rows(path):
    if not path.is_file():
        return []
    text = path.read_text()
    # A running writer may be in the middle of appending a record.
    text = text[:text.rfind('\n')+1]
    return [row for row in csv.DictReader(io.StringIO(text)) if None not in row and None not in row.values()]


def duration(seconds):
    minutes = max(0, round(seconds/60))
    return f'{minutes//60} h {minutes%60:02d} min'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, help='default: most recent run containing training.csv')
    parser.add_argument('--window', type=int, default=100, help='recent completed training episodes')
    args = parser.parse_args()
    if args.window < 1:
        parser.error('--window must be positive')
    root = Path(__file__).resolve().parent
    candidates = list((root/'runs').glob('*/training.csv'))
    if args.run_dir is None and not candidates:
        print('Nessun log di training disponibile: l’inizializzazione potrebbe essere ancora in corso.')
        return
    run = args.run_dir or max(candidates, key=lambda p: p.stat().st_mtime).parent
    progress = rows(run/'training.csv')
    print(f'Run: {run.name}')
    if not progress:
        print('In attesa del primo rollout completato. L\'ETA sarà disponibile dopo il primo aggiornamento.')
        return
    config = json.loads((run/'config.json').read_text())
    target = int(config['arguments']['timesteps'])
    initial = int(config.get('initial_transitions', 0))
    last = progress[-1]
    transitions = int(last['transitions'])
    elapsed = float(last['wall_seconds'])
    # Up to five recent complete intervals; periodic evaluation time is included
    # once the following rollout has been logged.
    if len(progress) > 1:
        first = progress[max(0, len(progress)-6)]
        rate = (transitions-int(first['transitions']))/max(1e-9, elapsed-float(first['wall_seconds']))
    else:
        rate = (transitions-initial)/max(1e-9, elapsed)
    age = max(0, time.time()-(run/'training.csv').stat().st_mtime)
    print(f'Training registrato: {transitions:,}/{target:,} ({100*transitions/target:.2f}%)')
    print(f'Tempo trascorso registrato: {duration(elapsed)} | velocità recente: {rate:.1f} transizioni/s')
    if rate > 0:
        print(f'Tempo residuo stimato dall\'ultimo rollout: {duration(max(0, target-transitions)/rate)}')
    print(f'Ultimo aggiornamento del log: {age:.0f} secondi fa; aggiornamento a fine rollout.')
    summary = run/'summary.json'
    if summary.exists():
        saved = json.loads(summary.read_text())
        print('STATO: interrotto e salvato.' if saved['interrupted'] else 'STATO: training completato.')
    elif transitions >= target:
        print('Budget raggiunto: possono essere ancora in corso valutazione finale e salvataggio.')
    else:
        print('ETA indicativa: le valutazioni e il carico della macchina possono allungarla.')
    episodes = rows(run/'episodes.csv')
    recent = episodes[-args.window:]
    for label, group in [('Tutti gli episodi di training', episodes), (f'Ultimi {len(recent)} episodi', recent)]:
        if group:
            successes = sum(int(r['success']) for r in group)
            print(f'{label}: {successes}/{len(group)} successi ({100*successes/len(group):.1f}%) '
                  f'| esiti: {dict(Counter(r["reason"] for r in group))}')
    print(f'Celle valide medie ultimo rollout: visive {float(last["visual_cells"]):.1f}/81, '
          f'geometriche {float(last["geometry_cells"]):.1f}/81')
    if 'visual_remembered_cells' in last:
        print(f'Celle con misure precedenti: visive {float(last["visual_remembered_cells"]):.1f}, '
              f'geometriche {float(last["geometry_remembered_cells"]):.1f}')
    print(f'KL: {float(last["kl"]):.4f} | entropia: {float(last["entropy"]):.3f} '
          f'| azioni sature: {100*float(last["action_saturation"]):.2f}%')
    evaluations = rows(run/'evaluations.csv')
    print('\nValutazioni deterministiche (più informative dei successi durante esplorazione):')
    if not evaluations:
        interval = int(config['arguments']['eval_every'])
        if interval:
            print(f'Nessuna ancora: prima valutazione al confine di rollout dopo {interval:,} transizioni.')
        else:
            print('Valutazione periodica disabilitata.')
    for record in evaluations[-5:]:
        energy = record['mean_success_energy_j']
        energy_text = f'{float(energy):.0f} J' if energy else 'n/d (nessun successo)'
        print(f'  {int(record["transitions"]):,} transizioni: '
              f'{100*float(record["success_rate"]):.1f}% successi su {record["episodes"]} episodi; '
              f'energia media dei successi {energy_text}')


if __name__ == '__main__':
    main()
