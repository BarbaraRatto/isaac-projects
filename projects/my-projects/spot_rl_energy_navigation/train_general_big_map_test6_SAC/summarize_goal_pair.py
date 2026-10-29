"""Validate paired missions and summarize simulation results without Isaac imports."""
import csv
import math
import sys
from pathlib import Path


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def rotations(directory):
    previous, total = {}, {}
    for r in read(directory/'trajectories.csv'):
        key = (r['env_id'], r['episode'])
        if r['phase'] != 'navigation':
            previous.pop(key, None)
            continue
        yaw = float(r['yaw'])
        if key in previous:
            delta = yaw-previous[key]
            total[key] = total.get(key, 0.)+abs(math.atan2(math.sin(delta), math.cos(delta)))
        previous[key] = yaw
    return total


def main():
    root = Path(sys.argv[1])
    folders = [root/'original_300k', root/'goal_norm_100k']
    episodes = [{(r['env_id'], r['episode']): r for r in read(p/'episodes.csv')} for p in folders]
    if episodes[0].keys() != episodes[1].keys() or len(episodes[0]) != 9:
        raise ValueError('Expected nine matching episodes in each evaluation')
    turns = [rotations(p) for p in folders]
    lines = ['# Confronto simulato: normalizzazione goal', '',
             '| Episodio | Variante | Esito | Percorso m | Energia J | Durata s | Rotazione assoluta rad* |',
             '|---|---|---|---:|---:|---:|---:|']
    for key in sorted(episodes[0], key=lambda k: (int(k[0]), int(k[1]))):
        a, b = [e[key] for e in episodes]
        for field in ('start_id', 'goal_id', 'band', 'layout'):
            if a[field] != b[field]:
                raise ValueError(f'Mission mismatch: {key}/{field}')
        for field in ('start_x', 'start_y', 'goal_x', 'goal_y', 'initial_yaw'):
            if not math.isclose(float(a[field]), float(b[field]), abs_tol=1e-4, rel_tol=0):
                raise ValueError(f'Mission pose mismatch: {key}/{field}')
        for i, r in enumerate((a, b)):
            label = 'originale' if i == 0 else 'solo goal 100k'
            lines.append(f"| {key[1]} | {label} | {r['reason']} | {float(r['path_m']):.2f} | {float(r['energy_j']):.0f} | {int(r['steps'])*.2:.1f} | {turns[i].get(key, 0.):.2f} |")
    lines += ['', '*Rotazione: somma degli incrementi assoluti di yaw durante la navigazione, campionati a 0.2 s;',
              'esclude allineamento iniziale e ultimo intervallo senza campione successivo. Non è una misura diretta dei giri inutili.',
              'Percorso, energia e durata riportano i totali degli episodi, incluso allineamento.',
              'Start, goal, orientamento iniziale e identificativi delle nove missioni verificati uguali.',
              'Una sola esecuzione per variante: possibili differenze numeriche della simulazione; risultati da confermare.',
              'Confrontare efficienza soprattutto nelle missioni completate da entrambe le varianti.']
    report = '\n'.join(lines)+'\n'
    (root/'REPORT.md').write_text(report)
    print(report)
    print(f'Results: {root}')


if __name__ == '__main__':
    main()
