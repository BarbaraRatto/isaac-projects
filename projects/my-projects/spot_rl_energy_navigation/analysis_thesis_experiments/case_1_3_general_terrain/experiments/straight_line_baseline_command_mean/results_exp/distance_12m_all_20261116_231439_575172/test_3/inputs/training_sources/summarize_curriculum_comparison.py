"""Validate mission pairing and summarize the 150k/200k curriculum comparison."""
import argparse
import csv
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE/'runs/general_map_test7_sac_20261029_163507_402204'


def load(folder, stage, checkpoint, transitions):
    config = json.loads((folder/'evaluation_config.json').read_text())
    doc = json.loads((folder/'summary.json').read_text())
    assert Path(config['checkpoint']).resolve() == checkpoint.resolve(), 'Wrong checkpoint'
    assert Path(doc['checkpoint']).resolve() == checkpoint.resolve(), 'Wrong summary checkpoint'
    assert config['seed'] == doc['seed'] == 20000
    assert config['num_envs'] == doc['num_envs'] == 1
    assert config['requested_stage'] == doc['summary']['stage'] == stage
    assert config['episodes'] == doc['summary']['episodes'] == len(doc['episodes']) == 9
    assert config['goal_scaling'] == {'mode': 'fixed_scale', 'scale_m': 10.0}
    with (folder/'episodes.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 9 and all(int(r['transitions']) == transitions for r in rows)
    assert (folder/'trajectories.csv').is_file()
    episodes = {int(e['episode']): e for e in doc['episodes']}
    assert set(episodes) == set(range(1, 10))
    return doc['summary'], episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path, nargs='?')
    parser.add_argument('--validate-reuse', type=Path)
    args = parser.parse_args()
    if args.validate_reuse:
        load(args.validate_reuse, 1, RUN/'policy_final.pt', 200000)
        print('OK: completed 200k stage-1 evaluation can be reused (nine episodes, seed 20000).')
        return
    if args.folder is None:
        parser.error('Provide comparison folder or --validate-reuse')
    lines = ['# Test 7: confronto prima/dopo il cambio di curriculum', '',
             '| Fase | Checkpoint | Successi | Timeout | Cadute |', '|---|---|---:|---:|---:|']
    detailed = []
    for stage in (0, 1):
        pairs = []
        for label, checkpoint, transitions in [('150k', 'policy_150017.pt', 150017), ('200k', 'policy_final.pt', 200000)]:
            summary, episodes = load(args.folder/f'{label}_stage{stage}', stage, RUN/checkpoint, transitions)
            pairs.append(episodes)
            lines.append(f"| {stage} | {transitions} | {summary['successes']}/9 | {summary['outcomes']['timeout']} | {summary['outcomes']['fallen']} |")
            for n, e in sorted(episodes.items()):
                detailed.append(dict(stage=stage, checkpoint=transitions, episode=n, reason=e['reason'],
                                     path_m=e['path_m'], energy_j=e['energy_j'], steps=e['steps'],
                                     final_distance_m=e['final_distance_m']))
        for n in range(1, 10):
            a, b = pairs[0][n], pairs[1][n]
            for key in ('env_id', 'start_id', 'goal_id', 'band', 'layout'):
                assert a[key] == b[key], f'Mission mismatch stage {stage}, episode {n}, {key}'
            for key in ('start_x', 'start_y', 'goal_x', 'goal_y', 'initial_yaw'):
                assert math.isclose(a[key], b[key], abs_tol=1e-4, rel_tol=0), f'Pose mismatch: {stage}/{n}/{key}'
    lines += ['', 'Start, goal e orientamento verificati uguali fra checkpoint all’interno di ogni fase.',
              'Le missioni della fase 0 e della fase 1 sono diverse: confrontare i checkpoint all’interno della stessa fase.',
              'Il risultato 200k/fase 1 è riutilizzato dalla valutazione precedente; le altre tre valutazioni sono nuove.',
              'Nove missioni per condizione sono una diagnosi preliminare, non una stima precisa della robustezza.',
              'Successi inferiori a 200k in fase 0 suggeriscono regressione; successi conservati in fase 0 ma bassi in fase 1 suggeriscono difficoltà di trasferimento.',
              'Entrambi i problemi possono coesistere. La simulazione può introdurre variabilità fra esecuzioni.']
    (args.folder/'REPORT.md').write_text('\n'.join(lines)+'\n')
    with (args.folder/'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(detailed[0]))
        writer.writeheader(); writer.writerows(detailed)
    print('\n'.join(lines))
    print(f'Results: {args.folder}')


if __name__ == '__main__':
    main()
