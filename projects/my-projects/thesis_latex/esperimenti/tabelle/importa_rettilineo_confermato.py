"""Freeze the user-confirmed remote results from the existing LaTeX tables."""
import hashlib
import json
import math
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXPERIMENTS = ROOT / 'spot_rl_energy_navigation/analysis_thesis_experiments/case_1_3_general_terrain/experiments'
CAUSES = {'caduta': 'fallen', 'timeout': 'timeout', 'out of bounds': 'out_of_bounds'}


def number(cell):
    return float(cell.strip('$ ').replace('{,}', '.'))


def table_rows(path):
    rows = {}
    for line in path.read_text().splitlines():
        if not re.match(r'^\d+ &', line):
            continue
        cells = line.removesuffix(r' \\').split(' & ')
        if len(cells) != 8:
            raise ValueError('Expected eight table columns')
        episode = int(cells[0])
        reason = 'goal' if cells[-1] == 'Successo' else CAUSES[cells[-1].removeprefix('Fallimento (').removesuffix(')')]
        record = dict(episode=episode, reason=reason, success=int(reason == 'goal'), latex_row=line,
                      start_cell=cells[1], goal_cell=cells[2])
        if reason == 'goal':
            record.update(zip(('path_m', 'energy_j', 'time_s', 'mean_speed_m_s'), map(number, cells[3:7])))
        elif cells[3:7] != ['---'] * 4:
            raise ValueError('Failed-run metrics must remain masked')
        if episode in rows:
            raise ValueError('Duplicate episode')
        rows[episode] = record
    if set(rows) != set(range(1, 41)):
        raise ValueError('Expected exactly E1 to E40')
    return rows


def main():
    for distance in (12, 18):
        target = HERE / f'big_map_{distance}m_rettilineo_confermato.json'
        if target.exists():
            raise FileExistsError('Keep the confirmed snapshot; do not overwrite it')
        source = HERE / f'straight_line_command_mean_{distance}m.tex'
        policy_table = HERE / f'big_map_{distance}m_risultati.tex'
        baseline, policy = table_rows(source), table_rows(policy_table)
        panel_file = EXPERIMENTS / f'straight_line_baseline_command_mean/inputs/missions_{distance}m.json'
        panel = json.loads(panel_file.read_text())
        speed_checks = []
        for case in panel['cases']:
            ep = case['episode']; b = baseline[ep]; p = policy[ep]; raw = case['reference_policy_result']
            if (b['start_cell'], b['goal_cell']) != (p['start_cell'], p['goal_cell']):
                raise ValueError('Paired mission endpoints differ')
            if p['reason'] != raw['reason']:
                raise ValueError('Policy outcomes differ from archived results')
            if p['reason'] == 'goal':
                for field, places in [('path_m', 2), ('energy_j', 1), ('time_s', 1)]:
                    if not math.isclose(p[field], float(f"{float(raw[field]):.{places}f}"), abs_tol=1e-9):
                        raise ValueError('Policy table differs from archived measurements')
            case['confirmed_straight_line_result'] = b
            if b['reason'] == 'goal':
                expected = b['path_m'] / b['time_s']
                if abs(expected - b['mean_speed_m_s']) > .007:
                    speed_checks.append(dict(episode=ep, displayed_speed_m_s=b['mean_speed_m_s'],
                        path_over_time_m_s=expected))
        snapshot = dict(distance_m=distance,
            source_table=str(source.relative_to(ROOT)),
            source_table_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            source_table_content=source.read_text(),
            policy_table_sha256=hashlib.sha256(policy_table.read_bytes()).hexdigest(),
            note='User-confirmed remote results imported from tables 26–29 (12 m) and 30–33 (18 m). Original remote raw measurements were not supplied here. Preserve displayed values exactly; never replace them with earlier local run data.',
            precision='Baseline metrics use displayed precision: path 0.01 m, energy 0.1 J, time 0.1 s. Policy metrics use validated full-precision archived values.',
            speed_consistency_checks=speed_checks, panel=panel)
        target.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + '\n')
        print(f'{distance} m: frozen forty confirmed rows; {len(speed_checks)} displayed speed discrepancies recorded.')


if __name__ == '__main__':
    main()
