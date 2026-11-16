"""Build the four 12 m tables for phase-specific mean commanded speeds."""
from pathlib import Path
import csv
import importlib.util
import json
import math

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUN = ROOT / 'spot_rl_energy_navigation/analysis_thesis_experiments/case_1_3_general_terrain/experiments/straight_line_baseline_command_mean/results_exp/distance_12m_all_20261116_231439_575172'


def main():
    spec = importlib.util.spec_from_file_location('tables12', HERE / 'genera_tabelle_big_map_12m.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    complete = json.loads((RUN / 'complete.json').read_text())
    if complete.get('logging_checks') != 'passed' or complete.get('completed_missions') != 40:
        raise ValueError('Expected forty completed and checked missions')
    panel = json.loads((RUN / 'inputs/selected_missions.json').read_text())
    records = list(csv.DictReader((RUN / 'episodes.csv').open()))
    results = {int(r['episode']): r for r in records}
    if len(records) != 40 or set(results) != set(range(1, 41)):
        raise ValueError('Expected exactly E1 to E40')
    tables = []
    for index, category in enumerate(panel['category_order'], 1):
        cases = [c for c in panel['cases'] if c['category'] == category]
        if len(cases) != 10:
            raise ValueError('Expected ten missions per category')
        rows = []
        for case in cases:
            r = results[case['episode']]
            if not math.isclose(float(r['reference_speed_m_s']), case['command_means']['navigation_forward_m_s'], abs_tol=1e-9):
                raise ValueError('Incorrect assigned speed')
            if not math.isclose(float(r['alignment_reference_yaw_rad_s']), case['command_means']['alignment_yaw_rad_s'], abs_tol=1e-9):
                raise ValueError('Incorrect assigned alignment yaw speed')
            if r['category'] != category:
                raise ValueError('Category mismatch')
            m = case['mission']
            for field, value in [('start_x', m['start'][0]), ('start_y', m['start'][1]),
                                 ('goal_x', m['goal'][0]), ('goal_y', m['goal'][1])]:
                if not math.isclose(float(r[field]), value, abs_tol=1e-9):
                    raise ValueError('Endpoint mismatch')
            fixed = [str(case['episode']), helper.point(m['start']), helper.point(m['goal'])]
            if r['reason'] == 'goal':
                values = [helper.number(r['path_m']), helper.number(r['energy_j'], 1),
                          helper.number(r['time_s'], 1),
                          helper.number(float(r['path_m']) / float(r['time_s'])), 'Successo']
            else:
                values = ['---'] * 4 + ['Fallimento (' + helper.CAUSES[r['reason']] + ')']
            rows.append(' & '.join(fixed + values) + r' \\')
        tables.append(helper.table(
            f'Missioni a 12~m, categoria {index}: riferimento rettilineo con velocità comandate medie, calcolate separatamente per rotazione iniziale e avanzamento. La velocità media in tabella è quella effettiva, pari alla distanza percorsa divisa per il tempo totale.',
            f'tab:straight-line-command-mean-12m-cat{index}', rows))
    (HERE / 'straight_line_command_mean_12m.tex').write_text('\n'.join(tables))
    print('Generated four tables from 40 checked missions.')


if __name__ == '__main__':
    main()
