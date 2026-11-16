"""Build straight-line tables from completed runs and paired policy records."""
from pathlib import Path
import csv
import hashlib
import json
import math
from statistics import mean

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RUNS = {12: 'distance_12m_all_20261116_200032_934528',
        18: 'distance_18m_all_20261116_202444_543554'}
BASE = ROOT / 'spot_rl_energy_navigation/analysis_thesis_experiments/case_1_3_general_terrain/experiments/straight_line_baseline/results_exp'
FIELDS = ('path_m', 'energy_j', 'time_s')


def generate(distance, table, number, point, causes):
    run = BASE / RUNS[distance]
    complete = json.loads((run / 'complete.json').read_text())
    if complete != dict(logging_checks='passed', completed_missions=40):
        raise ValueError('Expected a complete, checked run')
    panel = json.loads((run / 'inputs/selected_missions.json').read_text())
    rows = list(csv.DictReader((run / 'episodes.csv').open()))
    baseline = {int(row['episode']): row for row in rows}
    if len(rows) != 40 or set(baseline) != set(range(1, 41)):
        raise ValueError('Expected exactly E1 to E40')
    for case in panel['cases']:
        b = baseline[case['episode']]
        for field, value in [('start_x', case['mission']['start'][0]),
                             ('start_y', case['mission']['start'][1]),
                             ('goal_x', case['mission']['goal'][0]),
                             ('goal_y', case['mission']['goal'][1]),
                             ('initial_yaw', case['mission']['yaw']),
                             ('reference_speed_m_s', case['reference_speed_m_s'])]:
            if not math.isclose(float(b[field]), value, abs_tol=1e-9):
                raise ValueError(f'Mission mismatch: {case["episode"]}, {field}')
        if b['category'] != case['category']:
            raise ValueError('Category mismatch')
    prefix = f'big_map_{distance}m'
    comparison = [r'\begin{table}[H]', r'\centering',
        r'\caption{Statistiche per categoria con e senza la policy. Le medie di ciascun metodo comprendono soltanto i suoi successi; gli insiemi di missioni riuscite possono essere diversi.}',
        rf'\label{{tab:big-map-{distance}m-confronto-statistiche}}', r'\small',
        r'\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{@{}clcccc@{}}', r'\toprule',
        r'Categoria & Metodo & \shortstack{Successo\\{}[\%]} & \shortstack{Distanza media\\{}[m]} & \shortstack{Energia media\\{}[J]} & \shortstack{Tempo medio\\{}[s]} \\', r'\midrule']
    differences = [r'\begin{table}[H]', r'\centering',
        r'\caption{Variazioni con la policy rispetto al riferimento rettilineo. Energia e distanza sono confrontate solo sulle missioni riuscite con entrambi i metodi ($N$). Un risparmio negativo indica un maggior consumo con la policy. Il trattino indica che non ci sono successi comuni.}',
        rf'\label{{tab:big-map-{distance}m-confronto-variazioni}}', r'\small',
        r'\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{@{}ccccc@{}}', r'\toprule',
        r'Categoria & $N$ & \shortstack{Variazione successo\\{}[punti percentuali]} & \shortstack{Risparmio energetico\\{}[\%]} & \shortstack{Allungamento percorso\\{}[\%]} \\', r'\midrule']
    tables = []
    statistics = []
    for index, category in enumerate(panel['category_order'], 1):
        cases = [c for c in panel['cases'] if c['category'] == category]
        if len(cases) != 10:
            raise ValueError('Expected ten missions per category')
        entries = []
        for c in cases:
            b = baseline[c['episode']]
            fixed = [str(c['episode']), point(c['mission']['start']), point(c['mission']['goal'])]
            if b['reason'] == 'goal':
                values = [number(b['path_m']), number(b['energy_j'], 1), number(b['time_s'], 1),
                          number(float(b['path_m']) / float(b['time_s'])), 'Successo']
            else:
                values = ['---'] * 4 + ['Fallimento (' + causes[b['reason']] + ')']
            entries.append(' & '.join(fixed + values) + r' \\')
        tables.append(table(f'Missioni a {distance}~m, categoria {index}: risultati del riferimento rettilineo.',
                            f'tab:big-map-{distance}m-baseline-cat{index}', entries))
        group = dict(category=index, category_name=category)
        for method, records in [('Policy', [c['reference_policy_result'] for c in cases]),
                                ('Rettilineo', [baseline[c['episode']] for c in cases])]:
            successes = [r for r in records if r['reason'] == 'goal']
            metrics = [mean(float(r[f]) for r in successes) if successes else None for f in FIELDS]
            rate = 100 * len(successes) / len(cases)
            values = [number(v, precision) if v is not None else '---' for v, precision in zip(metrics, (2, 1, 2))]
            comparison.append(' & '.join([str(index), method, number(rate, 0)] + values) + r' \\')
            group[method] = dict(successes=len(successes), success_percent=rate,
                                 **dict(zip(FIELDS, metrics)))
        paired = [c for c in cases if c['reference_policy_result']['reason'] == 'goal'
                  and baseline[c['episode']]['reason'] == 'goal']
        delta = group['Policy']['success_percent'] - group['Rettilineo']['success_percent']
        savings = extension = None
        if paired:
            pm = {f: mean(float(c['reference_policy_result'][f]) for c in paired) for f in FIELDS}
            bm = {f: mean(float(baseline[c['episode']][f]) for c in paired) for f in FIELDS}
            savings = 100 * (1 - pm['energy_j'] / bm['energy_j'])
            extension = 100 * (pm['path_m'] / bm['path_m'] - 1)
            group['paired_means'] = dict(policy=pm, straight_line=bm)
        group.update(paired_episodes=[c['episode'] for c in paired],
                     success_change_pp=delta, energy_saving_percent=savings,
                     path_extension_percent=extension)
        statistics.append(group)
        differences.append(' & '.join([str(index), str(len(paired)), number(delta, 0),
            number(savings, 2) if savings is not None else '---',
            number(extension, 2) if extension is not None else '---']) + r' \\')
        if index < 4:
            comparison.append(r'\addlinespace')
    for content, suffix in [(comparison, 'confronto_statistiche'), (differences, 'confronto_variazioni')]:
        content += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
        (HERE / f'{prefix}_{suffix}.tex').write_text('\n'.join(content))
    (HERE / f'{prefix}_rettilineo.tex').write_text('\n'.join(tables))
    provenance = dict(run=str(run.relative_to(ROOT)),
        episodes_sha256=hashlib.sha256((run / 'episodes.csv').read_bytes()).hexdigest(),
        missions_sha256=hashlib.sha256((run / 'inputs/selected_missions.json').read_bytes()).hexdigest(),
        statistics=statistics)
    (HERE / f'{prefix}_rettilineo_statistiche.json').write_text(json.dumps(provenance, indent=2) + '\n')
