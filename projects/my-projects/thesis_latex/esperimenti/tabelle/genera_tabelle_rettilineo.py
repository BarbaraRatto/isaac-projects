"""Build comparisons from the user-confirmed remote straight-line results."""
from pathlib import Path
import csv
import hashlib
import json
import math
from statistics import mean

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FIELDS = ('path_m', 'energy_j', 'time_s')


def generate(distance, table, number, point, causes):
    snapshot_path = HERE / f'big_map_{distance}m_rettilineo_confermato.json'
    snapshot = json.loads(snapshot_path.read_text())
    panel = snapshot['panel']
    if panel['distance_m'] != distance:
        raise ValueError('Confirmed panel distance differs')
    baseline = {c['episode']: c['confirmed_straight_line_result'] for c in panel['cases']}
    if set(baseline) != set(range(1, 41)):
        raise ValueError('Expected exactly E1 to E40')
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
            # Preserve all user-confirmed cells, including the displayed mean speed.
            entries.append(b['latex_row'])
        tables.append(table(f'Missioni a {distance}~m, categoria {index}: riferimento rettilineo con velocità comandate medie per rotazione iniziale e avanzamento.',
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
    all_cases = panel['cases']
    common = [c for c in all_cases if c['reference_policy_result']['reason'] == 'goal'
              and baseline[c['episode']]['reason'] == 'goal']
    overall = dict(policy_successes=sum(c['reference_policy_result']['reason'] == 'goal' for c in all_cases),
                   straight_successes=sum(baseline[c['episode']]['reason'] == 'goal' for c in all_cases),
                   straight_failures={reason: sum(b['reason'] == reason for b in baseline.values())
                                      for reason in ('fallen', 'timeout', 'out_of_bounds')},
                   paired_episodes=[c['episode'] for c in common])
    if common:
        pm = {f: mean(float(c['reference_policy_result'][f]) for c in common) for f in FIELDS}
        bm = {f: mean(float(baseline[c['episode']][f]) for c in common) for f in FIELDS}
        overall.update(paired_means=dict(policy=pm, straight_line=bm),
                       energy_saving_percent=100 * (1 - pm['energy_j'] / bm['energy_j']),
                       path_extension_percent=100 * (pm['path_m'] / bm['path_m'] - 1),
                       time_change_percent=100 * (pm['time_s'] / bm['time_s'] - 1))
    provenance = dict(confirmed_snapshot=str(snapshot_path.relative_to(ROOT)),
        confirmed_snapshot_sha256=hashlib.sha256(snapshot_path.read_bytes()).hexdigest(),
        precision=snapshot['precision'], overall=overall, statistics=statistics)
    (HERE / f'{prefix}_rettilineo_statistiche.json').write_text(json.dumps(provenance, indent=2) + '\n')
