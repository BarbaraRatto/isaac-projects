"""Build the two swapped-map binary result tables from saved logs."""
from pathlib import Path
import csv
import hashlib
import json
import math

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
RUN=ROOT/'spot_rl_energy_navigation/analysis_thesis_experiments/case_1_1_binary_terrain/test with rocks/results_exp/run_20261117_234938_911765'
CAUSES={'fallen':'caduta','timeout':'timeout','out_of_bounds':'out of bounds'}


def number(value,places=2):
    return '$'+f'{float(value):.{places}f}'.replace('.','{,}')+'$'


def point(x,y):
    return '$('+f'{float(x):.4f}'.replace('.','{,}')+r';\,'+f'{float(y):.4f}'.replace('.','{,}')+')$'


def main():
    confirmed=HERE/'binary_rocks_risultati_confermati.json'
    if confirmed.exists():
        snapshot=json.loads(confirmed.read_text())
        (HERE/'binary_rocks_risultati.tex').write_text(snapshot['source_table_content'])
        print('Restored the user-confirmed swapped-map result tables.')
        return
    root_complete=json.loads((RUN/'complete.json').read_text())
    tables=[];sources=[]
    groups=[('a_routes_on_b','A','B','rocks_right','rocks_left'),
            ('b_routes_on_a','B','A','rocks_left','rocks_right')]
    for group,original,target,source_map,target_map in groups:
        if group not in root_complete['groups']:raise ValueError('Missing completed group')
        directory=RUN/group
        complete=json.loads((directory/'complete.json').read_text())
        if complete.get('logging_checks')!='passed' or complete.get('completed_missions')!=5:
            raise ValueError('Expected five completed and checked missions')
        with (directory/'episodes.csv').open() as stream:
            results=list(csv.DictReader(stream))
        if [r['mission_id'] for r in results]!=[f'{original}{i}' for i in range(1,6)]:
            raise ValueError('Unexpected mission order')
        lines=[]
        for index,r in enumerate(results,1):
            if r['source_map']!=source_map or r['target_map']!=target_map:
                raise ValueError('Map pair differs from the requested comparison')
            fixed=[str(index),point(r['start_x_m'],r['start_y_m']),point(r['goal_x_m'],r['goal_y_m'])]
            if r['reason']=='goal':
                speed=float(r['path_m'])/float(r['time_s'])
                if not math.isclose(speed,float(r['mean_speed_m_s']),rel_tol=1e-8):
                    raise ValueError('Recorded mean speed is inconsistent')
                values=[number(r['path_m']),number(r['energy_j'],1),number(r['time_s'],1),number(speed),'Successo']
            else:values=['---']*4+['Fallimento ('+CAUSES[r['reason']]+')']
            lines.append(' & '.join(fixed+values)+r' \\')
        tables.append('\n'.join([
            '% Generated from the completed swapped-map binary experiment.',
            r'\begin{table}[H]',r'\centering',
            rf'\caption{{Risultati dei percorsi {original}1--{original}5 riprodotti sulla mappa {target}, lungo il ramo con le rocce.}}',
            rf'\label{{tab:exp1-1-rocks-{original.lower()}}}',
            r'\begingroup',r'\fontsize{9.5}{11.5}\selectfont',
            r'\setlength{\tabcolsep}{2.5pt}',r'\renewcommand{\arraystretch}{1.25}',
            r'\begin{tabular}{@{}c cc cccc c@{}}',r'\toprule',
            r'Ep. & \shortstack{Start $(x;y)$\\{}[m]} & \shortstack{Goal $(x;y)$\\{}[m]} & \shortstack{Distanza\\percorsa [m]} & \shortstack{Energia\\{}[J]} & \shortstack{Tempo\\{}[s]} & \shortstack{Velocità\\media [m/s]} & Esito \\',
            r'\midrule',*lines,r'\bottomrule',r'\end{tabular}',r'\endgroup',r'\end{table}','']))
        sources.append(dict(group=group,episodes_file=str((directory/'episodes.csv').relative_to(ROOT)),
            episodes_sha256=hashlib.sha256((directory/'episodes.csv').read_bytes()).hexdigest()))
    (HERE/'binary_rocks_risultati.tex').write_text('\n'.join(tables))
    (HERE/'binary_rocks_fonti.json').write_text(json.dumps(dict(run=str(RUN.relative_to(ROOT)),sources=sources),indent=2)+'\n')
    print('Generated two tables with ten recorded mission outcomes.')


if __name__=='__main__':main()
