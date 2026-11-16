"""Build twelve-metre result tables from the frozen thesis selection."""
from pathlib import Path
import hashlib
import json
import math

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SELECTION=ROOT/'spot_rl_energy_navigation/analysis_thesis_experiments/case_1_3_general_terrain/experiments/thesis experiments_12m/selected_missions_v2.json'
CATEGORIES=('regular','ice','stairs','surface_choice')
CAUSES={'fallen':'caduta','out_of_bounds':'out of bounds','timeout':'timeout','alignment_timeout':'timeout di allineamento'}


def number(value,places=2):
    return '$'+f'{float(value):.{places}f}'.replace('.','{,}')+'$'


def point(values):
    return '$('+f'{values[0]:.3f}'.replace('.','{,}')+r';\,'+f'{values[1]:.3f}'.replace('.','{,}')+')$'


def table(caption,label,rows):
    return '\n'.join([
        '% Generated from the archived twelve-metre selection; one row per episode.',
        r'\begin{table}[H]',r'\centering',r'\caption{'+caption+'}',r'\label{'+label+'}',
        r'\begingroup',r'\fontsize{9}{11}\selectfont',r'\setlength{\tabcolsep}{2.5pt}',
        r'\renewcommand{\arraystretch}{1.2}',r'\begin{tabular}{@{}c cc cccc l@{}}',r'\toprule',
        r'Ep. & \shortstack{Start $(x;y)$\\{}[m]} & \shortstack{Goal $(x;y)$\\{}[m]} & \shortstack{Distanza\\percorsa [m]} & \shortstack{Energia\\{}[J]} & \shortstack{Tempo\\{}[s]} & \shortstack{Velocità\\media [m/s]} & Esito \\',
        r'\midrule',*rows,r'\bottomrule',r'\end{tabular}',r'\endgroup',r'\end{table}',''])


def main():
    document=json.loads(SELECTION.read_text())
    source=(SELECTION.parent/document['source_collection']).resolve()
    slots=document['slots']
    if [s['thesis_episode'] for s in slots]!=list(range(1,41)):
        raise ValueError('Expected the fixed E1–E40 selection')
    stats=[];result_tables=[];baseline_tables=[]
    for index,category in enumerate(CATEGORIES,1):
        subset=[s for s in slots if s['case']['category']==category]
        if len(subset)!=10:
            raise ValueError('Expected ten cases in each category')
        rows=[];baseline=[]
        for slot in subset:
            n=slot['thesis_episode'];mission=slot['case']['mission'];result=slot['result']
            archived=json.loads((source/slot['episode_directory']/'source_result.json').read_text())
            if archived['reason']!=result['reason']:
                raise ValueError('Archived outcome differs')
            for key in ('path_m','energy_j','time_s'):
                if not math.isclose(float(archived[key]),float(result[key]),abs_tol=1e-8):
                    raise ValueError('Archived measurements differ')
            fixed=[str(n),point(mission['start']),point(mission['goal'])]
            if result['reason']=='goal':
                seconds=float(result['time_s'])
                if seconds<=0:
                    raise ValueError('Mean speed requires a positive episode duration')
                mean_speed=float(result['path_m'])/seconds
                if 'mean_speed_m_s' in result and not math.isclose(mean_speed,float(result['mean_speed_m_s']),rel_tol=1e-8):
                    raise ValueError('Recorded mean speed differs from path divided by time')
                values=[number(result['path_m']),number(result['energy_j'],1),number(seconds,1),number(mean_speed),'Successo']
            else:
                values=['---','---','---','---','Fallimento ('+CAUSES[result['reason']]+')']
            rows.append(' & '.join(fixed+values)+r' \\')
        result_tables.append(table(f'Missioni a 12~m, categoria {index}: risultati con la policy.',f'tab:big-map-12m-cat{index}',rows))
        successful=[s['result'] for s in subset if s['result']['reason']=='goal']
        stats.append(dict(category=index,category_name=category,missions=len(subset),successes=len(successful),
                          success_percent=100*len(successful)/len(subset),
                          **{key:math.fsum(float(r[field]) for r in successful)/len(successful)
                             for key,field in [('mean_path_m','path_m'),('mean_energy_j','energy_j'),('mean_time_s','time_s')]}))
    (HERE/'big_map_12m_risultati.tex').write_text('\n'.join(result_tables))
    head=[r'\begin{table}[H]',r'\centering',r'\caption{Riepilogo delle 40 missioni selezionate a 12~m. Le medie comprendono solo le esecuzioni concluse con successo.}',r'\label{tab:big-map-12m-statistiche}',r'\small',r'\setlength{\tabcolsep}{5pt}',r'\begin{tabular}{@{}cccccc@{}}',r'\toprule',r'Categoria & Successi & \shortstack{Successo\\{}[\%]} & \shortstack{Distanza media\\{}[m]} & \shortstack{Energia media\\{}[J]} & \shortstack{Tempo medio\\{}[s]} \\',r'\midrule']
    for s in stats:
        head.append(' & '.join([str(s['category']),f"{s['successes']}/10",number(s['success_percent'],0),number(s['mean_path_m']),number(s['mean_energy_j'],1),number(s['mean_time_s'],2)])+r' \\')
    head += [r'\bottomrule',r'\end{tabular}',r'\end{table}','']
    (HERE/'big_map_12m_statistiche.tex').write_text('\n'.join(head))
    from genera_tabelle_rettilineo import generate
    generate(12, table, number, point, CAUSES)
    data=dict(selection=str(SELECTION.relative_to(ROOT)),selection_sha256=hashlib.sha256(SELECTION.read_bytes()).hexdigest(),
              note='Descriptive statistics of the selected cases; all raw outcomes and failed-run measurements remain archived.',
              metrics_note='Absolute mechanical joint work; timing and work include alignment. Means computed from unrounded successful-run data.',statistics=stats)
    (HERE/'big_map_12m_statistiche.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(stats,indent=2))


if __name__=='__main__':
    main()
