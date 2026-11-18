"""Compare the current confirmed asphalt and rock results, paired by mission."""
from pathlib import Path
import hashlib
import json
import re
from statistics import mean

HERE=Path(__file__).resolve().parent


def read_tables(path):
    tables=[]
    for block in path.read_text().split(r'\begin{table}')[1:]:
        records={}
        for line in block.split(r'\end{table}')[0].splitlines():
            if not re.match(r'^\d+ &',line):continue
            cells=line.removesuffix(r' \\').split(' & ')
            episode=int(cells[0]);success=cells[-1]=='Successo'
            record=dict(episode=episode,success=success,start=cells[1],goal=cells[2])
            if success:
                for field,cell in zip(('path_m','energy_j','time_s','mean_speed_m_s'),cells[3:7]):
                    record[field]=float(cell.strip('$ ').replace('{,}', '.'))
            records[episode]=record
        if set(records)!=set(range(1,6)):raise ValueError('Expected five ordered episodes per map')
        tables.append(records)
    if len(tables)!=2:raise ValueError('Expected the two binary result tables')
    return tables


def statistics(label,asphalt,rocks):
    for episode in asphalt:
        if (asphalt[episode]['start'],asphalt[episode]['goal'])!=(rocks[episode]['start'],rocks[episode]['goal']):
            raise ValueError('Paired endpoints differ')
    common=[i for i in asphalt if asphalt[i]['success'] and rocks[i]['success']]
    if not common:raise ValueError('Expected successful pairs')
    a={field:mean(asphalt[i][field] for i in common) for field in ('path_m','energy_j','time_s')}
    r={field:mean(rocks[i][field] for i in common) for field in ('path_m','energy_j','time_s')}
    sa=sum(x['success'] for x in asphalt.values());sr=sum(x['success'] for x in rocks.values())
    return dict(group=label,missions=len(asphalt),asphalt_successes=sa,rocks_successes=sr,
        asphalt_success_percent=100*sa/len(asphalt),rocks_success_percent=100*sr/len(rocks),
        success_change_pp=100*(sa-sr)/len(asphalt),paired_episodes=common,
        asphalt_paired_means=a,rocks_paired_means=r,
        energy_saving_percent=100*(1-a['energy_j']/r['energy_j']),
        time_saving_percent=100*(1-a['time_s']/r['time_s']))


def number(value,places=2):return '$'+f'{value:.{places}f}'.replace('.','{,}')+'$'


def main():
    ap=HERE/'tabels_exp1_1.tex';rp=HERE/'binary_rocks_risultati.tex'
    asphalt=read_tables(ap);rocks=read_tables(rp)
    stats=[statistics(name,a,r) for name,a,r in zip(('A','B'),asphalt,rocks)]
    combined_a={5*group+ep:r for group,table in enumerate(asphalt) for ep,r in table.items()}
    combined_r={5*group+ep:r for group,table in enumerate(rocks) for ep,r in table.items()}
    total=statistics('Totale',combined_a,combined_r)
    lines=[r'\begin{table}[H]',r'\centering',
        r'\caption{Confronto tra asfalto e rocce per le stesse missioni. Le variazioni sono riferite alla condizione sulle rocce; energia e tempo sono confrontati soltanto sui successi comuni.}',
        r'\label{tab:exp1-1-confronto-rocks}',r'\small',r'\setlength{\tabcolsep}{4pt}',
        r'\begin{tabular}{@{}cccccc@{}}',r'\toprule',
        r'Gruppo & \shortstack{Successo\\asfalto [\%]} & \shortstack{Successo\\rocce [\%]} & \shortstack{Variazione successo\\{}[punti percentuali]} & \shortstack{Risparmio\\energetico [\%]} & \shortstack{Risparmio\\temporale [\%]} \\',r'\midrule']
    for s in [*stats,total]:
        if s['group']=='Totale':lines.append(r'\midrule')
        lines.append(' & '.join([s['group'],number(s['asphalt_success_percent'],0),number(s['rocks_success_percent'],0),
            number(s['success_change_pp'],0),number(s['energy_saving_percent']),number(s['time_saving_percent'])])+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}',''])
    (HERE/'binary_rocks_confronto.tex').write_text('\n'.join(lines))
    data=dict(sources=[dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in (ap,rp)],
        note='Descriptive comparisons of current user-confirmed displayed measurements. Paired energy/time use successful missions only. Asphalt group A was updated from a rerun; the original rock replay references are retained separately.',
        groups=stats,total=total)
    (HERE/'binary_rocks_confronto.json').write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n')
    for s in [*stats,total]:
        print(s['group'], 'paired',len(s['paired_episodes']),'energy saving',s['energy_saving_percent'],'time saving',s['time_saving_percent'])


if __name__=='__main__':main()
