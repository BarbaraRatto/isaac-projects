"""Read completed/partial benchmark logs; no simulator dependencies."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import numpy as np


def describe(values):
    return dict(n=len(values), mean=statistics.mean(values),
                sd=statistics.stdev(values) if len(values)>1 else None,
                minimum=min(values), maximum=max(values)) if values else dict(n=0)


def analyze(directory):
    rows = list(csv.DictReader((directory/'episodes.csv').open()))
    for row in rows:
        for key in ('trial','map','success','route_followed'):
            row[key] = int(row[key])
        for key in ('energy_j','discounted_energy_j','nav_return','discounted_nav_return','path_m','seconds'):
            row[key] = float(row[key])
    valid = [r for r in rows if r['success'] and r['route_followed']]
    result = dict(episodes=len(rows), valid=len(valid), complete=(directory/'complete.json').exists(),groups={})
    for map_id in (0,1):
        for material in ('asphalt','rocks'):
            all_rows = [r for r in rows if r['map']==map_id and r['material']==material]
            selected = [r for r in valid if r['map']==map_id and r['material']==material]
            result['groups'][f'map{map_id}_{material}'] = dict(
                attempted=len(all_rows),successes=sum(r['success'] for r in all_rows),valid=len(selected),
                failures={reason:sum(r['reason']==reason for r in all_rows) for reason in ('fallen','out_of_bounds','timeout')},
                energy_j=describe([r['energy_j'] for r in selected]),
                discounted_energy_j=describe([r['discounted_energy_j'] for r in selected]),
                path_m=describe([r['path_m'] for r in selected]),seconds=describe([r['seconds'] for r in selected]),
                joules_per_m=describe([r['energy_j']/r['path_m'] for r in selected]))
    pairs=[]
    for trial in sorted({r['trial'] for r in rows}):
        for map_id in (0,1):
            pair={r['material']:r for r in valid if r['trial']==trial and r['map']==map_id}
            if set(pair) != {'asphalt','rocks'}:
                continue
            a,b=pair['asphalt'],pair['rocks']
            pairs.append(dict(trial=trial,map=map_id,delta_j=b['energy_j']-a['energy_j'],
                              discounted_delta_j=b['discounted_energy_j']-a['discounted_energy_j']))
    result['paired_by_map'] = {str(i):describe([p['delta_j'] for p in pairs if p['map']==i]) for i in (0,1)}
    # Each trial is one cluster: its two map pairs share the initial jitter.
    clusters=[]; discounted=[]
    for trial in sorted({p['trial'] for p in pairs}):
        p=[p for p in pairs if p['trial']==trial]
        if len(p)==2:
            clusters.append(statistics.mean(r['delta_j'] for r in p))
            discounted.append(statistics.mean(r['discounted_delta_j'] for r in p))
    result['balanced_paired_delta_j'] = describe(clusters)
    result['balanced_discounted_delta_j'] = describe(discounted)
    if len(clusters)>1:
        indices=np.random.default_rng(13).integers(0,len(clusters),size=(10000,len(clusters)))
        result['balanced_paired_delta_j']['bootstrap_95_interval'] = np.quantile(np.asarray(clusters)[indices].mean(1),[.025,.975]).tolist()
    if clusters and statistics.mean(clusters)>0:
        result['weights_for_energy_advantage']={str(b):100*b/statistics.mean(clusters) for b in (10,25,100)}
    if discounted and statistics.mean(discounted)>0:
        result['discounted_weights_for_energy_advantage']={str(b):100*b/statistics.mean(discounted) for b in (10,25,100)}
    result['notes'] = ['Only successful, correctly followed routes enter energy comparisons.',
        'Jitter repetitions on fixed maps and fixed controller, not independent terrain realizations.',
        'Bootstrap interval clusters both mirrored maps per trial; exploratory small-sample interval.',
        'Weights match only the energy advantage; total reward also includes progress/time/terminal terms.']
    (directory/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    analyze(parser.parse_args().directory)
