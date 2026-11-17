"""Frozen trajectories and commands for the swapped binary maps."""
import csv
import json
import math
from pathlib import Path
import sys
from binary_logging import file_hash, write_json

HERE=Path(__file__).resolve().parent
CASE=HERE.parent
PROJECT=HERE.parents[2]
sys.path.insert(0,str(PROJECT))
TRAINING=PROJECT/'multi_terrain_test13'
SOURCE=CASE/'experiments/results_exp/run_20261112_184253_967025'
PANEL=HERE/'inputs/reference_missions.json'
GROUPS=('a_routes_on_b','b_routes_on_a')


def rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def build():
    if PANEL.exists():
        raise FileExistsError('Keep the frozen references')
    cases=[]
    for source_map,target_map,group in [('rocks_right','rocks_left',GROUPS[0]),
                                        ('rocks_left','rocks_right',GROUPS[1])]:
        folder=SOURCE/source_map
        complete=json.loads((folder/'complete.json').read_text())
        if complete.get('logging_checks')!='passed':
            raise ValueError('Original reference log checks failed')
        config=json.loads((folder/'run_config.json').read_text())
        traces=rows(folder/'trajectories.csv')
        for result in rows(folder/'episodes.csv'):
            name=result['mission_id'];trace=[r for r in traces if r['mission_id']==name]
            mission={key:result[key] for key in ('mission_id','map','start_x_m','start_y_m','goal_x_m','goal_y_m','start_yaw_deg')}
            for key in ('start_x_m','start_y_m','goal_x_m','goal_y_m','start_yaw_deg'):
                mission[key]=float(mission[key])
            mission.update(map=target_map,source_map=source_map,target_map=target_map,
                source_energy_j=float(result['energy_j']),source_path_m=float(result['path_m']),
                source_time_s=float(result['time_s']))
            trace_file=HERE/'inputs'/f'{name}_reference.csv'
            with trace_file.open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(trace[0]));writer.writeheader();writer.writerows(trace)
            cases.append(dict(group=group,mission=mission,reference_file=trace_file.name,
                reference_sha256=file_hash(trace_file),source_result=result,
                source=dict(run=str(folder),trajectory_sha256=file_hash(folder/'trajectories.csv'),
                    episodes_sha256=file_hash(folder/'episodes.csv'),config=config)))
    maps={name:dict(path=str(TRAINING/'maps'/f'{name}.usd'),
                   sha256=file_hash(TRAINING/'maps'/f'{name}.usd')) for name in ('rocks_right','rocks_left')}
    document=dict(schema=1,name='Binary reference routes and forward commands on the opposite map',
        groups=list(GROUPS),physics_dt_s=.005,action_dt_s=.2,goal_radius_m=.55,timeout_s=40.,
        maps=maps,cases=cases)
    write_json(PANEL,document)
    fields=['mission_id','group','source_map','target_map','start_x_m','start_y_m','goal_x_m','goal_y_m',
            'start_yaw_deg','source_energy_j','source_path_m','source_time_s']
    with (HERE/'inputs/missions.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for case in cases:
            writer.writerow(dict(group=case['group'],**{k:case['mission'][k] for k in fields if k!='group'}))
    load()
    print(PANEL)


def load():
    from multi_terrain_test13.config import NAV_CFG
    document=json.loads(PANEL.read_text())
    if len(document['cases'])!=10 or {c['mission']['mission_id'] for c in document['cases']}!={f'{p}{n}' for p in ('A','B') for n in range(1,6)}:
        raise ValueError('Expected A1–A5 and B1–B5')
    if document['action_dt_s']!=NAV_CFG.action_dt or document['physics_dt_s']!=NAV_CFG.physics_dt:
        raise ValueError('Reference timing differs from the controller')
    for name,identity in document['maps'].items():
        if file_hash(identity['path'])!=identity['sha256']:
            raise ValueError(f'Map changed: {name}')
    for case in document['cases']:
        path=HERE/'inputs'/case['reference_file']
        if file_hash(path)!=case['reference_sha256']:
            raise ValueError('Frozen reference changed')
        trace=rows(path);mission=case['mission'];result=case['source_result']
        if mission['source_map']==mission['target_map']:
            raise ValueError('The terrain map must be swapped')
        if result['reason']!='goal' or len(trace)!=int(result['steps'])+1:
            raise ValueError('Reference must be a complete successful mission')
        for index,row in enumerate(trace):
            if row['mission_id']!=mission['mission_id'] or int(row['step'])!=index:
                raise ValueError('Reference sample ordering differs')
            if not math.isclose(float(row['time_s']),index*NAV_CFG.action_dt,abs_tol=1e-8):
                raise ValueError('Reference timing differs')
            if int(row['terminal'])!=int(index==len(trace)-1):
                raise ValueError('Reference terminal sample differs')
            if not all(math.isfinite(float(row[k])) for k in ('x_m','y_m','yaw_rad','command_forward_m_s','command_yaw_rad_s')):
                raise ValueError('Nonfinite reference sample')
            if not 0<=float(row['command_forward_m_s'])<=NAV_CFG.max_forward_m_s:
                raise ValueError('Reference forward command exceeds limits')
        for key,source_key in [('energy_j','energy_j'),('path_m','path_m')]:
            if not math.isclose(float(trace[-1][key]),float(result[source_key]),abs_tol=.001):
                raise ValueError('Reference terminal metrics differ')
    return document


if __name__=='__main__':
    build()
