"""Frozen policy/baseline pairs for the forty thesis missions at each distance."""
import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

FIXED_SPEED_M_S=0.9

HERE=Path(__file__).resolve().parent
CASE=HERE.parents[1]
PROJECT=HERE.parents[3]
TRAINING=PROJECT/'train_general_big_map_test7_SAC'
sys.path.insert(0,str(PROJECT))
from train_general_big_map_test7_SAC.config import NAV_CFG, TERRAIN_NAMES
from train_general_big_map_test7_SAC.missions import deadline_steps
MAP=CASE/'maps/real_terrains_v2_eval_01.usd'
METADATA=MAP.with_name(MAP.stem+'_metadata.json')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,data):
    path=Path(path);temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def suite_path(distance):
    return HERE/'inputs'/f'missions_{int(distance)}m.json'


def build_suite(distance):
    folder=HERE.parent/f'thesis experiments_{distance}m'
    selection=folder/('selected_missions_v2.json' if distance==12 else 'selected_missions.json')
    snapshot=json.loads(selection.read_text())
    collection=(folder/snapshot['source_collection']).resolve()
    categories=snapshot['category_order']
    sources={};cases=[]
    for slot in snapshot['slots']:
        run=Path(slot['source_run'])
        if str(run) not in sources:
            config=json.loads((run/'evaluation_config.json').read_text())
            complete=json.loads((run/'complete.json').read_text())
            if complete.get('logging_checks')!='passed' or config['map_sha256']!=sha(MAP):
                raise ValueError('Original run is incomplete or uses another map')
            original_file=run/'inputs'/config['mission_file']
            original_suite=json.loads(original_file.read_text())
            with (run/'episodes.csv').open(newline='') as stream:
                results={int(r['episode']):r for r in csv.DictReader(stream)}
            points=[c['mission'][key] for c in original_suite['cases'] for key in ('start','goal')]
            calibration=next(p for p in points if p[4]==0)
            sources[str(run)]=dict(config=config,cases={c['episode']:c for c in original_suite['cases']},
                results=results,calibration=calibration,identity=dict(path=str(run),
                config_sha256=sha(run/'evaluation_config.json'),episodes_sha256=sha(run/'episodes.csv'),
                mission_file=str(original_file),missions_sha256=sha(original_file),
                checkpoint_sha256=config['checkpoint_sha256'],simulation_seed=config.get('simulation_seed',20000)))
        original=sources[str(run)]
        result=original['results'][slot['source_episode']]
        original_case=original['cases'][slot['source_episode']]
        if slot['case']['mission']!=original_case['mission'] or original_case['distance_group_m']!=distance:
            raise ValueError('Selected mission differs from the original executed mission')
        archived=json.loads((collection/slot['episode_directory']/'source_result.json').read_text())
        for key in ('path_m','time_s','energy_j'):
            if not math.isclose(float(result[key]),float(slot['result'][key]),abs_tol=1e-8) or result[key]!=archived[key]:
                raise ValueError('Reference policy measurements differ')
        seconds=float(result['time_s']);speed=float(result['path_m'])/seconds
        if not 0<speed<=NAV_CFG.max_forward_m_s or not math.isfinite(speed):
            raise ValueError('Reference speed is outside locomotion command limits')
        category=slot.get('thesis_category',slot['case']['category'])
        cases.append(dict(episode=slot['thesis_episode'],distance_group_m=distance,category=category,
            original_category=original_case['category'],mission=copy.deepcopy(original_case['mission']),
            reference_speed_m_s=FIXED_SPEED_M_S,policy_mean_speed_m_s=speed,reference_policy_result=result,
            reference=dict(source_run=str(run),source_episode=slot['source_episode'],collection_episode=slot['episode']),
            source_calibration_point=original['calibration']))
    if len(cases)!=40 or len({s['identity']['checkpoint_sha256'] for s in sources.values()})!=1:
        raise ValueError('Expected forty missions and one reference navigation model')
    return dict(schema=1,name=f'Straight-line baseline at 0.9 m/s, {distance} m',distance_m=distance,
        source_selection=str(selection),source_selection_sha256=sha(selection),
        map_sha256=sha(MAP),metadata_sha256=sha(METADATA),category_order=categories,
        controller=dict(lookahead_m=1.0,heading_gain=1.5,max_yaw_rad_s=NAV_CFG.max_yaw_rad_s,
                        navigation_dt_s=NAV_CFG.action_dt,forward_speed='Fixed 0.9 m/s for every mission after alignment',fixed_forward_speed_m_s=FIXED_SPEED_M_S),
        calibration_point=cases[0]['source_calibration_point'],
        sources=[s['identity'] for s in sources.values()],cases=cases)


def load_suite(distance,path=None):
    if int(distance)!=12:
        raise ValueError('This experiment contains only 12 m missions')
    path=Path(path) if path else suite_path(distance)
    document=json.loads(path.read_text())
    if document['distance_m']!=int(distance) or document['map_sha256']!=sha(MAP) or document['metadata_sha256']!=sha(METADATA):
        raise ValueError('Frozen panel/map identity differs')
    if document['controller'].get('fixed_forward_speed_m_s')!=FIXED_SPEED_M_S:
        raise ValueError('Incorrect fixed-speed controller metadata')
    cases=document['cases'];categories=document['category_order']
    if [c['episode'] for c in cases]!=list(range(1,41)):
        raise ValueError('Expected thesis IDs E1–E40')
    for i,category in enumerate(categories):
        if [c['episode'] for c in cases if c['category']==category]!=list(range(i*10+1,i*10+11)):
            raise ValueError('Expected ten ordered missions per category')
    for case in cases:
        mission=case['mission'];reference=case['reference_policy_result']
        if case['distance_group_m']!=int(distance) or not math.isclose(math.dist(mission['start'][:2],mission['goal'][:2]),float(distance),abs_tol=1e-8):
            raise ValueError('Mission distance differs')
        speed=float(case['reference_speed_m_s'])
        if not 0<speed<=NAV_CFG.max_forward_m_s or not math.isclose(speed,FIXED_SPEED_M_S,rel_tol=1e-12):
            raise ValueError('Incorrect constant reference speed; expected 0.9 m/s')
        if not math.isclose(float(case['policy_mean_speed_m_s']),float(reference['path_m'])/float(reference['time_s']),rel_tol=1e-12):
            raise ValueError('Incorrect archived policy mean speed')
        if mission['deadline_steps']!=deadline_steps(float(distance)) or not math.isfinite(mission['yaw']):
            raise ValueError('Invalid deadline or heading')
        if any(not np.isfinite(mission[key]).all() for key in ('start','goal')):
            raise ValueError('Invalid endpoint')
    return document


class FixedSampler:
    def __init__(self,document):
        self.document=copy.deepcopy(document)
        self.points=np.array([document['calibration_point']]+[c['mission'][key] for c in document['cases'] for key in ('start','goal')])
        if self.points[0,4]!=0:
            raise ValueError('Standing calibration requires asphalt')

    def evaluation_mission(self,env_id,episode_index,seed,stage=0):
        if env_id!=0:
            raise ValueError('Use one robot for paired evaluation')
        return copy.deepcopy(self.document['cases'][min(episode_index,len(self.document['cases'])-1)]['mission'])

    def training(self,level):
        raise RuntimeError('Baseline evaluation only')


def export_suite(document):
    distance=document['distance_m'];path=suite_path(distance)
    if path.exists():
        raise FileExistsError('Keep the frozen panel; do not regenerate it in place')
    write_json(path,document)
    fields=['episode','category','start_x_m','start_y_m','goal_x_m','goal_y_m','initial_yaw_rad','deadline_s',
            'reference_speed_m_s','policy_path_m','policy_time_s','policy_energy_j','policy_outcome','collection_episode','source_episode','source_run']
    with path.with_suffix('.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for case in document['cases']:
            m=case['mission'];r=case['reference_policy_result'];ref=case['reference']
            writer.writerow(dict(episode=case['episode'],category=case['category'],start_x_m=m['start'][0],start_y_m=m['start'][1],
                goal_x_m=m['goal'][0],goal_y_m=m['goal'][1],initial_yaw_rad=m['yaw'],deadline_s=m['deadline_steps']*NAV_CFG.action_dt,
                reference_speed_m_s=case['reference_speed_m_s'],policy_path_m=r['path_m'],policy_time_s=r['time_s'],policy_energy_j=r['energy_j'],
                policy_outcome=r['reason'],collection_episode=ref['collection_episode'],source_episode=ref['source_episode'],source_run=ref['source_run']))
    print(path)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--build',type=int,choices=(12,),default=12)
    args=parser.parse_args();export_suite(build_suite(args.build));load_suite(args.build)
