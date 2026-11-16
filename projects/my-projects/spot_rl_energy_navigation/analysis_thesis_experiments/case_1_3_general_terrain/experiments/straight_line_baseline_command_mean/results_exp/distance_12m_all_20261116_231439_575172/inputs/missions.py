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
    if int(distance)!=12:
        raise ValueError('This experiment contains only 12 m missions')
    from command_means import add_command_means
    source=HERE.parent/'straight_line_baseline/inputs/missions_12m.json'
    document=json.loads(source.read_text())
    return add_command_means(document, source)


def load_suite(distance,path=None):
    if int(distance)!=12:
        raise ValueError('This experiment contains only 12 m missions')
    path=Path(path) if path else suite_path(distance)
    document=json.loads(path.read_text())
    if document['distance_m']!=int(distance) or document['map_sha256']!=sha(MAP) or document['metadata_sha256']!=sha(METADATA):
        raise ValueError('Frozen panel/map identity differs')
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
        if not 0<speed<=NAV_CFG.max_forward_m_s or not math.isclose(speed,case['command_means']['navigation_forward_m_s'],rel_tol=1e-12):
            raise ValueError('Incorrect constant reference speed')
        if mission['deadline_steps']!=deadline_steps(float(distance)) or not math.isfinite(mission['yaw']):
            raise ValueError('Invalid deadline or heading')
        if any(not np.isfinite(mission[key]).all() for key in ('start','goal')):
            raise ValueError('Invalid endpoint')
    from command_means import validate_command_means
    validate_command_means(document)
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
            'reference_speed_m_s','alignment_mean_wz_rad_s','alignment_mean_abs_wz_rad_s','policy_path_m','policy_time_s','policy_energy_j','policy_outcome','collection_episode','source_episode','source_run']
    with path.with_suffix('.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for case in document['cases']:
            m=case['mission'];r=case['reference_policy_result'];ref=case['reference']
            writer.writerow(dict(episode=case['episode'],category=case['category'],start_x_m=m['start'][0],start_y_m=m['start'][1],
                goal_x_m=m['goal'][0],goal_y_m=m['goal'][1],initial_yaw_rad=m['yaw'],deadline_s=m['deadline_steps']*NAV_CFG.action_dt,
                reference_speed_m_s=case['reference_speed_m_s'],alignment_mean_wz_rad_s=case['command_means']['alignment_yaw_rad_s'],alignment_mean_abs_wz_rad_s=case['command_means']['alignment_abs_yaw_rad_s'],policy_path_m=r['path_m'],policy_time_s=r['time_s'],policy_energy_j=r['energy_j'],
                policy_outcome=r['reason'],collection_episode=ref['collection_episode'],source_episode=ref['source_episode'],source_run=ref['source_run']))
    print(path)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--build',type=int,choices=(12,),default=12)
    args=parser.parse_args();export_suite(build_suite(args.build));load_suite(args.build)
