"""Contracts specific to the grey/black-ice experiment; no simulation required."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from train_general_big_map_test13_SAC_Re.missions_15 import build_suite, SuiteSampler
from train_general_big_map_test13_SAC_Re.missions import MissionSampler
from train_general_big_map_test13_SAC_Re.config import MISSION_CFG
from train_general_big_map_test13_SAC_Re.layouts import map_metadata


def test_heldout_nine_missions_never_target_ice_and_last_crosses_it():
    suite=json.loads((HERE/'artifacts/missions_9.json').read_text())
    assert suite==build_suite()
    sampler=SuiteSampler(HERE/'artifacts/missions_9.json')
    assert len(suite['cases'])==9
    assert [c['category'] for c in suite['cases']]==['vicino']*2+['2_celle']*2+['3_celle']*2+['4_celle']*2+['deviazione_ghiaccio']
    ids=[]
    for i,case in enumerate(suite['cases']):
        mission=sampler.evaluation_mission(0,i,44)
        assert mission['start'][4]==mission['goal'][4]==0
        ids.extend([mission['start_id'],mission['goal_id']])
    assert len(ids)==len(set(ids))
    assert suite['cases'][-1]['straight_line_black_fraction']>=.2
    assert MISSION_CFG.spawn_excluded==MISSION_CFG.goal_excluded==('lab_blackice',)
    assert np.any(MissionSampler().materials==1)  # still available to PCA, never mission endpoints


def test_grey_cells_form_connected_four_neighbour_route_network():
    grid=map_metadata()['grid']
    cells={(r,c) for r,row in enumerate(grid) for c,name in enumerate(row) if name=='t1_asphalt'}
    pending=[next(iter(cells))]; seen=set(pending)
    while pending:
        r,c=pending.pop()
        for x in ((r-1,c),(r+1,c),(r,c-1),(r,c+1)):
            if x in cells and x not in seen: seen.add(x);pending.append(x)
    assert len(seen)==len(cells)==17


def test_report_and_plots_work_with_zero_successes(tmp_path):
    record=dict(env_id=0,episode=1,reason='timeout',success=0,energy_j=1.,steps=1,
                alignment_steps=0,navigation_steps=1,start_x=3.9,start_y=.9,goal_x=5.9,goal_y=.9,
                final_x=4.,final_y=.9,initial_distance_m=2.,final_distance_m=1.9,path_m=.1,
                start_terrain='t1_asphalt',goal_terrain='t1_asphalt',fall_cause='',max_tilt_deg=0.)
    summary=dict(episodes=1,successes=0,stage=3,outcomes={'timeout':1},alignment_success_rate=1.,mean_success_energy_j=None)
    (tmp_path/'summary.json').write_text(json.dumps(dict(checkpoint='policy_best.pt',episodes=[record],summary=summary)))
    def write(name,rows):
        with (tmp_path/name).open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write('episodes.csv',[dict(transitions=1500000,**record)])
    write('trajectories.csv',[dict(env_id=0,episode=1,step=0,x=3.9,y=.9,phase='navigation',goal_x=5.9,goal_y=.9,command_vx=.5)])
    write('fall_diagnostics.csv',[dict(env_id=0,episode=1,step=1,x=4.,y=.9,time_s=.2,tilt_deg=0.)])
    result=subprocess.run([sys.executable,str(HERE/'analyze_paths.py'),str(tmp_path)],capture_output=True,text=True,
                          env=dict(os.environ,MPLCONFIGDIR=str(tmp_path/'mpl')))
    assert result.returncode==0,result.stderr
    for name in ('mappa_percorsi.pdf','percorsi_per_episodio.pdf','riepilogo_episodi.csv','REPORT.md'):
        assert (tmp_path/'analysis'/name).stat().st_size>0
    assert 'nessun successo' in (tmp_path/'analysis/REPORT.md').read_text()
