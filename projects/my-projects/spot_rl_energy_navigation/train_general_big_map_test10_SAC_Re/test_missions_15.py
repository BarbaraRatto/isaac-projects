"""Verify the actual 15-episode plan and launch arguments without simulation."""
import json
import hashlib
import os
from pathlib import Path
import subprocess
from collections import Counter

import numpy as np

from train_general_big_map_test10_SAC_Re.missions_15 import build_suite, SuiteSampler, DEFAULT_SUITE
from train_general_big_map_test10_SAC_Re.config import NAV_CFG

HERE=Path(__file__).resolve().parent


def test_suite_is_reproducible_and_has_four_distance_groups_plus_one_green():
    doc=build_suite()
    assert doc==build_suite()
    assert doc==json.loads(DEFAULT_SUITE.read_text())
    assert Counter(c['category'] for c in doc['cases'])==dict(vicino=4,**{'2_celle':4,'3_celle':3,'4_celle':3,'goal_verde':1})
    ranges={'vicino':(1.5,3.),'2_celle':(5.5,6.5),'3_celle':(8.5,9.5),
            '4_celle':(11.5,12.5),'goal_verde':(3.,6.)}
    for i,case in enumerate(doc['cases']):
        m=case['mission']
        assert case['episode']==i+1
        assert m['start'][4]==0
        assert m['goal'][4]==(1 if i==14 else 0)
        d=np.linalg.norm(np.array(m['start'][:2])-m['goal'][:2])
        assert ranges[case['category']][0]<=d<=ranges[case['category']][1]
        assert d==m['distance_m']
        assert m['deadline_steps']<=NAV_CFG.max_episode_steps
    assert len({c['mission']['start_id'] for c in doc['cases']})==15
    assert len({c['mission']['goal_id'] for c in doc['cases']})==15


def test_suite_sampler_sequence_is_independent_of_curriculum_and_safe_at_final_autoreset():
    sampler=SuiteSampler()
    for i,case in enumerate(sampler.cases):
        assert sampler.evaluation_mission(0,i,99,0)==case['mission']
        assert sampler.evaluation_mission(0,i,100,3)==case['mission']
    assert sampler.evaluation_mission(0,15,99,0)==sampler.cases[0]['mission']


def test_test15_launcher_uses_newest_final_and_fifteen_visible_episodes(tmp_path):
    wrapper=tmp_path/'isaaclab'
    wrapper.write_text('''#!/usr/bin/env python3
import json,os,sys
open(os.environ['ARGS_LOG'],'w').write(json.dumps(sys.argv[1:]))
''')
    wrapper.chmod(0o755)
    log=tmp_path/'args.json'
    checkpoint=tmp_path/'policy.pt'
    result=subprocess.run(['bash',str(HERE/'test_15.sh'),'--checkpoint',str(checkpoint)],
        env=dict(os.environ,GENERAL_MAP_ISAACLAB=str(wrapper),ARGS_LOG=str(log)),capture_output=True)
    assert result.returncode==0,result.stderr
    args=json.loads(log.read_text())
    assert Path(args[1]).name=='play.py'
    # Last CLI values win over test.sh defaults.
    assert args[max(i for i,x in enumerate(args) if x=='--episodes')+1]=='15'
    assert args[max(i for i,x in enumerate(args) if x=='--num-envs')+1]=='1'
    assert Path(args[args.index('--mission-suite')+1])==DEFAULT_SUITE
    assert '--headless' not in args


def test_best9_is_exact_subset_including_yaw_deadline_and_last_green_goal():
    source=json.loads(DEFAULT_SUITE.read_text())
    path=HERE/'artifacts/missions_9_best.json'
    doc=json.loads(path.read_text())
    assert doc['source_suite_sha256']==hashlib.sha256(DEFAULT_SUITE.read_bytes()).hexdigest()
    assert [c['source_episode'] for c in doc['cases']]==[1,2,5,6,9,10,12,13,15]
    assert Counter(c['category'] for c in doc['cases'])==dict(vicino=2,**{'2_celle':2,'3_celle':2,'4_celle':2,'goal_verde':1})
    sampler=SuiteSampler(path)
    for i,c in enumerate(doc['cases']):
        original=source['cases'][c['source_episode']-1]
        assert c['mission']==original['mission']
        assert sampler.evaluation_mission(0,i,20000)==original['mission']
    assert doc['cases'][-1]['mission']['goal'][4]==1


def test_best9_launcher_selects_best_from_latest_completed_training(tmp_path):
    wrapper=tmp_path/'isaaclab'
    wrapper.write_text('''#!/usr/bin/env python3
import json,os,sys
open(os.environ['ARGS_LOG'],'w').write(json.dumps(sys.argv[1:]))
''')
    wrapper.chmod(0o755)
    log=tmp_path/'args.json'
    result=subprocess.run(['bash',str(HERE/'test_best_9.sh')],
        env=dict(os.environ,GENERAL_MAP_ISAACLAB=str(wrapper),ARGS_LOG=str(log)),capture_output=True)
    assert result.returncode==0,result.stderr
    args=json.loads(log.read_text())
    actual=Path(args[args.index('--checkpoint')+1])
    latest=max((HERE/'runs').glob('*/policy_final.pt'),key=lambda p:p.stat().st_mtime)
    assert actual==latest.with_name('policy_best.pt') and actual.is_file()
    assert args[max(i for i,x in enumerate(args) if x=='--episodes')+1]=='9'
    assert Path(args[args.index('--mission-suite')+1]).name=='missions_9_best.json'
