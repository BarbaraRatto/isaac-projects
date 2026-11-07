"""Nine visible evaluation episodes, fixed geometry and best-checkpoint selection."""
import json
import os
from collections import Counter
from pathlib import Path
import subprocess

from train_general_big_map_test11_SAC_Re.missions_15 import SuiteSampler
from train_general_big_map_test11_SAC_Re.missions import deadline_steps

HERE=Path(__file__).resolve().parent


def test_nine_mission_endpoints_and_categories():
    sampler=SuiteSampler(HERE/'artifacts/missions_9.json')
    assert len(sampler.cases)==9
    assert Counter(c['category'] for c in sampler.cases)=={
        'vicino':2,'2_celle':2,'3_celle':2,'4_celle':2,'goal_verde':1}
    limits={'vicino':(1.5,3.),'2_celle':(5.5,6.5),'3_celle':(8.5,9.5),
            '4_celle':(11.5,12.5),'goal_verde':(3.,6.)}
    for i,c in enumerate(sampler.cases):
        m=sampler.evaluation_mission(0,i,231207)
        assert m==c['mission'] and c['episode']==i+1
        assert m['start'][4]==0
        assert m['goal'][4]==(1 if i==8 else 0)
        lo,hi=limits[c['category']]
        assert lo<=m['distance_m']<=hi
        assert m['deadline_steps']==deadline_steps(m['distance_m'])
    assert sampler.evaluation_mission(0,9,231207)==sampler.cases[0]['mission']


def test_launcher_selects_best_from_latest_completed_run(tmp_path):
    runs=tmp_path/'runs'
    for name,mtime in [('older',100),('newer',200)]:
        run=runs/name;run.mkdir(parents=True)
        for filename in ['policy_final.pt','policy_best.pt']:
            p=run/filename;p.write_text('mock checkpoint');os.utime(p,(mtime,mtime))
    wrapper=tmp_path/'isaaclab'
    wrapper.write_text('''#!/usr/bin/env python3
import json,os,sys
open(os.environ['ARGS_LOG'],'w').write(json.dumps(sys.argv[1:]))
''')
    wrapper.chmod(0o755)
    log=tmp_path/'args.json'
    result=subprocess.run(['bash',str(HERE/'test_9.sh'),'--runs-root',str(runs)],
        env=dict(os.environ,GENERAL_MAP_ISAACLAB=str(wrapper),ARGS_LOG=str(log)),capture_output=True)
    assert result.returncode==0,result.stderr
    args=json.loads(log.read_text())
    assert args[args.index('--checkpoint')+1]==str(runs/'newer/policy_best.pt')
    assert args[max(i for i,x in enumerate(args) if x=='--episodes')+1]=='9'
    assert args[max(i for i,x in enumerate(args) if x=='--num-envs')+1]=='1'
    assert Path(args[args.index('--mission-suite')+1])==HERE/'artifacts/missions_9.json'
    assert '--headless' not in args


def test_selection_skips_accidental_incomplete_run_and_follows_inherited_best(tmp_path):
    from train_general_big_map_test11_SAC_Re.select_checkpoint import select_best
    root=tmp_path/'runs'
    parent=root/'original';parent.mkdir(parents=True)
    checkpoint=parent/'policy_interrupted.pt';checkpoint.write_text('checkpoint')
    best=parent/'policy_best.pt';best.write_text('best')
    base=dict(signature={'algorithm':'SAC'},identity={'map':'lab'})
    (parent/'config.json').write_text(json.dumps(dict(**base,arguments={'resume':None},initial_transitions=0)))
    resumed=root/'completed';resumed.mkdir()
    (resumed/'policy_final.pt').write_text('final')
    (resumed/'config.json').write_text(json.dumps(dict(**base,arguments={'resume':str(checkpoint)},initial_transitions=60057)))
    accidental=root/'accidental';accidental.mkdir()
    (accidental/'policy_interrupted.pt').write_text('untrained')
    (accidental/'policy_best.pt').write_text('untrained best')
    assert select_best(root)==best
