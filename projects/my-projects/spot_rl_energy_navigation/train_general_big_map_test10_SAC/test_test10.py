"""Test10 contracts and real SAC orchestration with synthetic simulator doubles."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import ast
import copy
import json
import os
import signal
import subprocess
import sys
import types

import numpy as np
import pytest
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from train_general_big_map_test10_SAC.config import SAC_CFG,MISSION_CFG,OBSERVATION_SIZE
from train_general_big_map_test10_SAC.frozen_scalers import PHYSICAL,load_scalers,validate_scalers
from train_general_big_map_test10_SAC.missions import Curriculum,MissionSampler,category_schedule,evaluation_interval,deadline_steps
from train_general_big_map_test10_SAC.networks import SACPolicy
from train_general_big_map_test10_SAC.replay import ReplayBuffer
from train_general_big_map_test10_SAC.scaler_calibration import fit_scalers

torch.set_num_threads(2)


def fixture_scalers():
    return dict(schema=1,mode='frozen_final_cells_before_training',goal_scale_m=10.,clip_normalized=10.,
                visual_mean=[2.]*32,visual_scale=[3.]*32,**copy.deepcopy(PHYSICAL),
                identity={'unit_test':True},pca_sha256='unit_test',
                calibration=dict(fit_source='final_valid_grid_cells',disjoint_viewpoints=True,
                                 train_counts=[2000]*7,validation_counts=[500]*7,validation_clip_fraction=0.))


def observations(n):
    actor=torch.zeros(n,OBSERVATION_SIZE);actor[:,0]=2.;actor[:,13]=100.;actor[:,14]=1.
    cells=actor[:,15:].reshape(n,81,40);cells[:,:,:32]=2.;cells[:,:,36:38]=1.
    critic=torch.cat((actor,torch.zeros(n,24)),1)
    return actor,critic


@pytest.fixture(scope='module')
def sampler():return MissionSampler()


@pytest.mark.parametrize('stage',range(5))
def test_exact_evaluation_mix_and_independent_points(sampler,stage):
    schedule=category_schedule(stage)
    assert len(schedule)==128
    assert all(abs(schedule.count(name)-w*128)<1 for name,w in zip(MISSION_CFG.categories,MISSION_CFG.category_weights[stage]))
    assert {'direct','left','right'}<=set(schedule[:9])
    for i in range(128):
        m=sampler.evaluation_mission(i,0,710031,stage)
        assert m['category']==schedule[i]
        assert sampler.evaluation[m['start_id']] and sampler.evaluation[m['goal_id']]
        assert m['start'][4] not in (1,6) and m['goal'][4] not in (1,6)
        assert np.isfinite(m['reference_length_m'])
        assert m['deadline_steps']==deadline_steps(m['reference_length_m'])


def test_uniform_free_sampling_has_both_direct_and_hazard_routes(sampler):
    from train_general_big_map_test10_SAC.planning import RouteGrid
    planner=RouteGrid(json.loads((HERE/'artifacts/map_metadata.json').read_text()))
    rng=np.random.default_rng(12);clear=[]
    for _ in range(128):
        m=sampler.sample(3,rng,evaluation=True,category='free')
        clear.append(planner.straight_clear(m['start'][:2],m['goal'][:2],inflated=False)[0])
    assert 0<sum(clear)<128


def test_bank_corridors_and_classifications(sampler):
    from train_general_big_map_test10_SAC.planning import RouteGrid
    planner=RouteGrid(json.loads((HERE/'artifacts/map_metadata.json').read_text()))
    for key,rows in sampler.groups.items():
        a,b=rows[:,0].astype(int),rows[:,1].astype(int)
        evaluation=key.startswith('evaluation')
        assert np.all(sampler.evaluation[a]==evaluation) and np.all(sampler.evaluation[b]==evaluation)
        assert np.isfinite(rows[:,2]).all()
        for index in np.linspace(0,len(rows)-1,min(20,len(rows))).astype(int):
            clear=planner.straight_clear(sampler.points[a[index],:2],sampler.points[b[index],:2],inflated=False)[0]
            if key.endswith(('right','left')):assert not clear
            if key.endswith(('direct','energy')):assert clear


def test_promotion_needs_every_category_and_two_consecutive_evaluations():
    c=Curriculum()
    good={key:dict(episodes=32,success_rate=1.) for key in ('direct','left','right')}
    assert not c.update(good)
    bad=copy.deepcopy(good);bad['right']['success_rate']=.79
    assert not c.update(bad) and c.streak==0
    assert not c.update(good)
    assert c.update(good) and c.level==1
    for stage,expected in enumerate((10000,10000,20000,30000,30000)):
        assert evaluation_interval(stage)==expected
    assert evaluation_interval(4,50000)==50000 and evaluation_interval(2,0)==0
    c=Curriculum(2)
    assert not c.update(good) # missing energy/free cannot pass


def test_all_scalers_fixed_masked_and_embedded():
    policy=SACPolicy(fixture_scalers());a,c=observations(4)
    before={key:v.clone() for key,v in policy.normalizer.state_dict().items()}
    for _ in range(10):policy.normalizer.update(a*99,c*99)
    for key,value in before.items():assert torch.equal(value,policy.normalizer.state_dict()[key])
    normalized=policy.normalizer.actor(a)
    torch.testing.assert_close(normalized[:,0],a[:,0]/10)
    torch.testing.assert_close(normalized[:,13],a[:,13]/400)
    cells=normalized[:,15:].reshape(4,81,40)
    assert not cells[:,:,:32].any() # calibrated mean of 2, not a running estimate
    a[:,15:]=0
    assert not policy.normalizer.actor(a)[:,15:].any()
    broken=fixture_scalers();broken['state_scale'][0]=99
    with pytest.raises(ValueError,match='Physical reference'):load_scalers(data=broken)
    a,c=observations(1);a[:,13]=400.*11
    assert policy.normalizer.clipping(a,c)['clip_state']==pytest.approx(1/7)


def test_calibration_fits_cells_and_excludes_validation_from_statistics():
    rng=np.random.default_rng(1)
    train=rng.normal(2,3,(17500,32));labels=np.repeat(np.arange(7),2000)
    labels=np.r_[labels,np.full(3500,7)]
    valid=rng.normal(2,3,(4375,32));vlabels=np.r_[np.repeat(np.arange(7),500),np.full(875,7)]
    artifact=fit_scalers((train,labels),(valid,vlabels),{'id':1},'pca',{})
    np.testing.assert_allclose(artifact['visual_mean'],train.mean(0))
    np.testing.assert_allclose(artifact['visual_scale'],train.std(0))
    validate_scalers(artifact,{'id':1},'pca')


@pytest.mark.parametrize('interrupt',[False,True])
def test_real_train_save_resume_with_fake_simulator(tmp_path,monkeypatch,interrupt):
    from train_general_big_map_test10_SAC import train,evaluation,frozen_scalers
    from train_general_big_map_test10_SAC.artifacts import source_identity
    from train_general_big_map_test10_SAC.perception import file_sha256
    from train_general_big_map_test10_SAC.config import DEFAULT_PCA
    from train_general_big_map_test10_SAC.layouts import training_conditions
    scalers=fixture_scalers();scalers['identity']=source_identity();scalers['pca_sha256']=file_sha256(DEFAULT_PCA)
    original=frozen_scalers.load_scalers
    monkeypatch.setattr(frozen_scalers,'load_scalers',lambda path=None,data=None: original(data=data or scalers))
    sent=[]
    class App:
        def is_running(self):return True
        def close(self,**kwargs):pass
    class Launcher:
        def __init__(self,args):self.app=App()
        @staticmethod
        def add_app_launcher_args(parser):
            parser.add_argument('--device',default='cpu');parser.add_argument('--headless',action='store_true')
    class Env:
        device='cpu'
        def __init__(self,cfg,conditions,**kwargs):
            self.num_envs=cfg.scene.num_envs;self.conditions=conditions;self.curriculum=Curriculum();self.t=0
        def reset(self,**kwargs):
            self.t=0;a,c=observations(self.num_envs);return dict(policy=a,critic=c),{}
        def step(self,actions):
            self.t+=1
            if interrupt and self.t==2 and not sent:
                os.kill(os.getpid(),signal.SIGTERM);sent.append(True)
            a,c=observations(self.num_envs)
            return dict(policy=a,critic=c),None,torch.zeros(self.num_envs,dtype=torch.bool),torch.zeros(self.num_envs,dtype=torch.bool),dict(
                policy_mask=torch.tensor([self.t>1]+[True]*(self.num_envs-1)),applied_actions=actions.clone(),
                reward_components=torch.ones(self.num_envs,2),episodes=[])
        def close(self):pass
    class Cfg:
        def __init__(self):self.sim=SimpleNamespace(device='cpu');self.scene=SimpleNamespace(num_envs=2)
    app_module=types.ModuleType('isaaclab.app');app_module.AppLauncher=Launcher
    env_module=types.ModuleType('train_general_big_map_test10_SAC.env');env_module.GeneralMapEnv=Env
    scene_module=types.ModuleType('train_general_big_map_test10_SAC.scene');scene_module.GeneralMapCfg=Cfg
    monkeypatch.setitem(sys.modules,'isaaclab.app',app_module)
    monkeypatch.setitem(sys.modules,'train_general_big_map_test10_SAC.env',env_module)
    monkeypatch.setitem(sys.modules,'train_general_big_map_test10_SAC.scene',scene_module)
    monkeypatch.setattr(train,'SAC_CFG',replace(SAC_CFG,num_envs=2,buffer_size=32,batch_size=2,learning_starts=2,log_every=2))
    def evaluate(env,policy,device,**kwargs):
        categories={key:dict(episodes=2,success_rate=1.) for key,w in zip(MISSION_CFG.categories,MISSION_CFG.category_weights[env.curriculum.level]) if w>0}
        return dict(stage=env.curriculum.level,episodes=2,success_rate=1.,mean_success_energy_j=1.,by_category=categories,by_band={}),[],(1.,-1.)
    monkeypatch.setattr(evaluation,'evaluate',evaluate)
    output=tmp_path/'run'
    monkeypatch.setattr(sys,'argv',['train.py','--timesteps','9','--run-dir',str(output),'--num-envs','2','--gradient-steps','1','--eval-every','4','--checkpoint-every','4','--headless'])
    train.main()
    summary=json.loads((output/'summary.json').read_text());expected=3 if interrupt else 9
    assert summary['transitions']==expected and summary['interrupted']==interrupt
    state=torch.load(output/('policy_interrupted.pt' if interrupt else 'policy_final.pt'),weights_only=False)
    assert state['scalers']==scalers
    assert (output/'evaluation_free_final.json').exists()!=interrupt
    replay=ReplayBuffer(32);replay.load(output/state['training_state']['replay_snapshot']['file'])
    assert replay.total==expected
    resumed=tmp_path/'resumed'
    monkeypatch.setattr(sys,'argv',['train.py','--timesteps','12','--run-dir',str(resumed),'--resume',str(output),'--num-envs','2','--gradient-steps','1','--eval-every','0','--headless'])
    train.main()
    restored=torch.load(resumed/'policy_final.pt',weights_only=False)
    assert restored['learner']['transitions']==12 and restored['scalers']==scalers
    replay.load(resumed/restored['training_state']['replay_snapshot']['file'])
    assert replay.total==12


def test_source_syntax_and_no_dependence_on_other_experiment_code():
    for path in HERE.glob('*.py'):
        ast.parse(path.read_text(),filename=str(path))
        if not path.name.startswith('test_'):
            assert 'from train_general_big_map_test9_SAC' not in path.read_text()


@pytest.mark.parametrize('resume',[False,True])
def test_launcher_prepares_then_calibrates_only_fresh_runs(tmp_path,resume):
    import shutil
    shutil.copy2(HERE/'train.sh',tmp_path/'train.sh');(tmp_path/'artifacts').mkdir()
    launcher=tmp_path/'fake-python'
    launcher.write_text('''#!/usr/bin/env python3
import json,os,sys
from pathlib import Path
with open(os.environ['TEST10_CALLS'],'a') as stream:stream.write(json.dumps(sys.argv[1:])+'\\n')
if sys.argv[2].endswith('calibrate_scalers.py'):
    (Path(sys.argv[2]).parent/'artifacts/frozen_scalers.json').write_text('{}')
''')
    launcher.chmod(0o755)
    env=dict(os.environ,GENERAL_MAP_ISAACLAB=str(launcher),TEST10_CALLS=str(tmp_path/'calls.jsonl'))
    args=['bash',str(tmp_path/'train.sh'),'--device','cuda:1']
    if resume:args+=['--resume','/tmp/old checkpoint.pt']
    subprocess.run(args,env=env,check=True,capture_output=True)
    calls=[json.loads(line) for line in (tmp_path/'calls.jsonl').read_text().splitlines()]
    names=[Path(c[1]).name for c in calls]
    assert names==(['prepare_assets.py','train.py'] if resume else
                   ['prepare_assets.py','calibrate_scalers.py','check_setup.py','train.py'])
    assert calls[-1][-2:]==(['--resume','/tmp/old checkpoint.pt'] if resume else ['--device','cuda:1'])
