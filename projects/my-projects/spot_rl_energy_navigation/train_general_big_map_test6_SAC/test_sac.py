"""Numerical SAC and replay integration checks; never starts Isaac Sim."""
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_general_big_map_test6_SAC.config import SAC_CFG, NAV_CFG, OBSERVATION_SIZE
from train_general_big_map_test6_SAC.networks import ExportedActor
from train_general_big_map_test6_SAC.replay import ReplayBuffer, collect_step
from train_general_big_map_test6_SAC.sac import SAC, soft_bellman_target
from train_general_big_map_test6_SAC.test_offline import synthetic_observations, pca_fixture
from train_general_big_map_test6_SAC.artifacts import save_checkpoint

torch.set_num_threads(2)


def test_target_uses_min_q_entropy_and_cuts_every_terminal():
    result=soft_bellman_target(torch.tensor([3.,7.]),torch.tensor([False,True]),
        torch.tensor([10.,10000.]),torch.tensor([8.,20000.]),torch.tensor([-2.,-900.]),.5,.9)
    torch.testing.assert_close(result,torch.tensor([11.1,7.]))


def test_replay_wrap_raw_privileges_and_snapshot(tmp_path):
    replay=ReplayBuffer(5)
    for start,count in ((0,3),(3,4)):
        a,c=synthetic_observations(count)
        a[:,0]=torch.arange(start,start+count)
        c[:,:OBSERVATION_SIZE]=a
        c[:,OBSERVATION_SIZE:]=42+a[:,:1]
        replay.add(a,c,torch.zeros(count,2),a[:,0],torch.zeros(count,dtype=torch.bool),a+1,c+1)
    assert replay.size==5 and replay.position==2 and replay.total==7
    assert sorted(replay.arrays['actor'][:,0])==[2,3,4,5,6]
    replay.save(tmp_path/'replay.npz')
    restored=ReplayBuffer(5);restored.load(tmp_path/'replay.npz')
    for key in replay.arrays:
        np.testing.assert_array_equal(replay.arrays[key],restored.arrays[key])
    batch=restored.sample(20,'cpu')
    torch.testing.assert_close(batch['critic'][:,:OBSERVATION_SIZE],batch['actor'])
    torch.testing.assert_close(batch['critic'][:,OBSERVATION_SIZE:],(42+batch['actor'][:,:1]).expand(-1,24))
    torch.testing.assert_close(batch['next_actor'],batch['actor']+1)
    with pytest.raises(ValueError,match='capacity'):
        ReplayBuffer(6).load(tmp_path/'replay.npz')


class FakeEnv:
    def __init__(self):
        self.t=0

    def observation(self):
        a=torch.zeros(2,OBSERVATION_SIZE)
        a[:,0]=torch.tensor([self.t,100+self.t])
        return dict(policy=a,critic=torch.cat((a,torch.zeros(2,24)),1))

    def step(self,actions):
        masks=[torch.tensor([False,True]),torch.tensor([True,False]),torch.tensor([True,True])]
        mask=masks[self.t]
        self.t+=1
        return self.observation(),None,torch.tensor([False,self.t==3]),torch.zeros(2,dtype=torch.bool),dict(
            policy_mask=mask,applied_actions=actions.clone(),
            reward_components=torch.tensor([[10.,2.],[20.,3.]]),episodes=[])


def test_collection_excludes_alignment_exact_budget_and_applied_actions():
    observed=[]
    policy=SimpleNamespace(act=lambda a,c:(torch.zeros(len(a),2),),
        normalizer=SimpleNamespace(update=lambda a,c:observed.extend(a[:,0].tolist())))
    env=FakeEnv();obs=env.observation();replay=ReplayBuffer(10)
    for remaining in (3,2,1):
        obs,stats=collect_step(env,policy,obs,replay,'cpu',remaining)
    assert replay.total==3 and stats['discarded']==1
    assert observed==[100,1,2]
    assert replay.arrays['actor'][:3,0].tolist()==[100,1,2]
    assert replay.arrays['next_actor'][:3,0].tolist()==[101,2,3]
    np.testing.assert_allclose(replay.arrays['rewards'][:3],[20-3*NAV_CFG.energy_weight,10-2*NAV_CFG.energy_weight,10-2*NAV_CFG.energy_weight])


def test_sac_update_targets_independent_critics_actor_and_temperature():
    torch.manual_seed(7)
    learner=SAC(settings=replace(SAC_CFG,batch_size=8))
    a,c=synthetic_observations(8)
    learner.policy.normalizer.update(a,c)
    replay=ReplayBuffer(16)
    replay.add(a,c,torch.zeros(8,2),torch.arange(8,dtype=torch.float32),
        torch.tensor([True,False]*4),a,c)
    batch=replay.sample(8,'cpu')
    actor_before={k:v.clone() for k,v in learner.policy.actor.state_dict().items()}
    q1_before={k:v.clone() for k,v in learner.policy.q1.state_dict().items()}
    target_before={k:v.clone() for k,v in learner.target_q1.state_dict().items()}
    metrics=learner.update(batch)
    assert all(np.isfinite(v) for v in metrics.values())
    assert metrics['alpha']>0 and learner.updates==1
    assert any(not torch.equal(v,learner.policy.actor.state_dict()[k]) for k,v in actor_before.items())
    assert any(not torch.equal(v,learner.policy.q1.state_dict()[k]) for k,v in q1_before.items())
    for key,now in learner.target_q1.state_dict().items():
        torch.testing.assert_close(now,target_before[key]*(1-SAC_CFG.tau)+learner.policy.q1.state_dict()[key]*SAC_CFG.tau)
    actor_ids={id(p) for p in learner.policy.actor.parameters()}
    q1_ids={id(p) for p in learner.policy.q1.parameters()}
    q2_ids={id(p) for p in learner.policy.q2.parameters()}
    assert not actor_ids&q1_ids and not actor_ids&q2_ids and not q1_ids&q2_ids
    assert all(p.grad is None for p in learner.target_q1.parameters())
    assert all(p.grad is None for p in learner.critic_parameters)


def test_sac_checkpoint_export_privileges_and_next_update(tmp_path):
    learner=SAC(settings=replace(SAC_CFG,batch_size=8))
    a,c=synthetic_observations(8)
    learner.policy.normalizer.update(a,c)
    replay=ReplayBuffer(16)
    replay.add(a,c,torch.zeros(8,2),torch.ones(8),torch.ones(8,dtype=torch.bool),a,c)
    learner.update(replay.sample(8,'cpu'))
    learner.transitions=8
    path=tmp_path/'policy.pt'
    save_checkpoint(path,learner,pca_fixture(),{'test':True},training_state={'curriculum':{'level':1,'streak':0}})
    state=torch.load(path,weights_only=False)
    restored=SAC(settings=learner.cfg);restored.load_state_dict(state['learner'])
    assert restored.transitions==8 and restored.updates==1
    deterministic=learner.policy.act(a,c,True)[0]
    changed=c.clone();changed[:,OBSERVATION_SIZE:]+=1000
    torch.testing.assert_close(deterministic,restored.policy.act(a,changed,True)[0])
    exported=torch.jit.trace(ExportedActor(restored.policy),torch.zeros(1,OBSERVATION_SIZE))
    torch.testing.assert_close(exported(a),deterministic)
    batch=replay.sample(8,'cpu')
    torch.manual_seed(14);first=learner.update(batch)
    torch.manual_seed(14);second=restored.update(batch)
    for key in first:
        assert first[key]==pytest.approx(second[key],rel=1e-5,abs=1e-5)


def test_raw_replay_reuses_current_normalizer_without_reinserting_statistics():
    learner=SAC(settings=replace(SAC_CFG,batch_size=8))
    a,c=synthetic_observations(8)
    learner.policy.normalizer.update(a,c)
    replay=ReplayBuffer(16)
    replay.add(a,c,torch.zeros(8,2),torch.ones(8),torch.ones(8,dtype=torch.bool),a,c)
    count=learner.policy.normalizer.state.count.clone()
    learner.update(replay.sample(8,'cpu'))
    torch.testing.assert_close(count,learner.policy.normalizer.state.count)
    np.testing.assert_array_equal(replay.arrays['actor'][:8],a.numpy())


def test_training_orchestration_logs_checkpoints_and_matching_replay_without_simulator(tmp_path,monkeypatch):
    """Exercise the actual train.main loop with a fake app/env and real SAC updates."""
    import json
    import types
    from train_general_big_map_test6_SAC import train, sac, evaluation
    from train_general_big_map_test6_SAC.missions import Curriculum
    from train_general_big_map_test6_SAC.layouts import training_conditions

    class App:
        def is_running(self):return True
        def close(self,**kwargs):pass

    class Launcher:
        def __init__(self,args):self.app=App()
        @staticmethod
        def add_app_launcher_args(parser):
            parser.add_argument('--device',default='cpu')
            parser.add_argument('--headless',action='store_true')

    class Env:
        device='cpu'
        num_envs=2
        def __init__(self,cfg,conditions,**kwargs):
            self.conditions=conditions
            self.curriculum=Curriculum()
            self.t=0
        def reset(self,**kwargs):
            self.t=0
            a,c=synthetic_observations(2)
            return dict(policy=a,critic=c),{}
        def step(self,actions):
            self.t+=1
            a,c=synthetic_observations(2)
            return dict(policy=a,critic=c),None,torch.zeros(2,dtype=torch.bool),torch.zeros(2,dtype=torch.bool),dict(
                policy_mask=torch.tensor([self.t>1,True]),applied_actions=actions.clone(),
                reward_components=torch.ones(2,2),episodes=[])
        def close(self):pass

    class Cfg:
        def __init__(self):
            self.sim=SimpleNamespace(device='cpu')
            self.scene=SimpleNamespace(num_envs=2)

    app_module=types.ModuleType('isaaclab.app');app_module.AppLauncher=Launcher
    env_module=types.ModuleType('train_general_big_map_test6_SAC.env');env_module.GeneralMapEnv=Env
    scene_module=types.ModuleType('train_general_big_map_test6_SAC.scene');scene_module.GeneralMapCfg=Cfg
    monkeypatch.setitem(sys.modules,'isaaclab.app',app_module)
    monkeypatch.setitem(sys.modules,'train_general_big_map_test6_SAC.env',env_module)
    monkeypatch.setitem(sys.modules,'train_general_big_map_test6_SAC.scene',scene_module)
    settings=replace(SAC_CFG,num_envs=2,buffer_size=32,batch_size=4,learning_starts=4,log_every=4)
    monkeypatch.setattr(train,'SAC_CFG',settings)
    original_sac=sac.SAC
    monkeypatch.setattr(sac,'SAC',lambda device:original_sac(device,settings))

    def evaluate(env,policy,device,**kwargs):
        return dict(stage=0,episodes=2,success_rate=0.,mean_success_energy_j=None,
                    by_band={'0':{'success_rate':0.}}),[],(0.,float('-inf'))
    monkeypatch.setattr(evaluation,'evaluate',evaluate)
    output=tmp_path/'run'
    monkeypatch.setattr(sys,'argv',['train.py','--timesteps','9','--run-dir',str(output),
                                   '--checkpoint-every','4','--eval-every','4','--headless'])
    train.main()
    summary=json.loads((output/'summary.json').read_text())
    assert summary['transitions']==9 and summary['updates']>0
    checkpoint=torch.load(output/'policy_final.pt',weights_only=False)
    archive=output/checkpoint['training_state']['replay_snapshot']['file']
    replay=ReplayBuffer(32);replay.load(archive)
    assert replay.total==9
    assert (output/'actor.pt').is_file() and (output/'policy_best_stage_0.pt').is_file()
    # A real resume restores the matching replay, then counts only new samples.
    resumed=tmp_path/'resumed'
    monkeypatch.setattr(sys,'argv',['train.py','--timesteps','12','--resume',str(output/'policy_final.pt'),
                                   '--run-dir',str(resumed),'--eval-every','0','--headless'])
    train.main()
    assert json.loads((resumed/'summary.json').read_text())['transitions']==12
    restored=torch.load(resumed/'policy_final.pt',weights_only=False)
    replay.load(resumed/restored['training_state']['replay_snapshot']['file'])
    assert replay.total==12
