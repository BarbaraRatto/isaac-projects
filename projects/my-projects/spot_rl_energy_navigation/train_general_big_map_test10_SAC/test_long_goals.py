"""Offline checks for out-of-training-range evaluation. No simulator launched."""
from pathlib import Path
from types import SimpleNamespace
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from train_general_big_map_test10_SAC.missions import MissionSampler,evaluation_distance_range,deadline_steps
from train_general_big_map_test10_SAC.evaluation import evaluate


def test_custom_long_missions_are_reproducible_safe_and_can_exceed_twenty_metres():
    sampler=MissionSampler();sampler.eval_suite='free';sampler.evaluation_width=1
    sampler.eval_distance_range=(12.,24.)
    missions=[sampler.evaluation_mission(0,i,20000,4) for i in range(9)]
    for i,m in enumerate(missions):
        assert m==sampler.evaluation_mission(0,i,20000,4)
        assert 12<=m['distance_m']<=24
        assert sampler.evaluation[m['start_id']] and sampler.evaluation[m['goal_id']]
        assert m['start'][4] not in (1,6) and m['goal'][4] not in (1,6)
        assert np.isfinite(m['reference_length_m'])
        assert m['deadline_steps']==deadline_steps(m['reference_length_m'])
    assert max(m['distance_m'] for m in missions)>20
    # A test override must not change future training mission distributions.
    training=sampler.sample(1,np.random.default_rng(123),evaluation=False,category='free')
    sampler.eval_distance_range=None
    assert training==sampler.sample(1,np.random.default_rng(123),evaluation=False,category='free')
    assert 3<=training['distance_m']<=6


@pytest.mark.parametrize('limits',[(0,24),(-1,24),(24,12),(12,12),(12,np.inf),(np.nan,24),(12,), (12,24,30)])
def test_invalid_custom_distances_fail(limits):
    with pytest.raises(ValueError,match='Distance range'):evaluation_distance_range(limits)


@pytest.mark.parametrize('fail',[False,True])
def test_evaluation_restores_sampler_and_reports_custom_distance(fail):
    class Env:
        num_envs=1
        curriculum=SimpleNamespace(level=1)
        sampler=SimpleNamespace(eval_suite='mixed',eval_distance_range=None)
        eval_stage=1
        evaluating=False
        eval_seed=123
        eval_indices=np.array([7])
        def reset(self,**kwargs):
            assert self.sampler.eval_suite=='free'
            assert self.sampler.eval_distance_range==(12.,24.)
            assert self.eval_stage==4 and self.evaluating
            if fail:raise RuntimeError('synthetic reset failure')
            return dict(policy=torch.zeros(1,3),critic=torch.zeros(1,3)),{}
        def step(self,action):
            record=dict(env_id=0,success=True,reason='goal',steps=1,band=4,
                        mission_category='free',energy_j=10.,path_m=12.,
                        alignment_success=True,alignment_energy_j=0.,alignment_steps=0)
            return None,None,None,None,dict(episodes=[record])
    class Policy:
        def act(self,a,c,deterministic=False):
            assert deterministic
            return torch.zeros(1,2),None,None,None
    env=Env()
    if fail:
        with pytest.raises(RuntimeError,match='synthetic'):
            evaluate(env,Policy(),'cpu',stage=4,suite='free',distance_range=(12,24))
    else:
        summary,_,_=evaluate(env,Policy(),'cpu',stage=4,suite='free',distance_range=(12,24))
        assert summary['evaluation_distance_range_m']==[12.,24.]
        assert '12–24 m' in summary['stage_name']
    assert env.sampler.eval_suite=='mixed' and env.sampler.eval_distance_range is None
    assert env.eval_stage==1 and not env.evaluating and env.eval_seed==123
    np.testing.assert_array_equal(env.eval_indices,[7])
    with pytest.raises(ValueError,match='free'):
        evaluate(env,Policy(),'cpu',stage=4,suite='mixed',distance_range=(12,24))


def test_long_goal_launcher_selects_imported_checkpoint_and_forwards_options(tmp_path):
    (tmp_path/'test_long_goals.sh').write_bytes((HERE/'test_long_goals.sh').read_bytes())
    (tmp_path/'test.sh').write_text('printf "%s\\n" "$@"\n')
    result=subprocess.run(['bash',str(tmp_path/'test_long_goals.sh'),'--headless'],
                          env=os.environ.copy(),text=True,capture_output=True,check=True)
    assert 'policy_best_stage_1.pt' in result.stdout
    assert '--suite\nfree\n--distance-range\n12\n24\n' in result.stdout
    assert '--episodes\n9\n--seed\n20000\n--headless' in result.stdout
