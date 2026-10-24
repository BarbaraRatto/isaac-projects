"""Offline checks for sampling, curriculum, independent resets and asset provenance."""
import ast
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_general_big_map_test1.missions import MissionSampler, Curriculum, deadline_steps
from train_general_big_map_test1.config import MISSION_CFG, NAV_CFG, TERRAIN_NAMES
from train_general_big_map_test1.layouts import map_metadata


def test_sampling_materials_distance_bands_and_splits():
    s = MissionSampler()
    rng = np.random.default_rng(55)
    types = set()
    for evaluation in (False, True):
        for band, (lo, hi) in enumerate(MISSION_CFG.distance_bands):
            for _ in range(200):
                m = s.sample(band, rng, evaluation)
                assert lo <= m['distance_m'] <= hi
                assert s.evaluation[m['start_id']] == evaluation
                assert s.evaluation[m['goal_id']] == evaluation
                a, b = int(m['start'][4]), int(m['goal'][4])
                assert TERRAIN_NAMES[a] != 't2_slippery'
                assert TERRAIN_NAMES[b] not in ('t2_slippery', 't7_stairs')
                assert 150 <= m['deadline_steps'] <= 450
                assert -math.pi <= m['yaw'] <= math.pi
                types.add(a)
    assert types == {0, 2, 3, 4, 5, 6}


def test_local_pose_checks_and_immutable_map():
    s = MissionSampler()
    p = s.points
    assert np.isfinite(p).all()
    assert np.all(p[:, 3]-p[:, 2] <= .16+1e-8)
    assert np.all(abs(p[:, 0]-p[:, 6]*6) <= 3-.65+1e-8)
    assert np.all(abs(p[:, 1]-p[:, 5]*3) <= 1.5-.65+1e-8)
    meta = map_metadata()
    path = Path(__file__).parent/'maps/real_terrains_v2.usd'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == meta['usd_sha256']
    assert not meta['reachability_checked']
    assert p.shape == (4200, 7)


def test_fixed_evaluation_independent_of_reset_order_and_training_rng():
    s = MissionSampler()
    before = s.evaluation_mission(3, 1, MISSION_CFG.evaluation_seed)
    for i in range(10):
        s.training(2)
        s.evaluation_mission(0, i, MISSION_CFG.evaluation_seed)
    assert before == s.evaluation_mission(3, 1, MISSION_CFG.evaluation_seed)
    assert before != s.evaluation_mission(4, 1, MISSION_CFG.evaluation_seed)
    assert [s.evaluation_mission(0, i, 77)['band'] for i in range(3)] == [0, 1, 2]


def test_curriculum_promotion_streak_and_resume():
    c = Curriculum()
    assert not c.update(.9)
    assert not c.update(.7)
    assert c.streak == 0
    assert not c.update(.8)
    assert c.update(.8)
    assert c.level == 1
    assert not c.update(.9)
    restored = Curriculum(); restored.load_state_dict(c.state_dict())
    assert restored.update(1.) and restored.level == 2
    assert not restored.update(1.)
    assert deadline_steps(3) == 150
    assert deadline_steps(12) == 315
    assert deadline_steps(20) == 450


def test_partial_reset_preserves_other_goals_and_robot_states():
    # Compile the actual reset method with a minimal simulator API double.
    tree = ast.parse((Path(__file__).parent/'env.py').read_text())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef))
    reset = next(x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name == '_reset_idx')
    stub = ast.parse('class Probe(Base):\n    pass').body[0]
    stub.body = [reset]
    class Base:
        def _reset_idx(self, ids): self.episode_length_buf[ids] = 0
    namespace = dict(Base=Base, torch=torch, np=np, math=math)
    exec(compile(ast.Module(body=[stub], type_ignores=[]), 'reset_contract', 'exec'), namespace)
    e = namespace['Probe']()
    n=3
    e.device='cpu'; e.physics_dt=.005
    e.scene=NS(env_origins=torch.tensor([[0.,0.,0.],[80.,0.,0.],[160.,0.,0.]]))
    e.sampler=MissionSampler(); e.curriculum=Curriculum(); e.evaluating=False
    e.eval_indices=np.zeros(n,dtype=int); e.eval_seed=77
    e.missions=[None]*n; e.deadline_steps=torch.full((n,),450)
    e.goal_w=torch.full((n,3),999.); e.episode_length_buf=torch.full((n,),50)
    e.settled_root=torch.zeros(n,13); e.settled_root[:,3]=1
    e.standing_clearance=torch.full((n,),.5)
    e.settled_joint_pos=torch.zeros(n,12); e.settled_joint_vel=torch.zeros(n,12)
    e.previous_action=torch.ones(n,2); e.command=torch.ones(n,3)
    e.previous_xy=torch.ones(n,2); e.previous_distance=torch.ones(n)
    for name in ('power_ema','step_energy','episode_energy','path_length','episode_nav_return','episode_cost'):
        setattr(e,name,torch.ones(n))
    written={}
    e.robot=NS(write_root_state_to_sim=lambda root, ids: written.update(root=root.clone(),ids=ids.clone()),
               write_joint_state_to_sim=lambda *args, **kw: None, reset=lambda ids: None, update=lambda dt: None)
    e.controller=NS(reset=lambda ids: None); e.vision=NS(reset=lambda ids: None)
    e._reset_idx(torch.tensor([1]))
    assert torch.all(e.goal_w[[0,2]]==999.)
    assert e.episode_length_buf.tolist()==[50,0,50]
    assert e.episode_energy.tolist()==[1.,0.,1.]
    assert e.missions[0] is None and e.missions[2] is None
    root=written['root'][0]; m=e.missions[1]
    np.testing.assert_allclose(root[:2],np.array(m['start'][:2])+[80.,0.],atol=1e-5)
    assert np.isclose(root[2],m['start'][3]+.52)
    assert not root[7:].any()
    assert np.isclose(torch.linalg.vector_norm(root[3:7]),1.)


def test_evaluation_fixed_bands_restores_rng_and_mode():
    from train_general_big_map_test1.evaluation import evaluate
    class FakeEnv:
        num_envs=2
        evaluating=False
        eval_seed=23
        eval_indices=np.array([7,8])
        def reset(self, seed=None):
            np.random.seed(seed)
            self.elapsed=np.zeros(2,dtype=int)
            self.index=np.zeros(2,dtype=int)
            return {'policy':torch.zeros(2,1),'critic':torch.zeros(2,1)}, {}
        def step(self, action):
            assert self.evaluating
            np.random.random()
            self.elapsed+=1
            records=[]
            for i in range(2):
                if self.elapsed[i]>=i+1:
                    records.append(dict(env_id=i, band=int(self.index[i]%3), success=1,
                                        energy_j=100.,path_m=4.,steps=int(self.elapsed[i]),reason='goal'))
                    self.elapsed[i]=0;self.index[i]+=1
            return {'policy':torch.zeros(2,1),'critic':torch.zeros(2,1)},None,None,None,{'episodes':records}
    policy=NS(act=lambda *args, **kwargs: (torch.zeros(2,2),None,None,None))
    env=FakeEnv()
    np.random.seed(91); expected=np.random.random()
    np.random.seed(91)
    summary, records, score=evaluate(env,policy,'cpu',episodes_per_env=3,seed=44)
    assert np.random.random()==expected
    assert not env.evaluating and env.eval_seed==23
    assert env.eval_indices.tolist()==[7,8]
    assert len(records)==6 and score==(1.,-100.)
    assert [summary['by_band'][str(i)]['episodes'] for i in range(3)]==[2,2,2]
