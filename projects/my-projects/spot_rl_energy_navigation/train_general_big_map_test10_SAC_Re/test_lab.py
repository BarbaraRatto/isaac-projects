"""Offline contracts for the two-material map, missions, PCA and physical drag."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from pxr import Usd, UsdGeom

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from train_general_big_map_test10_SAC_Re.config import SAC_CFG, MISSION_CFG, TERRAIN_NAMES
from train_general_big_map_test10_SAC_Re.layouts import map_metadata, TRAIN_LAYOUTS
from train_general_big_map_test10_SAC_Re.missions import MissionSampler, Curriculum
from train_general_big_map_test10_SAC_Re.prepare_map import layout
from train_general_big_map_test10_SAC_Re.terrain import validate_terrain
from train_general_big_map_test10_SAC_Re.surface_drag import SurfaceDrag
from train_general_big_map_test10_SAC_Re.calibration import view_plan, point_labels, validate_calibration
from train_general_big_map_test10_SAC_Re.networks import ObservationNormalizer
from train_general_big_map_test10_SAC_Re.test_offline import environment_method, synthetic_observations


def test_lab_map_physics_flat_cells_and_frozen_layout():
    meta=validate_terrain()
    assert meta['grid']==layout(42)
    assert meta['bounds_max'][:2]==[15.,15.] and meta['bounds_min'][:2]==[0.,0.]
    assert np.sum(np.array(meta['grid'])=='lab_green')==8
    stage=Usd.Stage.Open(str(TRAIN_LAYOUTS[0].usd))
    for r,row in enumerate(meta['grid']):
        for c,name in enumerate(row):
            mesh=UsdGeom.Mesh.Get(stage,f'/World/Grid/Cell_{r}_{c}_{name}/terrain/mesh')
            points=np.asarray(mesh.GetPointsAttr().Get())
            np.testing.assert_allclose(points[:,:2].min(0),[3*c,3*r])
            np.testing.assert_allclose(points[:,:2].max(0),[3*c+3,3*r+3])
            np.testing.assert_allclose(np.unique(points[:,2]),[-.1,0.])
    manifest=json.loads((HERE/'artifacts/drag_manifest.json').read_text())
    assert manifest['materials']['green']['horizontal_drag_ns_per_m']==90.
    for material in manifest['materials'].values():
        assert material['static_friction']==1. and material['dynamic_friction']==.8
        assert material.get('compliant_contact_stiffness',0.)==0.
    assert SAC_CFG.timesteps==100000 and SAC_CFG.num_envs==32


@pytest.mark.parametrize('evaluation',[False,True])
@pytest.mark.parametrize('band',range(4))
def test_missions_respect_endpoints_distance_and_disjoint_split(evaluation,band):
    sampler=MissionSampler()
    rng=np.random.default_rng(17)
    seen=set()
    for _ in range(100):
        m=sampler.sample(band,rng,evaluation)
        assert sampler.evaluation[m['start_id']]==evaluation
        assert sampler.evaluation[m['goal_id']]==evaluation
        lo,hi=MISSION_CFG.distance_bands[band]
        assert lo<=m['distance_m']<=hi
        assert m['start_id']!=m['goal_id']
        seen.update([int(m['start'][4]),int(m['goal'][4])])
        if band==0:
            assert m['start'][4]==m['goal'][4]==0
    assert seen==({0} if band==0 else {0,1})
    assert sampler.evaluation_mission(0,1,55,band)==sampler.evaluation_mission(0,1,55,band)


def test_curriculum_requires_two_consecutive_eighty_percent_evaluations():
    c=Curriculum()
    assert not c.update(.8) and c.streak==1
    assert not c.update(.79) and c.streak==0
    assert not c.update(.8)
    assert c.update(.8) and c.level==1


def test_calibration_balances_two_colors_and_excludes_evaluation_endpoints():
    sampler=MissionSampler()
    plan=view_plan(sampler.points,sampler.evaluation,map_metadata())
    assert len(plan)==2*(96+24)
    ids={}
    for split,count in [('train',96),('validation',24)]:
        selected=[x for x in plan if x['split']==split]
        assert np.bincount([x['target_material'] for x in selected]).tolist()==[count]*2
        ids[split]={x['point_id'] for x in selected}
        assert not sampler.evaluation[list(ids[split])].any()
    assert ids['train'].isdisjoint(ids['validation'])
    centers=np.array([[3*c+1.5,3*r+1.5] for r in range(5) for c in range(5)])
    np.testing.assert_array_equal(point_labels(centers,map_metadata()),
                                 [TERRAIN_NAMES.index(x) for row in map_metadata()['grid'] for x in row])
    with pytest.raises(ValueError,match='not calibrated'):
        validate_calibration(NS(metadata={'dataset':'old_seven_terrains'}),{})


def test_batched_drag_uses_local_base_xy_but_world_velocity_and_clears():
    meta=map_metadata()
    r,c=np.argwhere(np.array(meta['grid'])=='lab_green')[0]
    gr,gc=np.argwhere(np.array(meta['grid'])=='t1_asphalt')[0]
    origins=torch.tensor([[0.,0.,0.],[80.,0.,0.],[0.,80.,0.]])
    points=torch.tensor([[3*c+1.5,3*r+1.5,.5],[3*c+1.5,3*r+1.5,.5],
                         [3*gc+1.5,3*gr+1.5,.5]])+origins
    velocity=torch.tensor([[.9,0.,.5],[-.9,.2,0.],[.9,0.,0.]])
    calls=[]
    robot=NS(data=NS(root_pos_w=points,root_lin_vel_w=velocity),
             set_external_force_and_torque=lambda **kw:calls.append(kw))
    drag=SurfaceDrag(HERE/'artifacts/drag_manifest.json',TRAIN_LAYOUTS[0].usd)
    forces=drag.apply(robot,origins)
    torch.testing.assert_close(forces,torch.tensor([[-81.,0.,0.],[81.,-18.,0.],[0.,0.,0.]]))
    assert bool(((forces*velocity).sum(1)<=0).all())
    assert calls[-1]['body_ids']==[0] and calls[-1]['is_global'] is True
    assert not bool(calls[-1]['torques'].any())
    robot.data.root_pos_w=origins+torch.tensor([3*gc+1.5,3*gr+1.5,.5])
    assert not bool(drag.apply(robot,origins).any())
    assert not bool(calls[-1]['forces'].any())


def test_environment_updates_drag_before_controller_and_scene_write():
    calls=[]
    env=NS(robot=object(),scene=NS(env_origins=object()),command=object(),
           surface_drag=NS(apply=lambda *args:calls.append('drag')),
           controller=NS(apply=lambda *args:calls.append('control')))
    method=environment_method('_apply_action')
    method(env)
    assert calls==['drag','control']


def test_goal_scaling_stays_fixed_and_other_statistics_update():
    a,c=synthetic_observations()
    norm=ObservationNormalizer()
    before=norm.visual.count.clone()
    for distance in (3.,6.,12.):
        a[:,:2]=distance
        c[:,:a.shape[1]]=a
        norm.update(a,c)
        torch.testing.assert_close(norm.actor(a)[:,:2],a[:,:2]/10)
        torch.testing.assert_close(norm.critic(c)[:,:2],c[:,:2]/10)
    assert bool((norm.visual.count>before).all())


@pytest.mark.parametrize('args,fail,expected',[
    ([],False,['prepare_map.py','prepare_pca.py','train.py']),
    ([],True,['prepare_map.py','prepare_pca.py']),
    (['--resume','/tmp/model.pt'],False,['train.py']),
    (['--help'],False,['train.py']),
])
def test_launcher_calibrates_then_trains_and_stops_on_failed_pca(tmp_path,args,fail,expected):
    launcher=tmp_path/'isaaclab'
    launcher.write_text('''#!/usr/bin/env python3
import os,json,sys
with open(os.environ['CALL_LOG'],'a') as f:f.write(json.dumps(sys.argv[1:])+'\\n')
if os.environ['FAIL_PREP']=='1' and sys.argv[2].endswith('prepare_pca.py'):sys.exit(12)
''')
    launcher.chmod(0o755)
    log=tmp_path/'calls'
    env=dict(os.environ,GENERAL_MAP_ISAACLAB=str(launcher),CALL_LOG=str(log),FAIL_PREP=str(int(fail)))
    result=subprocess.run(['bash',str(HERE/'train.sh'),*args],env=env,capture_output=True)
    assert (result.returncode!=0)==fail,result.stderr
    calls=[json.loads(line) for line in log.read_text().splitlines()]
    assert [Path(call[1]).name for call in calls]==expected
    if not args and not fail:
        assert calls[-1][calls[-1].index('--timesteps')+1]=='100000'
        assert calls[-1][calls[-1].index('--agent-device')+1]=='cpu'
        assert '--rgb-backend' not in calls[-1] and '--replay-device' not in calls[-1]
