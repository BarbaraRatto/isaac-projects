"""Contracts for the online configuration, batched force and standalone monitor."""
import csv
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from train_general_big_map_test12_SAC_Re.config import experiment_signature
from train_general_big_map_test12_SAC_Re.missions import evaluation_interval
from train_general_big_map_test12_SAC_Re.fast_drag import BatchedSurfaceDrag
from train_general_big_map_test12_SAC_Re.surface_drag import SurfaceDrag


def test_adaptive_evaluation_intervals():
    assert [evaluation_interval(i) for i in range(4)]==[10000,10000,30000,30000]
    assert evaluation_interval(3,12000)==12000
    assert evaluation_interval(0,0)==0
    with pytest.raises(ValueError):
        evaluation_interval(0,-1)


def test_batched_drag_matches_reference_at_boundaries_and_translated_scenes():
    manifest=HERE/'artifacts/drag_manifest.json'
    terrain=HERE/'maps/lab_15x15.usda'
    reference=SurfaceDrag(manifest,terrain)
    fast=BatchedSurfaceDrag(manifest,terrain)
    xy=torch.cartesian_prod(torch.tensor([-.1,0.,2.99,3.,6.,9.,12.,14.99,15.]),
                            torch.tensor([-.1,0.,3.,6.,9.,12.,14.99,15.]))
    origins=torch.zeros((len(xy),3))
    origins[:,0]=torch.arange(len(xy))*80.
    positions=origins.clone(); positions[:,:2]+=xy
    data=SimpleNamespace(root_pos_w=positions,root_lin_vel_w=torch.randn((len(xy),3),generator=torch.Generator().manual_seed(2)))
    calls=[]
    robot=SimpleNamespace(data=data,set_external_force_and_torque=lambda **kw:calls.append(kw))
    expected=reference.apply(robot,origins)
    actual=fast.apply(robot,origins)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert calls[-1]['body_ids']==[0] and calls[-1]['is_global']
    assert torch.count_nonzero(calls[-1]['torques'])==0
    assert torch.count_nonzero(actual[:,2])==0


def test_monitor_before_first_adaptive_evaluation(tmp_path):
    (tmp_path/'config.json').write_text(json.dumps({'arguments':{'timesteps':1500000,'eval_every':None},'signature':experiment_signature()}))
    row=dict(transitions=1024,wall_seconds=20,visual_cells=30,geometry_cells=30,
             updates=0,replay_size=1024,action_saturation=0,alignment_transitions=10,curriculum_level=0)
    with (tmp_path/'training.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(row)); writer.writeheader(); writer.writerow(row)
    result=subprocess.run([sys.executable,str(HERE/'status.py'),'--run-dir',str(tmp_path)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert '10,000' in result.stdout and 'livello 0' in result.stdout
