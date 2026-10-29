"""Physics-frequency fall persistence and logging, without starting Isaac Sim."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_general_big_map_test6_SAC.falls import FallTracker, terrain_contact_force, FALL_FIELDS
from train_general_big_map_test6_SAC.config import NAV_CFG
from train_general_big_map_test6_SAC.test_offline import environment_method


def test_exactly_one_second_and_independent_reset():
    f = FallTracker(3, 'cpu', .005)
    gravity = torch.tensor([[1.,0.,0.], [0.,0.,-1.], [0.,0.,-1.]])
    force = torch.tensor([0.,20.,0.])
    for _ in range(199): f.update(gravity, force)
    assert not f.fallen.any()
    f.update(gravity, force)
    assert f.fallen.tolist() == [True,True,False]
    assert f.cause(0) == 'tilt' and f.cause(1) == 'body_contact'
    f.reset(torch.tensor([0]))
    assert f.fallen.tolist() == [False,True,False]
    assert f.body_steps.tolist() == [0,200,0]


def test_transient_contact_and_tilt_do_not_accumulate():
    f = FallTracker(1,'cpu',.005)
    bad = torch.tensor([[1.,0.,0.]])
    good = torch.tensor([[0.,0.,-1.]])
    for _ in range(3):
        for _ in range(150): f.update(bad,torch.tensor([15.]))
        f.update(good,torch.zeros(1))
    assert not f.fallen.any()  # >2 seconds total, but never 1 consecutive second
    assert int(f.max_tilt_steps[0]) == 150
    for _ in range(200): f.update(bad,torch.tensor([15.]))
    assert f.cause(0) == 'tilt+body_contact'
    f.update(good,torch.zeros(1))
    assert f.fallen.item()  # confirmed inside an action: retained until terminal/reset


def test_missing_height_and_low_clearance_are_not_falls():
    done = environment_method('_get_dones', terrain_height=lambda _:torch.tensor([float('nan'), .5]))
    e = NS(nav=NAV_CFG,fall_tracker=FallTracker(2,'cpu',.005),
           geometry=NS(world_x_bounds=(-3,39),world_y_bounds=(-1.5,19.5)),
           scene=NS(env_origins=torch.zeros(2,3)),
           robot=NS(data=NS(root_pos_w=torch.tensor([[0.,0.,.6],[0.,0.,.6]]),
                            projected_gravity_b=torch.tensor([[0.,0.,-1.],[0.,0.,-1.]]))),
           alignment=NS(failed=torch.zeros(2,dtype=torch.bool)),
           episode_length_buf=torch.ones(2,dtype=torch.long),deadline_steps=torch.full((2,),150),
           low_clearance_steps=torch.zeros(2,dtype=torch.long),invalid_height_steps=torch.zeros(2,dtype=torch.long),
           legacy_fall_steps=torch.zeros(2,dtype=torch.long),_distance=lambda:torch.full((2,),2.))
    terminal,_ = done(e)
    assert not terminal.any() and not e.fallen.any()
    assert e.low_clearance_steps.tolist()==[0,1]
    assert e.invalid_height_steps.tolist()==[1,0]
    assert e.legacy_fall_steps.tolist()==[1,1]
    e.physics_dt=.005
    record=environment_method('_fall_record')(e,0)
    assert set(record)==set(FALL_FIELDS)
    assert record['base_clearance_m'] is None
    json.dumps(record,allow_nan=False)
    e.fall_tracker.body_confirmed[1]=True
    terminal,_=done(e)
    assert terminal.tolist()==[False,True] and e.fallen.tolist()==[False,True]


def test_contact_force_uses_only_single_trunk_filtered_groups():
    force=torch.zeros(2,1,49,3)
    force[0,0,2,2]=12.
    force[0,0,3,2]=-7.
    torch.testing.assert_close(terrain_contact_force(force),torch.tensor([19.,0.]))
    with pytest.raises(RuntimeError): terrain_contact_force(torch.zeros(2,4,49,3))
    with pytest.raises(RuntimeError):
        FallTracker(1,'cpu',.005).update(torch.tensor([[float('nan'),0.,0.]]),torch.zeros(1))
