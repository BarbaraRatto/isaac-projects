"""Offline checks for paired missions, line commands and physical logging."""
import copy
import csv
import io
import contextlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch
import numpy as np
import torch
from controller import commands
from missions import HERE, MAP, METADATA, FixedSampler, load_suite, write_json
from create_exp import CSVLog, validate_logs
import runtime

class FakeBase:
    def __init__(self):
        self.eval_indices = np.zeros(1, dtype=int)
        self._sim_step_counter = 50
        self.step_dt, self.physics_dt = .2, .005
        self.step_was_alignment = torch.ones(1, dtype=torch.bool)
        self.episode_length_buf = torch.zeros(1, dtype=torch.long)
        self.scene = SimpleNamespace(env_origins=torch.tensor([[50., 80., 0.]]))
        self.robot = SimpleNamespace(data=SimpleNamespace(
            root_pos_w=torch.tensor([[51., 81., .5]]),
            root_lin_vel_w=torch.tensor([[.3, .4, 0.]]), root_ang_vel_w=torch.tensor([[0., 0., .2]]),
            root_lin_vel_b=torch.tensor([[.1, .2, 0.]]), root_ang_vel_b=torch.tensor([[0., 0., .2]])))
        self.command = torch.tensor([[1., 0., .5]])
        self.previous_action = torch.zeros(1, 2)
        self.episode_energy, self.path_length = torch.tensor([2.]), torch.tensor([.2])

    def _reset_idx(self, env_ids):
        self.eval_indices[0] += 1
        self.episode_length_buf.zero_()
        self.robot.data.root_pos_w[0, :2] = torch.tensor([51., 81.])

    def yaw(self):
        return torch.tensor([0.])

    def _distance(self):
        return torch.tensor([.4])

    def _episode_records(self):
        return [dict(steps=1, path_m=.2, energy_j=2., reason='goal')]

    def _measured_power(self):
        return torch.tensor([10.])


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.document=load_suite(12)

    def fake_env(self):
        class Base(FakeBase):
            def __init__(self):
                super().__init__()
                self.device='cpu';self.num_envs=1
                from train_general_big_map_test7_SAC.alignment import InitialAlignment
                self.alignment=InitialAlignment(1,'cpu')
            def _pre_physics_step(self,actions):
                from train_general_big_map_test7_SAC.actions import navigation_commands
                self.step_was_alignment=self.alignment.active.clone()
                self.previous_action=self.alignment.actions(actions,torch.zeros(1))
                self.command=navigation_commands(self.previous_action)
                self.step_energy=torch.zeros(1)
        env=runtime.make_logging_env(Base,torch)(suite=self.document)
        env._reset_idx(torch.tensor([0]))
        return env

    def test_eighty_pairs_match_the_frozen_thesis_selections(self):
        for distance in (12,18):
            doc=load_suite(distance)
            selection=json.loads(Path(doc['source_selection']).read_text())
            self.assertEqual(len(doc['cases']),40)
            self.assertEqual([c['episode'] for c in doc['cases']],list(range(1,41)))
            for c,s in zip(doc['cases'],selection['slots']):
                self.assertEqual(c['mission'],s['case']['mission'])
                self.assertEqual(c['category'],s.get('thesis_category',s['case']['category']))
                self.assertEqual(c['reference']['source_episode'],s['source_episode'])
                self.assertAlmostEqual(c['reference_speed_m_s'],c['command_means']['navigation_forward_m_s'],places=12)
                self.assertEqual(c['mission']['deadline_steps'],315 if distance==12 else 435)
            self.assertTrue(any(c['reference_policy_result']['reason']!='goal' for c in doc['cases']))

    def test_heading_tracking_and_constant_forward_speed(self):
        straight=commands([0,0],[12,0],[3,0],0,1.2)
        self.assertEqual(straight['yaw_rad_s'],0)
        for y in (-2.,2.):
            state=commands([0,0],[12,0],[3,y],0,1.2)
            self.assertEqual(state['forward_m_s'],1.2)
            self.assertLess(state['yaw_rad_s']*y,0)
        vertical=commands([0,0],[0,18],[0,6],math.pi/2,.8)
        self.assertAlmostEqual(vertical['yaw_rad_s'],0)
        wrapped=commands([0,0],[-12,-.1],[-3,0],math.pi-.01,1.1)
        self.assertLess(abs(wrapped['yaw_rad_s']),.1)
        capped=commands([0,0],[12,0],[3,10],math.pi,1.1)
        self.assertLessEqual(abs(capped['yaw_rad_s']),1.5)

    def test_invalid_controller_inputs_are_rejected(self):
        for speed in (0.,-1.,float('nan')):
            with self.assertRaises(ValueError):commands([0,0],[12,0],[1,0],0,speed)
        with self.assertRaises(ValueError):commands([0,0],[0,0],[1,0],0,1.)

    def test_speed_corruption_is_rejected(self):
        with TemporaryDirectory() as folder:
            doc=copy.deepcopy(self.document);doc['cases'][0]['reference_speed_m_s']+=.05
            path=Path(folder)/'bad.json';write_json(path,doc)
            with self.assertRaisesRegex(ValueError,'reference speed'):load_suite(12,path)

    def test_fixed_sampler_final_reset_and_copy(self):
        sampler=FixedSampler(self.document)
        for i in (0,12,39,40):
            self.assertEqual(sampler.evaluation_mission(0,i,0),self.document['cases'][min(i,39)]['mission'])
        m=sampler.evaluation_mission(0,0,0);m['start'][0]=-100
        self.assertNotEqual(sampler.evaluation_mission(0,0,0)['start'][0],-100)
        with self.assertRaises(RuntimeError):sampler.training(0)
        self.assertEqual(sampler.points[0,4],0)

    def test_initial_alignment_overrides_forward_then_releases_exact_speed(self):
        env=self.fake_env()
        env._pre_physics_step(torch.zeros((1,2)))
        self.assertEqual(float(env.command[0,0]),0.)
        self.assertTrue(bool(env.step_was_alignment[0]))
        for _ in range(3):env.alignment.advance(torch.zeros(1))
        env._pre_physics_step(torch.zeros((1,2)))
        self.assertFalse(bool(env.step_was_alignment[0]))
        self.assertAlmostEqual(float(env.command[0,0]),env.active_case['reference_speed_m_s'],places=6)

    def test_terminal_is_logged_before_reset_with_actual_velocity(self):
        env=self.fake_env();env.episode_length_buf[0]=1
        env.step_was_alignment[0]=False
        env.robot.data.root_pos_w[0,:2]=torch.tensor([55.,86.])
        record=env._episode_records()[0];snapshot=env.terminal_sample
        env._reset_idx(torch.tensor([0]))
        self.assertEqual(record['episode'],1);self.assertEqual(snapshot['episode'],1)
        self.assertEqual(env.active_case['episode'],2);self.assertEqual(snapshot['terminal'],1)
        self.assertEqual((snapshot['x_m'],snapshot['y_m']),(5.,6.))
        self.assertAlmostEqual(snapshot['speed_xy_m_s'],.5,places=6)
        self.assertAlmostEqual(record['reference_speed_m_s'],self.document['cases'][0]['reference_speed_m_s'])
        self.assertEqual(record['policy_source_episode'],self.document['cases'][0]['reference']['source_episode'])

    def test_power_boundary_not_duplicated(self):
        env=self.fake_env();samples=[];env.power_log=SimpleNamespace(write=samples.append)
        env._measured_power();env._measured_power();env._sim_step_counter+=1;env._measured_power()
        self.assertEqual([r['physics_step'] for r in samples],[0,1])

    def test_constructor_disables_navigation_perception_and_restores_modules(self):
        import train_general_big_map_test7_SAC as package
        env_module,scene_module=ModuleType('fake_env'),ModuleType('fake_scene')
        env_module.map_metadata=lambda:'old';env_module.MissionSampler=lambda:'old'
        env_module.LocalVision=lambda:'old vision';old_vision=env_module.LocalVision
        scene_module.validate_terrain=lambda:'old'
        metadata=json.loads(METADATA.read_text())
        class Base:
            def __init__(self,cfg,conditions,**kwargs):
                self.metadata=env_module.map_metadata();self.sampler=env_module.MissionSampler()
                self.vision=env_module.LocalVision();self.conditions=conditions
        env_module.GeneralMapEnv=Base
        with patch.object(package,'env',env_module,create=True),patch.object(package,'scene',scene_module,create=True),patch.object(runtime,'validate_map',return_value=metadata):
            env=runtime.create_environment(None,self.document,None,None,'cpu')
        self.assertIsInstance(env.vision,runtime.UnusedVision)
        self.assertIs(env_module.LocalVision,old_vision)
        self.assertIsInstance(env.sampler,FixedSampler)
        self.assertEqual(env.conditions[0].layout.usd,MAP)
        self.assertEqual(env.eval_stage,3)

    def test_raw_energy_and_missing_terminal_checks(self):
        doc=copy.deepcopy(self.document);doc['cases']=doc['cases'][:1]
        with TemporaryDirectory() as folder:
            output=Path(folder)
            episodes=CSVLog(output/'episodes.csv',['episode','steps','energy_j','path_m'])
            trajectory=CSVLog(output/'trajectories.csv',['episode','step','time_s','x_m','y_m','energy_j','terminal'])
            power=CSVLog(output/'power_samples.csv',['episode','physics_step','time_s','power_w'])
            x,y=doc['cases'][0]['mission']['start'][:2]
            episodes.write(dict(episode=1,steps=1,energy_j=2.,path_m=.2))
            for step in (0,1):trajectory.write(dict(episode=1,step=step,time_s=step*.2,x_m=x+step*.2,y_m=y,energy_j=step*2,terminal=step))
            for tick in range(41):power.write(dict(episode=1,physics_step=tick,time_s=tick*.005,power_w=10.))
            for log in (episodes,trajectory,power):log.close()
            self.assertEqual(validate_logs(output,doc,.005,40)['completed_missions'],1)
            text=(output/'episodes.csv').read_text().replace('2.0','20.0');(output/'episodes.csv').write_text(text)
            with self.assertRaisesRegex(ValueError,'Power integral'):validate_logs(output,doc,.005,40)


class MeanCommandTests(unittest.TestCase):
    def test_time_weighting_and_phase_separation(self):
        from command_means import extract
        def row(step,time,phase,v,w):
            return dict(step=str(step),time_s=str(time),phase=phase,
                command_vx_m_s=str(v),command_wz_rad_s=str(w))
        rows=[row(0,0,'alignment',99,99),row(1,.2,'alignment',0,-.6),
              row(2,.6,'alignment',0,-.3),row(3,.8,'navigation',1,.8),
              row(4,1.4,'navigation',1.4,-.8)]
        result=extract(rows)
        self.assertAlmostEqual(result['alignment_yaw_rad_s'],-.4)
        self.assertAlmostEqual(result['navigation_forward_m_s'],1.3)
        self.assertEqual(result['sample_count'],4)

    def test_alignment_direction_stop_and_navigation_handoff(self):
        from controller import mean_alignment_actions
        actions=torch.tensor([[.2,.3],[.2,.3],[.2,.3],[.2,.3]])
        error=torch.tensor([.5,-.5,.02,.5])
        active=torch.tensor([True,True,True,False])
        result=mean_alignment_actions(actions,error,active,-.6,.1,1.5)
        torch.testing.assert_close(result,torch.tensor([[-1.,.4],[-1.,-.4],[-1.,0.],[.2,.3]]))


if __name__=='__main__':
    unittest.main()
