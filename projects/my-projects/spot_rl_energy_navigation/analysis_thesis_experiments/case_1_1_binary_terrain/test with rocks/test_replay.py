"""Offline checks for route replay, map swapping and terminal logging."""
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
import torch
from controller import RouteTracker
from references import HERE,load,rows
from runtime import make_tracking_env
from checks import validate_forward_commands


@dataclass(frozen=True)
class Geometry:
    start_jitter_m:float=.05


class FakeBase:
    def __init__(self):
        self.geometry=Geometry();self.device='cpu';self.num_envs=1
        self.scene=SimpleNamespace(env_origins=torch.tensor([[10.,20.,0.]]))
        self.settled_root=torch.zeros((1,13));self.goal_w=torch.zeros((1,3))
        self.episode_length_buf=torch.zeros(1,dtype=torch.long);self._sim_step_counter=0
        self.step_dt=.2;self.physics_dt=.005
        self.robot=SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.zeros((1,3)),
            root_lin_vel_w=torch.tensor([[.3,.4,0.]]),root_lin_vel_b=torch.tensor([[.2,0.,0.]]),
            root_ang_vel_b=torch.tensor([[0.,0.,.5]])))
        self.previous_action=torch.zeros((1,2));self.command=torch.zeros((1,3))
        self.episode_energy=torch.tensor([2.]);self.path_length=torch.tensor([.1])
        self.fallen=torch.tensor([False]);self.timeout=torch.tensor([False])

    def _reset_idx(self,ids):
        if self.geometry.start_jitter_m!=0:raise AssertionError('Unexpected reset noise')
        self.robot.data.root_pos_w=self.settled_root[:,:3].clone()
        self.episode_length_buf.zero_()

    def yaw(self):return torch.tensor([math.pi/4])
    def _distance(self):return (self.robot.data.root_pos_w[:,:2]-self.goal_w[:,:2]).norm(dim=1)
    def _pre_physics_step(self,actions):
        from multi_terrain_test13.actions import navigation_commands
        self.previous_action=actions;self.command=navigation_commands(actions)
    def _get_rewards(self):return torch.zeros(1)
    def _episode_records(self):return [dict(steps=1,energy_j=2.,path_m=.1,reason='goal')]


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.document=load()

    def test_both_maps_swapped_and_reference_commands_preserved(self):
        from multi_terrain_test13.layouts import ROCKS,MIRRORED
        maps={'rocks_right':ROCKS,'rocks_left':MIRRORED}
        for case in self.document['cases']:
            mission=case['mission'];ref=rows(HERE/'inputs'/case['reference_file'])
            self.assertNotEqual(mission['source_map'],mission['target_map'])
            self.assertEqual(mission['target_map'],'rocks_left' if mission['mission_id'].startswith('A') else 'rocks_right')
            source=maps[mission['source_map']].grid;target=maps[mission['target_map']].grid
            for r in range(3):
                for c in range(3):
                    if (r,c) not in ((0,0),(1,1),(2,2)):
                        self.assertEqual({source[r][c],target[r][c]},{'t1_asphalt','t5_large_rocks'})
            tracker=RouteTracker(ref,(mission['goal_x_m'],mission['goal_y_m']))
            for i,row in enumerate(ref[:-1]):
                result=tracker.control([float(row['x_m']),float(row['y_m'])],float(row['yaw_rad']),i)
                self.assertEqual(result['forward_m_s'],float(ref[i+1]['command_forward_m_s']))
                self.assertEqual(result['reference_step'],i+1)
                self.assertFalse(result['speed_profile_extended'])
                self.assertLessEqual(abs(result['yaw_rad_s']),1.5)

    def test_tracking_correction_is_steering_only(self):
        trace=[dict(x_m=x,y_m=0,time_s=i*.2,command_forward_m_s=1.,command_yaw_rad_s=0.) for i,x in enumerate([0,1,2])]
        for y in (-.2,.2):
            tracker=RouteTracker(trace,(2,0))
            command=tracker.control([.3,y],0,0)
            self.assertEqual(command['forward_m_s'],1.)
            self.assertLess(command['yaw_rad_s']*y,0)
            self.assertAlmostEqual(command['tracking_error_m'],abs(y))

    def test_end_of_profile_is_explicitly_flagged(self):
        case=self.document['cases'][0];trace=rows(HERE/'inputs'/case['reference_file']);m=case['mission']
        tracker=RouteTracker(trace,(m['goal_x_m'],m['goal_y_m']))
        row=trace[0];result=tracker.control([float(row['x_m']),float(row['y_m'])],float(row['yaw_rad']),len(trace)+3)
        self.assertTrue(result['speed_profile_extended'])
        self.assertEqual(result['forward_m_s'],float(trace[-1]['command_forward_m_s']))

    def test_exact_spawn_commands_and_terminal_before_reset(self):
        cases=self.document['cases'][:2]
        references={c['mission']['mission_id']:rows(HERE/'inputs'/c['reference_file']) for c in cases}
        env=make_tracking_env(FakeBase,torch)(missions=[c['mission'] for c in cases],references=references)
        env._reset_idx(torch.tensor([0]));first=cases[0]['mission']
        self.assertAlmostEqual(float(env.robot.data.root_pos_w[0,0])-10,first['start_x_m'],places=5)
        env._pre_physics_step(torch.zeros((1,2)))
        expected=float(references[first['mission_id']][1]['command_forward_m_s'])
        self.assertAlmostEqual(float(env.command[0,0]),expected,places=6)
        env.episode_length_buf[0]=1;env._get_rewards()
        result=env._episode_records()[0];terminal=dict(env.terminal_sample)
        self.assertEqual(terminal['mission_id'],first['mission_id'])
        self.assertEqual(terminal['terminal'],1)
        self.assertAlmostEqual(terminal['speed_xy_m_s'],.5,places=6)
        self.assertIn('mean_tracking_error_m',result)
        env._reset_idx(torch.tensor([0]))
        self.assertEqual(env.active_mission['mission_id'],cases[1]['mission']['mission_id'])
        self.assertEqual(terminal['mission_id'],first['mission_id'])
        self.assertIsNone(env.last_control)

    def test_saved_command_tampering_is_rejected(self):
        case=self.document['cases'][0];trace=rows(HERE/'inputs'/case['reference_file'])
        with TemporaryDirectory() as directory:
            path=Path(directory)/'trajectories.csv'
            def write(value):
                with path.open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=['mission_id','step','command_forward_m_s','reference_step','speed_profile_extended'])
                    writer.writeheader();writer.writerow(dict(mission_id=case['mission']['mission_id'],step=1,
                        command_forward_m_s=value,reference_step=1,speed_profile_extended=0))
            expected=float(trace[1]['command_forward_m_s'])
            write(expected);validate_forward_commands(Path(directory),[case])
            write(expected+.1)
            with self.assertRaisesRegex(ValueError,'forward command'):validate_forward_commands(Path(directory),[case])


if __name__=='__main__':unittest.main()
