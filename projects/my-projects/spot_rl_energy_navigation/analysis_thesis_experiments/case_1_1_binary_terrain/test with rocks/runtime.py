"""Binary evaluation with geometric route tracking and frozen locomotion."""
import math
from contextlib import ExitStack
from unittest.mock import patch
from binary_logging import make_experiment_env
from controller import RouteTracker
from references import HERE, rows


class UnusedVision:
    def __init__(self,*args,**kwargs):pass
    def reset(self,env_ids):pass


def make_tracking_env(base_class,torch):
    logged_class=make_experiment_env(base_class,torch)
    class TrackingEnv(logged_class):
        def __init__(self,*args,references,lookahead=.35,**kwargs):
            self.references=references;self.lookahead=lookahead
            self.tracker=None;self.last_control=None;self.tracking_errors=[]
            super().__init__(*args,**kwargs)

        def _reset_idx(self,env_ids):
            super()._reset_idx(env_ids)
            mission=self.active_mission
            trace=self.references[mission['mission_id']]
            self.tracker=RouteTracker(trace,(mission['goal_x_m'],mission['goal_y_m']),self.lookahead)
            self.last_control=None;self.tracking_errors=[];self.extended_steps=0

        def _pre_physics_step(self,actions):
            from multi_terrain_test13.config import NAV_CFG
            position=(self.robot.data.root_pos_w[0]-self.scene.env_origins[0]).tolist()
            state=self.tracker.control(position,float(self.yaw()[0]),int(self.episode_length_buf[0]))
            normalized=torch.tensor([[2*state['forward_m_s']/NAV_CFG.max_forward_m_s-1,
                state['yaw_rad_s']/NAV_CFG.max_yaw_rad_s]],device=self.device)
            self.last_control=state
            self.extended_steps+=state['speed_profile_extended']
            super()._pre_physics_step(normalized)

        def _get_observations(self):
            from multi_terrain_test13.config import OBSERVATION_SIZE,CRITIC_SIZE
            self._sync_cameras();self.sim.render();self.camera.update(0.,force_recompute=True)
            return dict(policy=torch.zeros((self.num_envs,OBSERVATION_SIZE),device=self.device),
                        critic=torch.zeros((self.num_envs,CRITIC_SIZE),device=self.device))

        def _get_rewards(self):
            reward=super()._get_rewards()
            position=(self.robot.data.root_pos_w[0]-self.scene.env_origins[0]).tolist()
            self.tracking_errors.append(self.tracker.projection(position)[0])
            return reward

        def sample(self,terminal=False):
            row=super().sample(terminal)
            state=self.last_control
            position=(self.robot.data.root_pos_w[0]-self.scene.env_origins[0]).tolist()
            d=self.robot.data;velocity=d.root_lin_vel_w[0].tolist()
            row.update(source_map=self.active_mission['source_map'],target_map=self.active_mission['target_map'],
                vx_m_s=velocity[0],vy_m_s=velocity[1],speed_xy_m_s=math.hypot(*velocity[:2]),
                body_forward_m_s=float(d.root_lin_vel_b[0,0]),body_yaw_rad_s=float(d.root_ang_vel_b[0,2]),
                tracking_error_m=self.tracker.projection(position)[0],
                reference_step=state['reference_step'] if state else 0,
                reference_time_s=state['reference_time_s'] if state else 0.,
                reference_forward_m_s=state['forward_m_s'] if state else 0.,
                source_yaw_command_rad_s=state['source_yaw_command_rad_s'] if state else 0.,
                route_progress_m=state['route_progress_m'] if state else 0.,
                speed_profile_extended=state['speed_profile_extended'] if state else 0,
                endpoint_extension=state['endpoint_extension'] if state else 0)
            return row

        def _episode_records(self):
            records=super()._episode_records()
            for row in records:
                row.update(mean_speed_m_s=row['path_m']/row['time_s'],
                    mean_tracking_error_m=sum(self.tracking_errors)/len(self.tracking_errors),
                    max_tracking_error_m=max(self.tracking_errors),
                    speed_profile_extended_s=self.extended_steps*self.step_dt,
                    tracking_error_above_025m_steps=sum(x>.25 for x in self.tracking_errors))
            return records
    return TrackingEnv


def create_environment(cfg,layout,cases,lookahead):
    import torch
    from multi_terrain_test13 import env as module
    from multi_terrain_test13.layouts import Condition
    references={c['mission']['mission_id']:rows(HERE/'inputs'/c['reference_file']) for c in cases}
    env_class=make_tracking_env(module.Test13Env,torch)
    with ExitStack() as stack:
        stack.enter_context(patch.object(module,'LocalVision',UnusedVision))
        return env_class(cfg,(Condition(0,layout),),missions=[c['mission'] for c in cases],
            references=references,lookahead=lookahead,pca=None,model_path=None)
