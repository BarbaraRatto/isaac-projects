"""Isaac Lab environment with explicit sensor timing and true finite-horizon terminals."""
import math

import numpy as np
import torch
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul, yaw_quat

from train_general_big_map_test4.actions import navigation_commands
from train_general_big_map_test4.alignment import InitialAlignment, heading_error
from train_general_big_map_test4.rewards import orientation_penalty
from train_general_big_map_test4.config import MISSION_CFG, TERRAIN_NAMES, NAV_CFG, FOOT_NAMES, OBSERVATION_SIZE, CRITIC_SIZE
from train_general_big_map_test4.locomotion import BatchedSpotVelocityController
from train_general_big_map_test4.privileged import PrivilegedContacts, terrain_height
from train_general_big_map_test4.scene import setup_scene
from train_general_big_map_test4.vision import LocalVision


from train_general_big_map_test4.missions import Curriculum, MissionSampler
from train_general_big_map_test4.layouts import map_metadata


class GeneralMapEnv(DirectRLEnv):
    def __init__(self, cfg, conditions, pca=None, dino_batch_size=4, model_path=None,
                 capture=False, **kwargs):
        self.conditions = conditions
        self.nav = NAV_CFG
        if len(conditions) != cfg.scene.num_envs or tuple(c.env_id for c in conditions) != tuple(range(len(conditions))):
            raise ValueError('Conditions must match and be ordered by environment index')
        super().__init__(cfg, **kwargs)
        meta = map_metadata()
        from types import SimpleNamespace
        self.geometry = SimpleNamespace(world_x_bounds=(meta['bounds_min'][0]+.2, meta['bounds_max'][0]-.2),
                                        world_y_bounds=(meta['bounds_min'][1]+.2, meta['bounds_max'][1]-.2))
        self.sampler = MissionSampler()
        self.curriculum = Curriculum()
        self.evaluating = False
        self.eval_seed = MISSION_CFG.evaluation_seed
        self.eval_stage = 0
        self.eval_indices = np.zeros(self.num_envs, dtype=int)
        self.missions = [None]*self.num_envs
        self.deadline_steps = torch.full((self.num_envs,), self.nav.max_episode_steps, device=self.device, dtype=torch.long)
        self.foot_ids = [self.robot.body_names.index(name) for name in FOOT_NAMES]
        self.controller = BatchedSpotVelocityController(self.robot, self.num_envs, self.device)
        self.vision = LocalVision(self.device, pca, dino_batch_size, model_path, capture)
        self.contacts = PrivilegedContacts(self)
        self.goal_w = self.scene.env_origins.clone()
        self.previous_action = torch.zeros((self.num_envs, 2), device=self.device)
        self.command = torch.zeros((self.num_envs, 3), device=self.device)
        self.previous_xy = torch.zeros((self.num_envs, 2), device=self.device)
        for name in ('previous_distance', 'power_ema', 'step_energy', 'episode_energy',
                     'path_length', 'episode_nav_return', 'episode_cost'):
            setattr(self, name, torch.zeros(self.num_envs, device=self.device))
        self.alignment = InitialAlignment(self.num_envs, self.device)
        self.alignment_energy = torch.zeros(self.num_envs, device=self.device)
        self.alignment_path = torch.zeros(self.num_envs, device=self.device)
        self.navigation_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.step_was_alignment = self.alignment.active.clone()
        self._settle_once()

    def _setup_scene(self):
        setup_scene(self)

    def _settle_once(self):
        # Calibrate a standing pose once on a flat asphalt candidate, independently of missions.
        points = self.sampler.points
        flat = points[points[:, 4] == TERRAIN_NAMES.index('t1_asphalt')][0]
        root = self.robot.data.default_root_state.clone()
        root[:, :3] = self.scene.env_origins
        root[:, :2] += torch.tensor(flat[:2], device=self.device, dtype=root.dtype)
        root[:, 2] += float(flat[3])+self.nav.spawn_height_m
        root[:, 3:7] = torch.tensor([1., 0., 0., 0.], device=self.device)
        root[:, 7:] = 0
        self.robot.write_root_state_to_sim(root)
        self.robot.write_joint_state_to_sim(self.robot.data.default_joint_pos, torch.zeros_like(self.robot.data.joint_vel))
        self.robot.reset()
        self.robot.update(self.physics_dt)
        zero = torch.zeros_like(self.command)
        for _ in range(round(self.nav.settle_seconds/self.physics_dt)):
            self.controller.apply(zero)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(self.physics_dt)
        self.settled_root = self.robot.data.root_state_w.clone()
        if bool((self.robot.data.projected_gravity_b[:, 2] > -0.5).any()):
            raise RuntimeError('Spot fell during initial stance calibration')
        self.standing_clearance = self.settled_root[:, 2]-self.scene.env_origins[:, 2]-float(flat[3])
        if not bool(((self.standing_clearance>.3)&(self.standing_clearance<.8)).all()):
            raise RuntimeError('Invalid calibrated standing clearance')
        self.settled_joint_pos = self.robot.data.joint_pos.clone()
        self.settled_joint_vel = torch.zeros_like(self.robot.data.joint_vel)
        self.settled_root[:, 7:] = 0
        self.controller.counter = 0

    def _measured_power(self):
        effort = self.robot.root_physx_view.get_dof_projected_joint_forces()
        speed = self.robot.data.joint_vel
        if effort.shape != speed.shape:
            raise RuntimeError('Joint effort/velocity mismatch')
        power = (effort*speed).abs().sum(1)
        if not bool(torch.isfinite(power).all()):
            raise RuntimeError('Non-finite mechanical power')
        return power

    def _distance(self):
        return (self.goal_w[:, :2]-self.robot.data.root_pos_w[:, :2]).norm(dim=1)

    def yaw(self):
        q = self.robot.data.root_quat_w
        return torch.atan2(2*(q[:, 0]*q[:, 3]+q[:, 1]*q[:, 2]),
                           1-2*(q[:, 2]**2+q[:, 3]**2))

    def _sync_cameras(self):
        rot = self.robot.data.root_quat_w
        offset = torch.tensor((0.42, 0, 0.07), device=self.device).expand(self.num_envs, -1)
        half = math.radians(7.5)
        pitch = torch.tensor((math.cos(half), 0, math.sin(half), 0), device=self.device)
        self.camera.set_world_poses(self.robot.data.root_pos_w+quat_apply(rot, offset),
                                   quat_mul(rot, pitch.expand(self.num_envs, -1)), convention='world')

    def _pre_physics_step(self, actions):
        if actions.shape != (self.num_envs, 2) or not bool(torch.isfinite(actions).all()):
            raise ValueError('Expected finite N x 2 navigation actions')
        self.step_was_alignment = self.alignment.active.clone()
        error = heading_error(self.goal_w[:, :2], self.robot.data.root_pos_w[:, :2], self.yaw())
        self.previous_action = self.alignment.actions(actions, error)
        self.command = navigation_commands(self.previous_action)
        self.step_energy.zero_()

    def _apply_action(self):
        self.controller.apply(self.command)

    def step(self, action):
        """Same auto-reset contract as DirectRLEnv, but render AFTER final pose synchronization.

        These fixed configs intentionally have no event/noise managers. Integrate each of
        the 40 actual physics intervals once, including the last one, before resets.
        """
        self._pre_physics_step(action.to(self.device))
        power = self._measured_power()
        for _ in range(self.cfg.decimation):
            self._sim_step_counter += 1
            self._apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(self.physics_dt)
            next_power = self._measured_power()
            self.step_energy += 0.5*(power+next_power)*self.physics_dt
            power = next_power
        self.episode_length_buf += 1
        self.common_step_counter += 1
        policy_mask = ~self.step_was_alignment
        applied_actions = self.previous_action.clone()
        self.navigation_steps += policy_mask.long()
        self.alignment.advance(heading_error(self.goal_w[:, :2], self.robot.data.root_pos_w[:, :2], self.yaw()))
        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()
        # Capture pre-reset episode metrics. Rewards/components must never alias reset buffers.
        self.extras['episodes'] = self._episode_records()
        self.extras['policy_mask'] = policy_mask.clone()
        self.extras['applied_actions'] = applied_actions
        reset_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(reset_ids):
            self._reset_idx(reset_ids)
        self.obs_buf = self._get_observations()
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _get_dones(self):
        pos = self.robot.data.root_pos_w-self.scene.env_origins
        self.out_of_bounds = ((pos[:, 0] < self.geometry.world_x_bounds[0]) |
                              (pos[:, 0] > self.geometry.world_x_bounds[1]) |
                              (pos[:, 1] < self.geometry.world_y_bounds[0]) |
                              (pos[:, 1] > self.geometry.world_y_bounds[1]))
        height = terrain_height(self)
        clearance = self.robot.data.root_pos_w[:, 2]-height
        self.fallen = ((clearance < self.nav.minimum_body_height_m) |
                       (self.robot.data.projected_gravity_b[:, 2] > -0.5) |
                       (~torch.isfinite(height) & ~self.out_of_bounds))
        unsafe = self.fallen | self.out_of_bounds
        self.success = (self._distance() <= self.nav.goal_radius_m) & ~unsafe
        self.alignment_timeout = self.alignment.failed & ~self.success & ~unsafe
        self.timeout = (self.episode_length_buf >= self.deadline_steps) & ~self.success & ~unsafe & ~self.alignment_timeout
        # Per-mission finite deadline is a true task terminal.
        terminal = unsafe | self.success | self.timeout | self.alignment_timeout
        return terminal, torch.zeros_like(terminal)

    def _get_rewards(self):
        distance = self._distance()
        progress = self.previous_distance-distance
        self.previous_distance = distance.clone()
        xy = self.robot.data.root_pos_w[:, :2]
        travelled = (xy-self.previous_xy).norm(dim=1)
        self.path_length += travelled
        self.alignment_path += travelled*self.step_was_alignment
        self.previous_xy = xy.clone()
        self.episode_energy += self.step_energy
        self.alignment_energy += self.step_energy*self.step_was_alignment
        beta = math.exp(-self.step_dt/self.nav.power_filter_seconds)
        self.power_ema = beta*self.power_ema + (1-beta)*self.step_energy/self.step_dt
        failure = self.fallen | self.out_of_bounds | self.timeout | self.alignment_timeout
        heading_reward = orientation_penalty(
            heading_error(self.goal_w[:, :2], xy, self.yaw()), ~self.step_was_alignment,
            self.step_dt, self.nav.heading_penalty_start_deg, self.nav.heading_penalty_per_second)
        nav = (self.nav.progress_reward_per_m*progress - self.nav.time_penalty_per_step
               + self.nav.success_bonus*self.success.float() - self.nav.failure_penalty*failure.float()
               + heading_reward)
        cost = self.step_energy/self.nav.energy_reference_j
        self.episode_nav_return += nav
        self.episode_cost += cost
        self.extras = dict(reward_components=torch.stack((nav, cost), dim=1).clone(),
                           heading_reward=heading_reward.clone())
        return nav-self.nav.energy_weight*cost

    def _episode_records(self):
        records = []
        for i in self.reset_buf.nonzero(as_tuple=False).flatten().tolist():
            reason = ('goal' if self.success[i] else 'out_of_bounds' if self.out_of_bounds[i]
                      else 'fallen' if self.fallen[i] else 'alignment_timeout' if self.alignment_timeout[i] else 'timeout')
            mission = self.missions[i]
            records.append(dict(env_id=i, layout=self.conditions[i].layout.name, reason=reason,
                                success=int(self.success[i]), steps=int(self.episode_length_buf[i]),
                                alignment_steps=int(self.alignment.steps[i]), navigation_steps=int(self.navigation_steps[i]),
                                alignment_energy_j=float(self.alignment_energy[i]),
                                navigation_energy_j=float(self.episode_energy[i]-self.alignment_energy[i]),
                                alignment_path_m=float(self.alignment_path[i]),
                                navigation_path_m=float(self.path_length[i]-self.alignment_path[i]),
                                alignment_success=int(self.alignment.succeeded[i]),
                                alignment_end_error_rad=float(self.alignment.end_error[i]),
                                final_x=float(self.robot.data.root_pos_w[i, 0]-self.scene.env_origins[i, 0]),
                                final_y=float(self.robot.data.root_pos_w[i, 1]-self.scene.env_origins[i, 1]),
                                energy_j=float(self.episode_energy[i]), path_m=float(self.path_length[i]),
                                final_distance_m=float(self._distance()[i]),
                                nav_return=float(self.episode_nav_return[i]), energy_cost=float(self.episode_cost[i]),
                                band=mission['band'], start_id=mission['start_id'], goal_id=mission['goal_id'],
                                start_x=mission['start'][0], start_y=mission['start'][1],
                                goal_x=mission['goal'][0], goal_y=mission['goal'][1], initial_yaw=mission['yaw'],
                                start_terrain=TERRAIN_NAMES[int(mission['start'][4])],
                                goal_terrain=TERRAIN_NAMES[int(mission['goal'][4])],
                                initial_distance_m=mission['distance_m'],
                                deadline_s=mission['deadline_steps']*self.step_dt))
        return records

    def _get_observations(self):
        # One synchronized RGB/depth acquisition at the final, possibly reset, robot pose.
        self._sync_cameras()
        self.sim.render()
        self.camera.update(0.0, force_recompute=True)
        d = self.robot.data
        heading = self.yaw()
        delta = quat_apply_inverse(yaw_quat(d.root_quat_w), self.goal_w-d.root_pos_w)
        time_left = (1-self.episode_length_buf.float()/self.deadline_steps.float()).clamp(0, 1)
        state = torch.cat((delta[:, :2], d.root_lin_vel_b, d.root_ang_vel_b,
                           d.projected_gravity_b, self.previous_action,
                           self.power_ema[:, None], time_left[:, None]), dim=1)
        cells = self.vision.extract(self.camera, d.root_pos_w, heading, d.body_link_pos_w,
                                    self.episode_length_buf.float()*self.step_dt)
        actor = torch.cat((state, cells.flatten(1)), dim=1).float()
        critic = torch.cat((actor, self.contacts.extract(self, heading)), dim=1).float()
        if actor.shape != (self.num_envs, OBSERVATION_SIZE) or critic.shape != (self.num_envs, CRITIC_SIZE):
            raise RuntimeError(f'Unexpected observation dimensions {actor.shape}, {critic.shape}')
        if not bool(torch.isfinite(critic).all()):
            raise RuntimeError('Non-finite observations')
        self.visual_valid = cells[..., 36].sum(1)
        self.geometry_valid = cells[..., 37].sum(1)
        self.visual_remembered = ((cells[..., 36] > .5) & (cells[..., 38] > 1e-6)).sum(1)
        self.geometry_remembered = ((cells[..., 37] > .5) & (cells[..., 39] > 1e-6)).sum(1)
        return {'policy': actor, 'critic': critic}

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        root = self.settled_root[env_ids].clone()
        for j, env_id in enumerate(env_ids.detach().cpu().tolist()):
            if self.evaluating:
                mission = self.sampler.evaluation_mission(env_id, int(self.eval_indices[env_id]), self.eval_seed, self.eval_stage)
                self.eval_indices[env_id] += 1
            else:
                mission = self.sampler.training(self.curriculum.level)
            self.missions[env_id] = mission
            if self.evaluating and self.num_envs == 1:
                print(f'[MISSION] band={mission["band"]} '
                      f'A=({mission["start"][0]:.2f}, {mission["start"][1]:.2f}) '
                      f'B=({mission["goal"][0]:.2f}, {mission["goal"][1]:.2f}) '
                      f'distance={mission["distance_m"]:.2f} m', flush=True)
            self.deadline_steps[env_id] = mission['deadline_steps']
            root[j, :2] = self.scene.env_origins[env_id, :2] + torch.tensor(mission['start'][:2], device=self.device)
            # Upright calibrated stance above the highest sampled footprint surface.
            # No global simulation steps during partial resets: other episodes are unaffected.
            root[j, 2] = self.scene.env_origins[env_id, 2]+mission['start'][3]+self.standing_clearance[env_id]+.02
            yaw = mission['yaw']
            root[j, 3:7] = torch.tensor([math.cos(yaw/2), 0., 0., math.sin(yaw/2)], device=self.device)
            root[j, 7:] = 0
            self.goal_w[env_id, :2] = self.scene.env_origins[env_id, :2]+torch.tensor(mission['goal'][:2], device=self.device)
            self.goal_w[env_id, 2] = self.scene.env_origins[env_id, 2]+mission['goal'][3]
        self.robot.write_root_state_to_sim(root, env_ids)
        self.robot.write_joint_state_to_sim(self.settled_joint_pos[env_ids], self.settled_joint_vel[env_ids], env_ids=env_ids)
        self.robot.reset(env_ids)
        self.robot.update(self.physics_dt)
        self.controller.reset(env_ids)
        self.vision.reset(env_ids)
        self.alignment.reset(env_ids)
        self.alignment_energy[env_ids] = 0
        self.alignment_path[env_ids] = 0
        self.navigation_steps[env_ids] = 0
        self.previous_action[env_ids] = 0
        self.command[env_ids] = 0
        for name in ('power_ema', 'step_energy', 'episode_energy', 'path_length',
                     'episode_nav_return', 'episode_cost'):
            getattr(self, name)[env_ids] = 0
        self.previous_xy[env_ids] = root[:, :2]
        self.previous_distance[env_ids] = (self.goal_w[env_ids, :2]-root[:, :2]).norm(dim=1)
