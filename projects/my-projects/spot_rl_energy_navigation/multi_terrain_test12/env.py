"""Isaac Lab environment with explicit sensor timing and true finite-horizon terminals."""
import math

import torch
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul, yaw_quat

from multi_terrain_test12.actions import navigation_commands
from multi_terrain_test12.config import NAV_CFG, FOOT_NAMES, OBSERVATION_SIZE, CRITIC_SIZE
from multi_terrain_test12.locomotion import BatchedSpotVelocityController
from multi_terrain_test12.privileged import PrivilegedContacts, terrain_height
from multi_terrain_test12.scene import setup_scene
from multi_terrain_test12.vision import LocalVision


class Test12Env(DirectRLEnv):
    def __init__(self, cfg, conditions, pca=None, dino_batch_size=4, model_path=None,
                 capture=False, **kwargs):
        self.conditions = conditions
        self.nav = NAV_CFG
        if len(conditions) != cfg.scene.num_envs or tuple(c.env_id for c in conditions) != tuple(range(len(conditions))):
            raise ValueError('Conditions must match and be ordered by environment index')
        super().__init__(cfg, **kwargs)
        self.geometry = conditions[0].case.geometry
        self.foot_ids = [self.robot.body_names.index(name) for name in FOOT_NAMES]
        self.controller = BatchedSpotVelocityController(self.robot, self.num_envs, self.device)
        self.vision = LocalVision(self.device, pca, dino_batch_size, model_path, capture)
        self.contacts = PrivilegedContacts(self)
        self.goal_w = self.scene.env_origins.clone()
        self.goal_w[:, :2] += torch.tensor([c.goal_xy for c in conditions], device=self.device)
        self.previous_action = torch.zeros((self.num_envs, 2), device=self.device)
        self.command = torch.zeros((self.num_envs, 3), device=self.device)
        self.previous_xy = torch.zeros((self.num_envs, 2), device=self.device)
        for name in ('previous_distance', 'power_ema', 'step_energy', 'episode_energy',
                     'path_length', 'episode_nav_return', 'episode_cost', 'right_path_m', 'left_path_m'):
            setattr(self, name, torch.zeros(self.num_envs, device=self.device))
        self._settle_once()

    def _setup_scene(self):
        setup_scene(self)

    def _settle_once(self):
        zero = torch.zeros_like(self.command)
        for _ in range(round(self.nav.settle_seconds/self.physics_dt)):
            self.controller.apply(zero)
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(self.physics_dt)
        self.settled_root = self.robot.data.root_state_w.clone()
        if bool((self.robot.data.projected_gravity_b[:, 2] > -0.5).any()):
            raise RuntimeError('Spot fell during settling')
        self.settled_joint_pos = self.robot.data.joint_pos.clone()
        self.settled_joint_vel = self.robot.data.joint_vel.clone()
        starts = torch.tensor([c.start_xy for c in self.conditions], device=self.device)
        goals = torch.tensor([c.goal_xy for c in self.conditions], device=self.device)
        yaw = torch.atan2(goals[:, 1]-starts[:, 1], goals[:, 0]-starts[:, 0])
        self.settled_root[:, :2] = self.scene.env_origins[:, :2]+starts
        self.settled_root[:, 3] = (yaw/2).cos()
        self.settled_root[:, 4:6] = 0
        self.settled_root[:, 6] = (yaw/2).sin()
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
        self.previous_action = actions.clamp(-1, 1).clone()
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
        self.reset_terminated[:], self.reset_time_outs[:] = self._get_dones()
        self.reset_buf = self.reset_terminated | self.reset_time_outs
        self.reward_buf = self._get_rewards()
        # Capture pre-reset episode metrics. Rewards/components must never alias reset buffers.
        self.extras['episodes'] = self._episode_records()
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
        self.timeout = (self.episode_length_buf >= self.nav.max_episode_steps) & ~self.success & ~unsafe
        # 40 seconds is a mission deadline; it is NOT an artificial truncation.
        terminal = unsafe | self.success | self.timeout
        return terminal, torch.zeros_like(terminal)

    def _get_rewards(self):
        distance = self._distance()
        progress = self.previous_distance-distance
        self.previous_distance = distance.clone()
        xy = self.robot.data.root_pos_w[:, :2]
        travelled = (xy-self.previous_xy).norm(dim=1)
        self.path_length += travelled
        # Evaluation diagnostics only: branch membership from base position.
        local_xy = xy-self.scene.env_origins[:, :2]
        col = torch.floor((local_xy[:, 0]+1)/2).long()
        row = torch.floor((local_xy[:, 1]+1)/2).long()
        right = ((row == 0) & ((col == 1) | (col == 2))) | ((row == 1) & (col == 2))
        left = ((col == 0) & ((row == 1) | (row == 2))) | ((row == 2) & (col == 1))
        self.right_path_m += travelled*right
        self.left_path_m += travelled*left
        self.previous_xy = xy.clone()
        self.episode_energy += self.step_energy
        beta = math.exp(-self.step_dt/self.nav.power_filter_seconds)
        self.power_ema = beta*self.power_ema + (1-beta)*self.step_energy/self.step_dt
        failure = self.fallen | self.out_of_bounds | self.timeout
        nav = (self.nav.progress_reward_per_m*progress - self.nav.time_penalty_per_step
               + self.nav.success_bonus*self.success.float() - self.nav.failure_penalty*failure.float())
        cost = self.step_energy/self.nav.energy_reference_j
        self.episode_nav_return += nav
        self.episode_cost += cost
        self.extras = dict(reward_components=torch.stack((nav, cost), dim=1).clone())
        return nav-self.nav.energy_weight*cost

    def _episode_records(self):
        records = []
        for i in self.reset_buf.nonzero(as_tuple=False).flatten().tolist():
            reason = ('goal' if self.success[i] else 'out_of_bounds' if self.out_of_bounds[i]
                      else 'fallen' if self.fallen[i] else 'timeout')
            rp, lp = float(self.right_path_m[i]), float(self.left_path_m[i])
            route = 'none' if max(rp, lp) < .1 else ('mixed' if min(rp, lp) > .25 else ('right' if rp > lp else 'left'))
            records.append(dict(env_id=i, layout=self.conditions[i].layout.name, reason=reason,
                                success=int(self.success[i]), steps=int(self.episode_length_buf[i]),
                                energy_j=float(self.episode_energy[i]), path_m=float(self.path_length[i]),
                                final_distance_m=float(self._distance()[i]),
                                nav_return=float(self.episode_nav_return[i]),
                                energy_cost=float(self.episode_cost[i]), route=route,
                                right_path_m=rp, left_path_m=lp,
                                asphalt_route=int(route in ('right', 'left') and route != self.conditions[i].layout.costly_side)))
        return records

    def _get_observations(self):
        # One synchronized RGB/depth acquisition at the final, possibly reset, robot pose.
        self._sync_cameras()
        self.sim.render()
        self.camera.update(0.0, force_recompute=True)
        d = self.robot.data
        heading = self.yaw()
        delta = quat_apply_inverse(yaw_quat(d.root_quat_w), self.goal_w-d.root_pos_w)
        time_left = (1-self.episode_length_buf.float()/self.nav.max_episode_steps).clamp(0, 1)
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
        root[:, :2] += (torch.rand((len(env_ids), 2), device=self.device)*2-1)*self.geometry.start_jitter_m
        self.robot.write_root_state_to_sim(root, env_ids)
        self.robot.write_joint_state_to_sim(self.settled_joint_pos[env_ids], self.settled_joint_vel[env_ids], env_ids=env_ids)
        self.robot.reset(env_ids)
        self.robot.update(self.physics_dt)
        self.controller.reset(env_ids)
        self.vision.reset(env_ids)
        self.previous_action[env_ids] = 0
        self.command[env_ids] = 0
        for name in ('power_ema', 'step_energy', 'episode_energy', 'path_length',
                     'episode_nav_return', 'episode_cost', 'right_path_m', 'left_path_m'):
            getattr(self, name)[env_ids] = 0
        self.previous_xy[env_ids] = root[:, :2]
        self.previous_distance[env_ids] = (self.goal_w[env_ids, :2]-root[:, :2]).norm(dim=1)
