"""Deterministic episodes excluded from PPO buffers and normalization statistics."""
import numpy as np
import torch

from train_general_big_map_test3.artifacts import rng_state, restore_rng
from train_general_big_map_test3.config import NAV_CFG, MISSION_CFG
from train_general_big_map_test3.alignment import heading_error


@torch.no_grad()
def evaluate(env, policy, device, episodes_per_env=1, seed=MISSION_CFG.evaluation_seed, app=None, trajectory=None, stage=None):
    stage = env.curriculum.level if stage is None else int(stage)
    if not 0 <= stage < len(MISSION_CFG.distance_bands):
        raise ValueError('Invalid evaluation stage')
    previous_stage = env.eval_stage
    env.eval_stage = stage
    saved_rng = rng_state()
    previous_mode = env.evaluating
    previous_seed = env.eval_seed
    previous_indices = env.eval_indices.copy()
    env.evaluating = True
    env.eval_seed = seed
    env.eval_indices[:] = 0
    records = []
    completed = np.zeros(env.num_envs, dtype=int)
    try:
        obs, _ = env.reset(seed=seed)
        for _ in range(NAV_CFG.max_episode_steps*episodes_per_env+1):
            if app is not None and not app.is_running():
                raise KeyboardInterrupt
            action, _, _, _ = policy.act(obs['policy'].to(device), obs['critic'].to(device), deterministic=True)
            if trajectory is not None:
                positions = (env.robot.data.root_pos_w-env.scene.env_origins).cpu().numpy()
                error = heading_error(env.goal_w[:, :2], env.robot.data.root_pos_w[:, :2], env.yaw())
                actions = env.alignment.actions(action.to(env.device), error).cpu().numpy()
                goals = (env.goal_w-env.scene.env_origins).cpu().numpy()
                heading = env.yaw().cpu().numpy()
                for i in range(env.num_envs):
                    if completed[i] < episodes_per_env:
                        trajectory.write(dict(env_id=i, episode=int(completed[i]+1),
                                              step=int(env.episode_length_buf[i]), x=positions[i, 0], y=positions[i, 1],
                                              energy_j=float(env.episode_energy[i]),
                                              phase='alignment' if env.alignment.active[i] else 'navigation',
                                              goal_x=goals[i, 0], goal_y=goals[i, 1], yaw=heading[i],
                                              command_vx=(actions[i, 0]+1)*NAV_CFG.max_forward_m_s/2,
                                              command_wz=actions[i, 1]*NAV_CFG.max_yaw_rad_s,
                                              action_forward=actions[i, 0], action_yaw=actions[i, 1]))
            obs, _, _, _, info = env.step(action)
            for record in info['episodes']:
                i = record['env_id']
                if completed[i] < episodes_per_env:
                    completed[i] += 1
                    records.append(dict(record, episode=int(completed[i])))
            if bool((completed >= episodes_per_env).all()):
                break
        if not bool((completed >= episodes_per_env).all()):
            raise RuntimeError('Evaluation did not finish within the mission deadline')
    finally:
        restore_rng(saved_rng)
        env.eval_stage = previous_stage
        env.evaluating = previous_mode
        env.eval_seed = previous_seed
        env.eval_indices[:] = previous_indices
    successes = [r for r in records if r['success']]
    success_rate = len(successes)/len(records)
    mean_energy = float(np.mean([r['energy_j'] for r in successes])) if successes else None
    summary = dict(stage=stage, stage_name=MISSION_CFG.stage_names[stage], episodes=len(records), successes=len(successes), success_rate=success_rate,
                   mean_success_energy_j=mean_energy,
                   mean_success_path_m=float(np.mean([r['path_m'] for r in successes])) if successes else None,
                   mean_success_seconds=float(np.mean([r['steps']*NAV_CFG.action_dt for r in successes])) if successes else None,
                   outcomes={reason: sum(r['reason'] == reason for r in records)
                             for reason in ('goal', 'fallen', 'out_of_bounds', 'timeout', 'alignment_timeout')})
    summary['alignment_success_rate'] = sum(r['alignment_success'] for r in records)/len(records)
    summary['mean_alignment_energy_j'] = float(np.mean([r['alignment_energy_j'] for r in records]))
    summary['mean_alignment_seconds'] = float(np.mean([r['alignment_steps']*NAV_CFG.action_dt for r in records]))
    summary['by_band'] = {}
    for band in sorted({r['band'] for r in records}):
        subset = [r for r in records if r['band'] == band]
        good = [r for r in subset if r['success']]
        summary['by_band'][str(band)] = dict(
            episodes=len(subset), success_rate=len(good)/len(subset),
            mean_success_energy_j=float(np.mean([r['energy_j'] for r in good])) if good else None,
            mean_success_path_m=float(np.mean([r['path_m'] for r in good])) if good else None,
            outcomes={reason: sum(r['reason'] == reason for r in subset)
                      for reason in ('goal', 'fallen', 'out_of_bounds', 'timeout', 'alignment_timeout')})
    return summary, records, (success_rate, -mean_energy if mean_energy is not None else float('-inf'))
