"""Asynchronous collection: autonomous startup actions are never PPO samples."""
import torch
from train_general_big_map_test3.config import ALIGN_CFG, CELL_SIZE, LOCAL_CELLS


@torch.no_grad()
def collect_rollout(env, policy, obs, device, target, remaining, initial_transitions=0, app=None):
    fields = ('actor', 'critic', 'latent', 'log_prob', 'values', 'rewards', 'terminals', 'next_values', 'env_ids')
    storage = {key: [] for key in fields}
    count = physical_steps = alignment_transitions = no_navigation = discarded = 0
    records = []
    while count < target:
        if app is not None and not app.is_running():
            raise KeyboardInterrupt
        actor, critic = obs['policy'].to(device), obs['critic'].to(device)
        action, latent, log_prob, values = policy.act(actor, critic)
        obs, _, terminal, truncated, info = env.step(action)
        if bool(truncated.any()):
            raise RuntimeError('Unexpected truncated episode')
        mask = info['policy_mask'].to(device)
        ids = mask.nonzero(as_tuple=False).flatten()
        alignment_transitions += int((~mask).sum())
        physical_steps += 1
        no_navigation = 0 if len(ids) else no_navigation+1
        if no_navigation >= ALIGN_CFG.max_steps_without_navigation:
            raise RuntimeError('No RL transitions for 60 s of simulation: inspect alignment failures before training')
        # Keep complete simultaneous steps at ordinary rollout boundaries. Only
        # the final budget boundary may discard <= num_envs-1 additional samples.
        available = remaining-count
        discarded += max(0, len(ids)-available)
        ids = ids[:available]
        if len(ids):
            next_values = policy.critic(policy.normalizer.critic(obs['critic'].to(device)))
            data = dict(actor=actor, critic=critic, latent=latent, log_prob=log_prob, values=values,
                        rewards=info['reward_components'].to(device), terminals=terminal.to(device),
                        next_values=next_values)
            for key, value in data.items():
                storage[key].append(value[ids].detach().clone())
            storage['env_ids'].append(ids.clone())
            count += len(ids)
        records.extend(dict(transitions=initial_transitions+count, **record) for record in info['episodes'])
    batch = {key: torch.cat(items) for key, items in storage.items()}
    cells = batch['actor'][:, 15:].reshape(-1, LOCAL_CELLS, CELL_SIZE)
    stats = dict(rollout_steps=physical_steps, rollout_transitions=count,
                 alignment_transitions=alignment_transitions, discarded_final_samples=discarded,
                 visual_cells=float(cells[..., 36].sum(1).mean()),
                 geometry_cells=float(cells[..., 37].sum(1).mean()),
                 visual_remembered_cells=float(((cells[..., 36]>.5)&(cells[..., 38]>1e-6)).float().sum(1).mean()),
                 geometry_remembered_cells=float(((cells[..., 37]>.5)&(cells[..., 39]>1e-6)).float().sum(1).mean()),
                 action_saturation=float((batch['latent'].tanh().abs()>.98).float().mean()))
    return batch, obs, records, stats
