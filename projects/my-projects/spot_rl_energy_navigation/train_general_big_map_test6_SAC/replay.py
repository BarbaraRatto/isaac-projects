"""CPU float32 ring buffer of raw observations; no duplicated actor in critic storage."""
from pathlib import Path
import numpy as np
import torch
from train_general_big_map_test6_SAC.config import OBSERVATION_SIZE, PRIVILEGED_SIZE, CRITIC_SIZE


class ReplayBuffer:
    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError('Replay capacity must be positive')
        self.capacity = int(capacity)
        self.size = self.position = self.total = 0
        shapes = dict(actor=(OBSERVATION_SIZE,), privileged=(PRIVILEGED_SIZE,),
                      next_actor=(OBSERVATION_SIZE,), next_privileged=(PRIVILEGED_SIZE,),
                      actions=(2,), rewards=(), terminals=())
        self.arrays = {key: np.empty((self.capacity, *shape), dtype=np.bool_ if key == 'terminals' else np.float32)
                       for key, shape in shapes.items()}

    @property
    def nbytes(self):
        return sum(array.nbytes for array in self.arrays.values())

    def add(self, actor, critic, actions, rewards, terminals, next_actor, next_critic):
        values = dict(actor=actor, privileged=critic[:, OBSERVATION_SIZE:], actions=actions,
                      rewards=rewards, terminals=terminals, next_actor=next_actor,
                      next_privileged=next_critic[:, OBSERVATION_SIZE:])
        count = len(actor)
        if not 0 < count <= self.capacity:
            raise ValueError('Replay insertion size out of range')
        if critic.shape != (count, CRITIC_SIZE) or next_critic.shape != (count, CRITIC_SIZE):
            raise ValueError('Critic observation size mismatch')
        for key, value in values.items():
            value = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
            if value.shape != (count, *self.arrays[key].shape[1:]):
                raise ValueError(f'Replay shape mismatch: {key}')
            if key != 'terminals' and not np.isfinite(value).all():
                raise ValueError(f'Non-finite replay data: {key}')
            values[key] = value
        ids = (np.arange(count) + self.position) % self.capacity
        for key, value in values.items():
            self.arrays[key][ids] = value
        self.position = (self.position + count) % self.capacity
        self.size = min(self.capacity, self.size + count)
        self.total += count

    def sample(self, count, device):
        if not self.size:
            raise ValueError('Cannot sample an empty replay')
        ids = np.random.randint(self.size, size=count)
        batch = {key: torch.as_tensor(value[ids], device=device) for key, value in self.arrays.items()}
        batch['critic'] = torch.cat((batch['actor'], batch.pop('privileged')), dim=1)
        batch['next_critic'] = torch.cat((batch['next_actor'], batch.pop('next_privileged')), dim=1)
        return batch

    def save(self, path):
        """Uncompressed final/interrupted snapshot; periodic model checkpoints omit replay."""
        path = Path(path)
        temporary = path.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.savez(stream, capacity=self.capacity, size=self.size, position=self.position, total=self.total,
                     **{key: value[:self.size] for key, value in self.arrays.items()})
        temporary.replace(path)

    def load(self, path):
        with np.load(path, allow_pickle=False) as stored:
            if int(stored['capacity']) != self.capacity:
                raise ValueError('Replay capacity mismatch')
            size, position, total = (int(stored[k]) for k in ('size', 'position', 'total'))
            if not 0 <= size <= self.capacity or not 0 <= position < self.capacity or size != min(total, self.capacity) or position != total % self.capacity:
                raise ValueError('Invalid replay metadata')
            for key, value in self.arrays.items():
                data = stored[key]
                if data.shape != value[:size].shape or data.dtype != value.dtype:
                    raise ValueError(f'Replay snapshot shape/dtype mismatch: {key}')
                value[:size] = data
            self.size, self.position, self.total = size, position, total


@torch.no_grad()
def collect_step(env, policy, obs, replay, device, remaining, warmup=False):
    """Keep only policy-controlled rows; all true terminals mask auto-reset next states."""
    actor = obs['policy'].to(device).detach().clone()
    critic = obs['critic'].to(device).detach().clone()
    if warmup:
        actions = torch.empty((len(actor), 2), device=device).uniform_(-1, 1)
    else:
        actions = policy.act(actor, critic)[0]
    next_obs, _, terminals, truncated, info = env.step(actions)
    if bool(truncated.any()):
        raise RuntimeError('Unexpected truncation; this experiment uses true terminal deadlines')
    mask = info['policy_mask'].to(device)
    ids = mask.nonzero(as_tuple=False).flatten()[:remaining]
    if len(ids):
        components = info['reward_components'].to(device)
        from train_general_big_map_test6_SAC.config import NAV_CFG
        rewards = components[:, 0] - NAV_CFG.energy_weight * components[:, 1]
        applied = info['applied_actions'].to(device)
        torch.testing.assert_close(applied[ids], actions[ids])
        replay.add(actor[ids], critic[ids], applied[ids], rewards[ids], terminals.to(device)[ids],
                   next_obs['policy'].to(device)[ids], next_obs['critic'].to(device)[ids])
        # Raw replay data are normalized at sampling time with the current statistics.
        policy.normalizer.update(actor[ids], critic[ids])
    return next_obs, dict(count=len(ids), alignment=int((~mask).sum()),
                          discarded=max(0, int(mask.sum())-len(ids)),
                          action_saturation=float((actions[ids].abs() > .98).float().mean()) if len(ids) else 0.,
                          episodes=info['episodes'])
