"""Two objective GAE and PPO. No SB3 scalar-reward buffer or hidden reward rescaling."""
from dataclasses import asdict

import torch
from train_general_big_map_test1.config import PPO_CFG, NAV_CFG
from train_general_big_map_test1.networks import ActorCritic, squashed_log_prob


def gae(rewards, values, terminals, bootstrap, gamma, lam):
    """T x N x 2 rewards/values; true terminals cut BOTH bootstrap and GAE recursion."""
    advantage = torch.zeros_like(rewards)
    carry = torch.zeros_like(bootstrap)
    next_value = bootstrap
    for t in reversed(range(len(rewards))):
        live = (~terminals[t]).float()[:, None]
        delta = rewards[t]+gamma*live*next_value-values[t]
        carry = delta+gamma*lam*live*carry
        advantage[t] = carry
        next_value = values[t]
    return advantage, advantage+values


def rollout_length(remaining, num_envs, maximum):
    if remaining <= 0 or remaining % num_envs:
        raise ValueError('Remaining transition budget must be a positive multiple of num_envs')
    return min(maximum, remaining//num_envs)


class PPO:
    def __init__(self, device='cpu', settings=PPO_CFG):
        self.device = torch.device(device)
        self.cfg = settings
        self.policy = ActorCritic().to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=settings.learning_rate, eps=1e-5)
        self.transitions = 0
        self.updates = 0

    def update(self, batch, bootstrap):
        cfg = self.cfg
        advantages, returns = gae(batch['rewards'], batch['values'], batch['terminals'],
                                 bootstrap, cfg.gamma, cfg.gae_lambda)
        combined = advantages[..., 0]-NAV_CFG.energy_weight*advantages[..., 1]
        combined = (combined-combined.mean())/(combined.std(unbiased=False)+1e-8)
        # Normalization stays fixed across collection AND optimization for this rollout.
        actor = self.policy.normalizer.actor(batch['actor'].flatten(0, 1))
        critic = self.policy.normalizer.critic(batch['critic'].flatten(0, 1))
        latent = batch['latent'].flatten(0, 1)
        old_log = batch['log_prob'].flatten()
        targets, advantage = returns.flatten(0, 1), combined.flatten()
        size = len(actor)
        metrics = []
        stopped = False
        for _ in range(cfg.epochs):
            order = torch.randperm(size, device=self.device)
            for start in range(0, size, cfg.batch_size):
                ids = order[start:start+cfg.batch_size]
                distribution = self.policy.distribution(actor[ids])
                log_prob = squashed_log_prob(distribution, latent[ids])
                log_ratio = log_prob-old_log[ids]
                ratio = log_ratio.exp()
                with torch.no_grad():
                    kl = ((ratio-1)-log_ratio).mean()
                if kl > cfg.target_kl:
                    stopped = True
                    break
                surrogate = torch.minimum(ratio*advantage[ids],
                                           ratio.clamp(1-cfg.clip_range, 1+cfg.clip_range)*advantage[ids])
                values = self.policy.critic(critic[ids])
                value_loss = (values-targets[ids]).square().mean(0)
                # Monte Carlo entropy of the actual tanh-squashed distribution, with pathwise gradients.
                entropy = -squashed_log_prob(distribution, distribution.rsample()).mean()
                loss = -surrogate.mean()+cfg.value_coef*value_loss.sum()-cfg.entropy_coef*entropy
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite PPO loss')
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    self.policy.log_std.clamp_(-5, 2)
                metrics.append([float(-surrogate.mean().detach()), float(value_loss[0].detach()),
                                float(value_loss[1].detach()), float(entropy.detach()), float(kl),
                                float(((ratio-1).abs() > cfg.clip_range).float().mean())])
                self.updates += 1
            if stopped:
                break
        names = ('actor_loss', 'nav_value_loss', 'energy_value_loss', 'entropy', 'kl', 'clip_fraction')
        result = dict(zip(names, torch.tensor(metrics).mean(0).tolist())) if metrics else dict.fromkeys(names, 0.0)
        result.update(early_stop=int(stopped), minibatch_updates=len(metrics), total_updates=self.updates)
        return result

    def state_dict(self):
        return dict(policy=self.policy.state_dict(), optimizer=self.optimizer.state_dict(),
                    transitions=self.transitions, updates=self.updates, settings=asdict(self.cfg))

    def load_state_dict(self, state):
        if state['settings'] != asdict(self.cfg):
            raise ValueError('PPO configuration differs from checkpoint')
        self.policy.load_state_dict(state['policy'])
        self.optimizer.load_state_dict(state['optimizer'])
        self.transitions, self.updates = state['transitions'], state['updates']
