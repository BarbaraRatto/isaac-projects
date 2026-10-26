"""Two objective GAE and PPO. No SB3 scalar-reward buffer or hidden reward rescaling."""
from dataclasses import asdict

import torch
from train_general_big_map_test3.config import PPO_CFG, NAV_CFG
from train_general_big_map_test3.networks import ActorCritic, squashed_log_prob


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


def packed_gae(rewards, values, terminals, next_values, env_ids, gamma, lam):
    """Chronologically packed navigation transitions; alignment never enters recursion.

    Consecutive entries for one env are consecutive RL steps or separated by a
    terminal/reset. Each row stores its own post-step value, including at cutoff.
    """
    result = torch.zeros_like(rewards)
    carry = torch.zeros((int(env_ids.max())+1, 2), device=values.device)
    for t in reversed(range(len(rewards))):
        i = env_ids[t]
        live = (~terminals[t]).float()
        delta = rewards[t]+gamma*live*next_values[t]-values[t]
        result[t] = delta+gamma*lam*live*carry[i]
        carry[i] = result[t]
    return result, result+values


def explained_variance(prediction, target):
    variance = target.var(unbiased=False)
    if float(variance) < 1e-8:
        return None
    return float(1-(target-prediction).var(unbiased=False)/variance)


class PPO:
    def __init__(self, device='cpu', settings=PPO_CFG):
        self.device = torch.device(device)
        self.cfg = settings
        self.policy = ActorCritic().to(self.device)
        self.actor_parameters = [*self.policy.actor.parameters(), self.policy.log_std]
        self.critic_parameters = list(self.policy.critic.parameters())
        self.actor_optimizer = torch.optim.Adam(self.actor_parameters, lr=settings.learning_rate, eps=1e-5)
        self.critic_optimizer = torch.optim.Adam(self.critic_parameters, lr=settings.learning_rate, eps=1e-5)
        self.transitions = self.updates = self.actor_updates = 0

    def update(self, batch, bootstrap=None):
        cfg = self.cfg
        if 'env_ids' in batch:
            advantages, returns = packed_gae(batch['rewards'], batch['values'], batch['terminals'],
                                             batch['next_values'], batch['env_ids'], cfg.gamma, cfg.gae_lambda)
            flat = batch
        else:
            # Also accept rectangular batches for numerical regression checks.
            advantages, returns = gae(batch['rewards'], batch['values'], batch['terminals'],
                                      bootstrap, cfg.gamma, cfg.gae_lambda)
            flat = {key: value.flatten(0, 1) for key, value in batch.items()}
            advantages, returns = advantages.flatten(0, 1), returns.flatten(0, 1)
        combined = advantages[:, 0]-NAV_CFG.energy_weight*advantages[:, 1]
        combined = (combined-combined.mean())/(combined.std(unbiased=False)+1e-8)
        actor = self.policy.normalizer.actor(flat['actor'])
        critic = self.policy.normalizer.critic(flat['critic'])
        latent, old_log = flat['latent'], flat['log_prob']
        size = len(actor)
        with torch.no_grad():
            before = self.policy.critic(critic)
        actor_metrics, critic_metrics, actor_norms, critic_norms = [], [], [], []
        stopped, stop_kl = False, 0.
        for _ in range(cfg.epochs):
            order = torch.randperm(size, device=self.device)
            for start in range(0, size, cfg.batch_size):
                ids = order[start:start+cfg.batch_size]
                if not stopped:
                    distribution = self.policy.distribution(actor[ids])
                    log_prob = squashed_log_prob(distribution, latent[ids])
                    log_ratio = log_prob-old_log[ids]
                    ratio = log_ratio.exp()
                    with torch.no_grad():
                        kl = ((ratio-1)-log_ratio).mean()
                    if float(kl) > cfg.target_kl:
                        stopped, stop_kl = True, float(kl)
                    else:
                        surrogate = torch.minimum(ratio*combined[ids],
                            ratio.clamp(1-cfg.clip_range, 1+cfg.clip_range)*combined[ids])
                        entropy = -squashed_log_prob(distribution, distribution.rsample()).mean()
                        actor_loss = -surrogate.mean()-cfg.entropy_coef*entropy
                        if not torch.isfinite(actor_loss):
                            raise RuntimeError('Non-finite actor loss')
                        self.actor_optimizer.zero_grad(set_to_none=True)
                        actor_loss.backward()
                        norm = torch.nn.utils.clip_grad_norm_(self.actor_parameters, cfg.max_grad_norm,
                                                              error_if_nonfinite=True)
                        self.actor_optimizer.step()
                        with torch.no_grad():
                            self.policy.log_std.clamp_(-5, 2)
                        actor_norms.append(float(norm))
                        actor_metrics.append([float(-surrogate.mean().detach()), float(entropy.detach()),
                                              float(kl), float(((ratio-1).abs()>cfg.clip_range).float().mean())])
                        self.actor_updates += 1
                # Critic completes its epochs even if the actor stopped on KL.
                values = self.policy.critic(critic[ids])
                value_loss = (values-returns[ids]).square().mean(0)
                loss = cfg.value_coef*value_loss.sum()
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite critic loss')
                self.critic_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(self.critic_parameters, cfg.max_grad_norm,
                                                      error_if_nonfinite=True)
                self.critic_optimizer.step()
                critic_norms.append(float(norm))
                critic_metrics.append(value_loss.detach())
                self.updates += 1
        def mean(items):
            return sum(items)/len(items) if items else 0.
        a = torch.tensor(actor_metrics).mean(0).tolist() if actor_metrics else [0.]*4
        c = torch.stack(critic_metrics).mean(0).tolist()
        result = dict(actor_loss=a[0], entropy=a[1], kl=a[2], clip_fraction=a[3],
                      nav_value_loss=c[0], energy_value_loss=c[1], early_stop=int(stopped), stop_kl=stop_kl,
                      minibatch_updates=len(critic_metrics), total_updates=self.updates,
                      actor_minibatch_updates=len(actor_metrics), critic_minibatch_updates=len(critic_metrics),
                      actor_grad_norm=mean(actor_norms), critic_grad_norm=mean(critic_norms),
                      actor_clip_rate=mean([float(n>cfg.max_grad_norm) for n in actor_norms]),
                      critic_clip_rate=mean([float(n>cfg.max_grad_norm) for n in critic_norms]))
        with torch.no_grad():
            after = self.policy.critic(critic)
            distribution = self.policy.distribution(actor)
            ratio_log = squashed_log_prob(distribution, latent)-old_log
            result['full_batch_kl'] = float((ratio_log.exp()-1-ratio_log).mean())
            for index, name in enumerate(('nav', 'energy')):
                for label, predictions in (('before', before), ('after', after)):
                    result[f'{name}_ev_{label}'] = explained_variance(predictions[:, index], returns[:, index])
                    result[f'{name}_mse_{label}'] = float((predictions[:, index]-returns[:, index]).square().mean())
        return result

    def state_dict(self):
        return dict(policy=self.policy.state_dict(), actor_optimizer=self.actor_optimizer.state_dict(),
                    critic_optimizer=self.critic_optimizer.state_dict(), transitions=self.transitions,
                    updates=self.updates, actor_updates=self.actor_updates, settings=asdict(self.cfg))

    def load_state_dict(self, state):
        if state['settings'] != asdict(self.cfg):
            raise ValueError('PPO configuration differs from checkpoint')
        self.policy.load_state_dict(state['policy'])
        self.actor_optimizer.load_state_dict(state['actor_optimizer'])
        self.critic_optimizer.load_state_dict(state['critic_optimizer'])
        self.transitions, self.updates = state['transitions'], state['updates']
        self.actor_updates = state['actor_updates']
