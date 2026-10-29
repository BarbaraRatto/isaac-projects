"""Asymmetric SAC: scalar total reward, twin Q, target Q and automatic temperature."""
import copy
from dataclasses import asdict
import torch
from train_general_big_map_test6_SAC.config import SAC_CFG
from train_general_big_map_test6_SAC.networks import SACPolicy


def soft_bellman_target(rewards, terminals, next_q1, next_q2, next_log_prob, alpha, gamma):
    soft_value = torch.minimum(next_q1, next_q2) - alpha * next_log_prob
    return rewards + gamma * (~terminals).float() * soft_value


class SAC:
    def __init__(self, device='cpu', settings=SAC_CFG):
        self.device, self.cfg = torch.device(device), settings
        self.policy = SACPolicy().to(self.device)
        self.target_q1 = copy.deepcopy(self.policy.q1).requires_grad_(False)
        self.target_q2 = copy.deepcopy(self.policy.q2).requires_grad_(False)
        self.critic_parameters = [*self.policy.q1.parameters(), *self.policy.q2.parameters()]
        self.actor_optimizer = torch.optim.Adam(self.policy.actor.parameters(), lr=settings.learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic_parameters, lr=settings.learning_rate)
        self.log_alpha = torch.tensor(float(settings.initial_alpha), device=self.device).log().requires_grad_()
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=settings.learning_rate)
        self.transitions = self.updates = 0

    def update(self, batch):
        policy, cfg = self.policy, self.cfg
        actor = policy.normalizer.actor(batch['actor'])
        critic = policy.normalizer.critic(batch['critic'])
        alpha = self.log_alpha.exp().detach()
        with torch.no_grad():
            next_actor = policy.normalizer.actor(batch['next_actor'])
            next_critic = policy.normalizer.critic(batch['next_critic'])
            next_actions, _, next_log_prob = policy.sample(next_actor)
            target = soft_bellman_target(batch['rewards'], batch['terminals'],
                self.target_q1(next_critic, next_actions).squeeze(-1),
                self.target_q2(next_critic, next_actions).squeeze(-1), next_log_prob, alpha, cfg.gamma)
        q1 = policy.q1(critic, batch['actions']).squeeze(-1)
        q2 = policy.q2(critic, batch['actions']).squeeze(-1)
        q1_loss = (q1-target).square().mean()
        q2_loss = (q2-target).square().mean()
        critic_loss = .5*(q1_loss+q2_loss)
        self.critic_optimizer.zero_grad(set_to_none=True)
        if not torch.isfinite(critic_loss):
            raise RuntimeError('Non-finite SAC critic loss')
        critic_loss.backward()
        self.critic_optimizer.step()
        # Keep action gradients through Q, while not modifying critic parameters in the actor step.
        self.critic_optimizer.zero_grad(set_to_none=True)
        for p in self.critic_parameters:
            p.requires_grad_(False)
        try:
            actions, _, log_prob = policy.sample(actor)
            q = torch.minimum(policy.q1(critic, actions), policy.q2(critic, actions)).squeeze(-1)
            actor_loss = (alpha*log_prob-q).mean()
            if not torch.isfinite(actor_loss):
                raise RuntimeError('Non-finite SAC actor loss')
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_optimizer.step()
        finally:
            for p in self.critic_parameters:
                p.requires_grad_(True)
        alpha_loss = -(self.log_alpha*(log_prob.detach()+cfg.target_entropy)).mean()
        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()
        with torch.no_grad():
            for target_network, current in ((self.target_q1, policy.q1), (self.target_q2, policy.q2)):
                for target_p, current_p in zip(target_network.parameters(), current.parameters()):
                    target_p.lerp_(current_p, cfg.tau)
        self.updates += 1
        metrics = dict(actor_loss=float(actor_loss.detach()), q1_loss=float(q1_loss.detach()),
                       q2_loss=float(q2_loss.detach()), alpha_loss=float(alpha_loss.detach()),
                       alpha=float(self.log_alpha.detach().exp()), entropy=float(-log_prob.detach().mean()),
                       q_mean=float(.5*(q1.detach().mean()+q2.detach().mean())), target_q_mean=float(target.mean()))
        if not all(torch.isfinite(torch.tensor(v)) for v in metrics.values()):
            raise RuntimeError('Non-finite SAC metrics')
        return metrics

    def state_dict(self):
        return dict(policy=self.policy.state_dict(), target_q1=self.target_q1.state_dict(),
                    target_q2=self.target_q2.state_dict(), log_alpha=self.log_alpha.detach(),
                    actor_optimizer=self.actor_optimizer.state_dict(), critic_optimizer=self.critic_optimizer.state_dict(),
                    alpha_optimizer=self.alpha_optimizer.state_dict(), transitions=self.transitions,
                    updates=self.updates, settings=asdict(self.cfg))

    def load_state_dict(self, state):
        if state['settings'] != asdict(self.cfg):
            raise ValueError('SAC configuration differs from checkpoint')
        self.policy.load_state_dict(state['policy'])
        self.target_q1.load_state_dict(state['target_q1'])
        self.target_q2.load_state_dict(state['target_q2'])
        with torch.no_grad():
            self.log_alpha.copy_(state['log_alpha'])
        for name in ('actor_optimizer','critic_optimizer','alpha_optimizer'):
            getattr(self,name).load_state_dict(state[name])
        self.transitions, self.updates = state['transitions'], state['updates']
