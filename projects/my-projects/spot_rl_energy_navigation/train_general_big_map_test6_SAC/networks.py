"""Independent actor/critic, mask-aware running normalization and bounded Gaussian policy."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

from train_general_big_map_test6_SAC.config import OBSERVATION_SIZE, CRITIC_SIZE, LAYERS, STATE_LAYERS, MAP_LAYERS, LOCAL_CELLS, CELL_SIZE


class RunningMoments(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.register_buffer('mean', torch.zeros(size, dtype=torch.float64))
        self.register_buffer('var', torch.ones(size, dtype=torch.float64))
        self.register_buffer('count', torch.full((size,), 1e-4, dtype=torch.float64))

    @torch.no_grad()
    def update(self, values, valid=None):
        values = values.double()
        valid = torch.ones_like(values, dtype=torch.bool) if valid is None else valid.bool()
        n = valid.sum(0).double()
        safe = torch.where(valid, values, 0)
        mean = safe.sum(0)/n.clamp_min(1)
        var = torch.where(valid, (values-mean)**2, 0).sum(0)/n.clamp_min(1)
        delta = mean-self.mean
        total = self.count+n
        combined = self.var*self.count+var*n+delta**2*self.count*n/total
        self.mean.add_(delta*n/total)
        self.var.copy_(combined/total)
        self.count.copy_(total)

    def forward(self, values):
        return ((values-self.mean.float())/torch.sqrt(self.var.float()+1e-6)).clamp(-10, 10)


class ObservationNormalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.state = RunningMoments(9)  # goal2, linear3, angular3, power1
        self.visual = RunningMoments(32)
        self.geometry = RunningMoments(4)
        self.privileged = RunningMoments(20)  # mu8, forces4, slip8; masks excluded

    @torch.no_grad()
    def update(self, actor, critic):
        state = torch.cat((actor[:, :8], actor[:, 13:14]), dim=1)
        self.state.update(state)
        cells = actor[:, 15:].reshape(-1, CELL_SIZE)
        self.visual.update(cells[:, :32], cells[:, 36:37].expand(-1, 32) > 0.5)
        self.geometry.update(cells[:, 32:36], cells[:, 37:38].expand(-1, 4) > 0.5)
        p = critic[:, OBSERVATION_SIZE:]
        contact = p[:, 8:12] > 0.5
        values = torch.cat((p[:, :8], p[:, 12:]), dim=1)
        masks = torch.cat((contact.repeat_interleave(2, dim=1), contact,
                           contact.repeat_interleave(2, dim=1)), dim=1)
        self.privileged.update(values, masks)

    def actor(self, raw):
        state = raw[:, :15].clone()
        normalized = self.state(torch.cat((raw[:, :8], raw[:, 13:14]), dim=1))
        state[:, :8] = normalized[:, :8]
        state[:, 13:14] = normalized[:, 8:9]
        cells = raw[:, 15:].reshape(-1, LOCAL_CELLS, CELL_SIZE)
        output = torch.cat((self.visual(cells[..., :32])*cells[..., 36:37],
                            self.geometry(cells[..., 32:36])*cells[..., 37:38], cells[..., 36:38],
                            cells[..., 38:40].clamp(0, 1)*cells[..., 36:38]), dim=-1)
        return torch.cat((state, output.flatten(1)), dim=1)

    def critic(self, raw):
        actor = self.actor(raw[:, :OBSERVATION_SIZE])
        p = raw[:, OBSERVATION_SIZE:]
        contact = p[:, 8:12]
        normalized = self.privileged(torch.cat((p[:, :8], p[:, 12:]), dim=1))
        return torch.cat((actor, normalized[:, :8]*contact.repeat_interleave(2, dim=1), contact,
                          normalized[:, 8:12]*contact,
                          normalized[:, 12:]*contact.repeat_interleave(2, dim=1)), dim=1)


def mlp(input_size, output_size, output_gain):
    modules = []
    for width in LAYERS:
        layer = nn.Linear(input_size, width)
        nn.init.orthogonal_(layer.weight, math.sqrt(2))
        nn.init.zeros_(layer.bias)
        modules.extend((layer, nn.ELU()))
        input_size = width
    layer = nn.Linear(input_size, output_size)
    nn.init.orthogonal_(layer.weight, output_gain)
    nn.init.zeros_(layer.bias)
    return nn.Sequential(*modules, layer)


class SplitNetwork(nn.Module):
    """State and map encoders with independent actor/critic weights."""
    def __init__(self, privileged=False):
        super().__init__()
        self.privileged = privileged
        def encoder(input_size, widths):
            layers = []
            for width in widths:
                layer = nn.Linear(input_size, width)
                nn.init.orthogonal_(layer.weight, math.sqrt(2))
                nn.init.zeros_(layer.bias)
                layers.extend((layer, nn.ELU()))
                input_size = width
            return nn.Sequential(*layers)
        self.state_encoder = encoder(41 if privileged else 15, STATE_LAYERS)
        self.map_encoder = encoder(OBSERVATION_SIZE-15, MAP_LAYERS)
        self.fusion = mlp(STATE_LAYERS[-1]+MAP_LAYERS[-1], 1 if privileged else 4, 1.0 if privileged else .01)

    def forward(self, observations, actions=None):
        state = (torch.cat((observations[:, :15], observations[:, OBSERVATION_SIZE:], actions), dim=1)
                 if self.privileged else observations[:, :15])
        cells = observations[:, 15:OBSERVATION_SIZE]
        return self.fusion(torch.cat((self.state_encoder(state), self.map_encoder(cells)), dim=1))


def squashed_log_prob(distribution, latent):
    # Stable log(1 - tanh(u)^2), including at extreme/saturated actions.
    correction = 2*(math.log(2)-latent-F.softplus(-2*latent))
    return (distribution.log_prob(latent)-correction).sum(-1)


class SACPolicy(nn.Module):
    """Actor observes only deployable inputs; independent Q1/Q2 also see privileges and action."""
    def __init__(self):
        super().__init__()
        self.normalizer = ObservationNormalizer()
        self.actor = SplitNetwork()
        self.q1 = SplitNetwork(privileged=True)
        self.q2 = SplitNetwork(privileged=True)

    def distribution(self, normalized_actor):
        mean, log_std = self.actor(normalized_actor).chunk(2, dim=-1)
        return Normal(mean, log_std.clamp(-20, 2).exp())

    def sample(self, normalized_actor, deterministic=False):
        distribution = self.distribution(normalized_actor)
        latent = distribution.mean if deterministic else distribution.rsample()
        return latent.tanh(), latent, squashed_log_prob(distribution, latent)

    @torch.no_grad()
    def act(self, raw_actor, raw_critic, deterministic=False):
        # Match the evaluation interface; critic outputs are intentionally not computed here.
        action, latent, log_prob = self.sample(self.normalizer.actor(raw_actor), deterministic)
        return action, latent, log_prob, torch.zeros((len(action), 2), device=action.device)


class ExportedActor(nn.Module):
    """Raw 3255-D deployable observations -> deterministic normalized [vx, wz] action."""
    def __init__(self, policy):
        super().__init__()
        self.normalizer = policy.normalizer
        self.actor = policy.actor

    def forward(self, observations):
        mean = self.actor(self.normalizer.actor(observations))[:, :2]
        return mean.tanh()
