"""Independent actor/critic, mask-aware immutable normalization and bounded Gaussian policy."""
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

from train_general_big_map_test10_SAC.config import OBSERVATION_SIZE, CRITIC_SIZE, LAYERS, STATE_LAYERS, MAP_LAYERS, LOCAL_CELLS, CELL_SIZE, GOAL_SCALE_M


class FixedAffine(nn.Module):
    def __init__(self, mean, scale):
        super().__init__()
        self.register_buffer('mean', torch.tensor(mean, dtype=torch.float32))
        self.register_buffer('scale', torch.tensor(scale, dtype=torch.float32))

    def forward(self, values):
        return ((values-self.mean)/self.scale).clamp(-10, 10)


class ObservationNormalizer(nn.Module):
    def __init__(self, scalers=None):
        super().__init__()
        from train_general_big_map_test10_SAC.frozen_scalers import load_scalers
        data = load_scalers(data=scalers)
        self.metadata = data
        self.state = FixedAffine([0.]*7, data['state_scale'])
        self.visual = FixedAffine(data['visual_mean'], data['visual_scale'])
        self.geometry = FixedAffine([0.]*4, data['geometry_scale'])
        self.privileged = FixedAffine([0.]*20, data['privileged_scale'])

    @torch.no_grad()
    def update(self, actor, critic):
        return None  # no statistics are ever updated during RL or evaluation

    @torch.no_grad()
    def clipping(self, actor, critic):
        def fraction(values, scaler, mask=None):
            clipped=((values-scaler.mean)/scaler.scale).abs()>10
            if mask is None:return clipped.float().mean()
            mask=mask.expand_as(clipped)
            return (clipped&mask).sum().float()/mask.sum().clamp_min(1)
        cells=actor[:,15:].reshape(-1,LOCAL_CELLS,CELL_SIZE)
        p=critic[:,OBSERVATION_SIZE:];contact=p[:,8:12]>.5
        pmask=torch.cat((contact.repeat_interleave(2,1),contact,contact.repeat_interleave(2,1)),1)
        return dict(clip_state=fraction(torch.cat((actor[:,2:8],actor[:,13:14]),1),self.state),
                    clip_visual=fraction(cells[...,:32],self.visual,cells[...,36:37]>.5),
                    clip_geometry=fraction(cells[...,32:36],self.geometry,cells[...,37:38]>.5),
                    clip_privileged=fraction(torch.cat((p[:,:8],p[:,12:]),1),self.privileged,pmask))

    def actor(self, raw):
        state = raw[:, :15].clone()
        normalized = self.state(torch.cat((raw[:, 2:8], raw[:, 13:14]), dim=1))
        state[:, :2] = raw[:, :2] / GOAL_SCALE_M
        state[:, 2:8] = normalized[:, :6]
        state[:, 13:14] = normalized[:, 6:7]
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
    def __init__(self, scalers=None):
        super().__init__()
        self.normalizer = ObservationNormalizer(scalers)
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
