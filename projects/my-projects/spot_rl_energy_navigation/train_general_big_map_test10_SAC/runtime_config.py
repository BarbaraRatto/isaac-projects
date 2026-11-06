"""Per-run SAC overrides without mutating the frozen experiment defaults."""
from dataclasses import asdict, replace
from train_general_big_map_test10_SAC.config import experiment_signature


def training_settings(defaults, num_envs, gradient_steps):
    if num_envs < 1 or gradient_steps < 1:
        raise ValueError('num-envs and gradient-steps must be positive')
    return replace(defaults, num_envs=num_envs, gradient_steps=gradient_steps)


def runtime_signature(settings):
    signature = experiment_signature()
    signature['sac'] = asdict(settings)
    return signature


def playback_signature(defaults, checkpoint_signature):
    # Playback may use a different robot count, but all non-CLI algorithm settings
    # must still match. Do not blindly accept the entire checkpoint signature.
    recorded = checkpoint_signature['sac']
    settings = training_settings(defaults, recorded['num_envs'], recorded['gradient_steps'])
    return runtime_signature(settings)
