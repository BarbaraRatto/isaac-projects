"""Regression: fixed goal coordinates and unchanged non-goal normalization."""
import torch
from train_general_big_map_test7_SAC.networks import ObservationNormalizer, SACPolicy, ExportedActor
from train_general_big_map_test7_SAC.test_offline import synthetic_observations
from train_general_big_map_test6_SAC.networks import ObservationNormalizer as PreviousNormalizer


def test_goal_scaling_survives_distribution_shift_and_other_inputs_match():
    actor, critic = synthetic_observations()
    actor[:, 0] = torch.linspace(-120., 120., len(actor))
    actor[:, 1] = -actor[:, 0]
    critic[:, :actor.shape[1]] = actor
    norm, previous = ObservationNormalizer(), PreviousNormalizer()
    original = actor.clone()
    for shift in (0., 50., -80.):
        changed = actor.clone()
        changed[:, :2] += shift
        full = critic.clone()
        full[:, :actor.shape[1]] = changed
        norm.update(changed, full)
        previous.update(changed, full)
        torch.testing.assert_close(norm.actor(actor)[:, :2], actor[:, :2]/10, rtol=0, atol=0)
        torch.testing.assert_close(norm.critic(critic)[:, :2], actor[:, :2]/10, rtol=0, atol=0)
        torch.testing.assert_close(norm.actor(actor)[:, 2:], previous.actor(actor)[:, 2:], rtol=0, atol=0)
        torch.testing.assert_close(norm.critic(critic)[:, 2:], previous.critic(critic)[:, 2:], rtol=0, atol=0)
    assert torch.equal(actor, original)
    assert norm.state.mean.numel() == 7  # No goal running-statistic slots.


def test_export_and_checkpoint_preserve_fixed_goal_scaling():
    actor, critic = synthetic_observations()
    policy = SACPolicy().eval()
    policy.normalizer.update(actor, critic)
    restored = SACPolicy().eval()
    restored.load_state_dict(policy.state_dict())
    with torch.no_grad():
        expected = policy.act(actor, critic, deterministic=True)[0]
        torch.testing.assert_close(restored.act(actor, critic, deterministic=True)[0], expected)
        traced = torch.jit.trace(ExportedActor(restored), actor)
        torch.testing.assert_close(traced(actor), expected)
