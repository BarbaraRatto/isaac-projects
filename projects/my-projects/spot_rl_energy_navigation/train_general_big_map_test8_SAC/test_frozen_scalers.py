"""All buffers and outputs must remain stable despite changing experience."""
import torch
from train_general_big_map_test8_SAC.networks import ObservationNormalizer, SACPolicy, ExportedActor
from train_general_big_map_test8_SAC.test_offline import synthetic_observations
from train_general_big_map_test8_SAC.config import OBSERVATION_SIZE


def test_no_scaler_updates_and_fixed_physical_units():
    actor, critic = synthetic_observations()
    norm=ObservationNormalizer()
    saved={k:v.clone() for k,v in norm.state_dict().items()}
    expected_actor=norm.actor(actor).clone()
    expected_critic=norm.critic(critic).clone()
    for factor in (1.,100.,-.5):
        norm.update(actor*factor,critic*factor)
    for k,v in norm.state_dict().items(): assert torch.equal(v,saved[k])
    torch.testing.assert_close(norm.actor(actor),expected_actor,rtol=0,atol=0)
    torch.testing.assert_close(norm.critic(critic),expected_critic,rtol=0,atol=0)
    torch.testing.assert_close(expected_actor[:,:2],actor[:,:2]/10,rtol=0,atol=0)
    torch.testing.assert_close(expected_actor[:,2:5],(actor[:,2:5]/2).clamp(-10,10))
    torch.testing.assert_close(expected_actor[:,5:8],(actor[:,5:8]/2).clamp(-10,10))
    torch.testing.assert_close(expected_actor[:,13],(actor[:,13]/400).clamp(-10,10))
    torch.testing.assert_close(expected_critic[:,:OBSERVATION_SIZE],expected_actor)


def test_actor_export_and_reload_use_same_frozen_scalers():
    actor,critic=synthetic_observations()
    policy=SACPolicy().eval(); restored=SACPolicy().eval()
    restored.load_state_dict(policy.state_dict())
    with torch.no_grad():
        expected=policy.act(actor,critic,deterministic=True)[0]
        torch.testing.assert_close(restored.act(actor,critic,deterministic=True)[0],expected)
        exported=torch.jit.trace(ExportedActor(restored),actor)
        torch.testing.assert_close(exported(actor),expected)
