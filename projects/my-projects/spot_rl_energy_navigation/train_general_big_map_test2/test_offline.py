"""CPU-only numerical/API-contract checks. Never imports/launches Isaac Sim or DINO."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_general_big_map_test2.config import PPO_CFG, NAV_CFG, experiment_signature, LOCAL_CELLS, CELL_SIZE, OBSERVATION_SIZE, CRITIC_SIZE
from train_general_big_map_test2.perception import FrozenPCA, local_cells, world_to_local, backproject, cell_indices
from train_general_big_map_test2.networks import ActorCritic, ObservationNormalizer, ExportedActor, squashed_log_prob
from train_general_big_map_test2.ppo import PPO, gae, rollout_length
from train_general_big_map_test2.privileged import combine_friction, PrivilegedContacts
from train_general_big_map_test2.artifacts import save_checkpoint, restore_rng
from train_general_big_map_test2.layouts import training_conditions

torch.set_num_threads(2)


def pca_fixture():
    return FrozenPCA(np.arange(384, dtype=np.float32)/100, np.eye(384, dtype=np.float32)[:32],
                     np.ones(32)/384, {'test': True})


def synthetic_observations(count=64):
    raw = torch.randn(count, OBSERVATION_SIZE)
    raw[:, 8:11] = torch.tensor([0., 0., -1.])
    raw[:, 11:13] = torch.tanh(raw[:, 11:13])
    raw[:, 14] = torch.rand(count)
    cells = raw[:, 15:].view(count, LOCAL_CELLS, CELL_SIZE)
    cells[..., 38:] = torch.rand(count, LOCAL_CELLS, 2)
    cells[..., 36:38] = torch.randint(0, 2, (count, LOCAL_CELLS, 2))
    priv = torch.randn(count, 24)
    priv[:, 8:12] = torch.randint(0, 2, (count, 4))
    return raw, torch.cat((raw, priv), dim=1)


def test_replicated_map_and_exact_budget():
    conditions = training_conditions()
    assert len(conditions) == 32
    assert len({c.layout.usd for c in conditions}) == 1
    assert all(c.layout.usd.is_file() for c in conditions)
    remaining, lengths = 100000, []
    while remaining:
        steps = rollout_length(remaining, 32, 128)
        lengths.append(steps)
        remaining -= steps*32
    assert lengths == [128]*24+[53]
    assert sum(lengths)*32 == 100000
    assert NAV_CFG.max_episode_steps == 450


def test_pca_before_mean_and_roundtrip(tmp_path):
    pca = pca_fixture()
    features = np.random.default_rng(1).normal(size=(50, 384)).astype(np.float32)
    np.testing.assert_allclose(pca.transform(features).mean(0), pca.transform(features.mean(0)), atol=1e-6)
    path = tmp_path/'pca.npz'
    pca.save(path)
    loaded = FrozenPCA.load(path)
    np.testing.assert_array_equal(loaded.transform(features), pca.transform(features))


def test_geometric_plane_rotation_and_missing_cells():
    x, y = np.meshgrid(np.linspace(0.3, 0.7, 12), np.linspace(-0.2, 0.2, 12))
    points = np.column_stack((x.ravel(), y.ravel(), 0.2*x.ravel()-0.1*y.ravel()-0.5))
    cells = local_cells(np.array([[0.5, 0, -0.4]]), np.ones((1, 32)), points)
    np.testing.assert_allclose(cells[49, 32:36], [-0.4, np.arctan(.2), np.arctan(-.1), 0], atol=1e-6)
    assert cells[49, 36:38].tolist() == [1, 1]
    assert not cells[0].any()
    yaw, base = 1.2, np.array([3., 7., 0.8])
    c, s = np.cos(yaw), np.sin(yaw)
    world = points @ np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]]) + base
    np.testing.assert_allclose(world_to_local(world, base, yaw), points, atol=1e-12)
    # First near-side slots exist: no artificial 50-degree cutoff.
    np.testing.assert_array_equal(cell_indices(np.array([[.5, -1, 0], [.5, 1, 0], [2, 1, 0]])), [47, 51, 78])
    degenerate = np.column_stack((np.linspace(.3, .7, 20), np.zeros(20), np.zeros(20)))
    assert local_cells(np.empty((0, 3)), np.empty((0, 32)), degenerate)[:, 37].sum() == 0


def test_backprojection_optical_z_and_invalid_depth():
    depth = np.full((4, 4), 2.)
    depth[1, 1] = np.nan
    k = np.array([[2., 0, 1], [0, 2., 1], [0, 0, 1]])
    points, ids = backproject(depth, k, np.zeros(3), np.array([1., 0, 0, 0]), stride=2)
    assert len(points) == 3
    np.testing.assert_array_equal(ids, [1, 2, 3])
    np.testing.assert_allclose(points, [[2, 0, 2], [0, 2, 2], [2, 2, 2]])


def test_masked_normalization_and_actor_privilege_isolation():
    actor, critic = synthetic_observations()
    norm = ObservationNormalizer()
    norm.update(actor, critic)
    normalized = norm.actor(actor)
    cells = actor[:, 15:].reshape(-1, CELL_SIZE)
    result = normalized[:, 15:].reshape(-1, CELL_SIZE)
    assert torch.all(result[cells[:, 36] == 0, :32] == 0)
    assert torch.all(result[cells[:, 37] == 0, 32:36] == 0)
    assert torch.equal(actor[:, 8:13], normalized[:, 8:13])
    assert torch.equal(actor[:, 14], normalized[:, 14])
    previous_count = norm.visual.count.clone()
    missing = actor.clone()
    missing[:, 15:].view(-1, LOCAL_CELLS, CELL_SIZE)[..., 36:] = 0
    norm.update(missing, critic)
    assert torch.equal(previous_count, norm.visual.count)
    policy = ActorCritic()
    first = policy.act(actor, critic, deterministic=True)[0]
    changed = critic.clone()
    changed[:, OBSERVATION_SIZE:] += 100
    second = policy.act(actor, changed, deterministic=True)[0]
    assert torch.equal(first, second)


def test_two_head_gae_terminal_and_rollout_bootstrap():
    rewards = torch.tensor([[[1., 2.]], [[3., 4.]], [[5., 6.]]])
    values = torch.zeros_like(rewards)
    done = torch.tensor([[False], [True], [False]])
    advantage, returns = gae(rewards, values, done, torch.tensor([[7., 8.]]), .9, 1.)
    expected = torch.tensor([[[3.7, 5.6]], [[3., 4.]], [[11.3, 13.2]]])
    torch.testing.assert_close(returns, expected)
    torch.testing.assert_close(advantage, expected)


def test_squashed_gaussian_stable_and_correct():
    from torch.distributions import Normal, TransformedDistribution, TanhTransform
    distribution = Normal(torch.zeros(2), torch.ones(2))
    u = torch.tensor([[.3, -.7], [20., -20.]])
    result = squashed_log_prob(distribution, u)
    assert torch.isfinite(result).all()
    reference = TransformedDistribution(distribution, [TanhTransform()]).log_prob(u[0].tanh()).sum()
    torch.testing.assert_close(result[0], reference)


def test_friction_combination_precedence():
    np.testing.assert_allclose(combine_friction((1, .8, 'average'), (.4, .3, 'average')), [.7, .55])
    np.testing.assert_allclose(combine_friction((1, .8, 'max'), (.4, .3, 'multiply')), [1, .8])
    np.testing.assert_allclose(combine_friction((1, .8, 'average'), (.4, .3, 'min')), [.4, .3])


def test_ppo_numerics_checkpoint_and_actor_export(tmp_path):
    learner = PPO(settings=replace(PPO_CFG, epochs=1, batch_size=32))
    actor, critic = synthetic_observations(64)
    learner.policy.normalizer.update(actor, critic)
    with torch.no_grad():
        action, latent, log_prob, values = learner.policy.act(actor, critic)
    batch = dict(actor=actor.reshape(2, 32, OBSERVATION_SIZE), critic=critic.reshape(2, 32, CRITIC_SIZE),
                 latent=latent.reshape(2, 32, 2), log_prob=log_prob.reshape(2, 32),
                 values=values.reshape(2, 32, 2), rewards=torch.randn(2, 32, 2),
                 terminals=torch.zeros(2, 32, dtype=torch.bool))
    metrics = learner.update(batch, torch.zeros(32, 2))
    assert metrics['minibatch_updates'] > 0
    assert all(np.isfinite(v) for v in metrics.values())
    path = tmp_path/'checkpoint.pt'
    learner.transitions = 64
    save_checkpoint(path, learner, pca_fixture(), {'test': True},
                    training_state={'curriculum': {'level': 1, 'streak': 1}})
    checkpoint = torch.load(path, weights_only=False)
    assert checkpoint['training_state']['curriculum'] == {'level': 1, 'streak': 1}
    restored = PPO(settings=learner.cfg)
    restored.load_state_dict(checkpoint['learner'])
    assert restored.transitions == 64
    torch.testing.assert_close(learner.policy.act(actor, critic, True)[0], restored.policy.act(actor, critic, True)[0])
    exported = torch.jit.trace(ExportedActor(learner.policy), torch.zeros(1, OBSERVATION_SIZE))
    torch.testing.assert_close(exported(actor), learner.policy.act(actor, critic, True)[0])


def environment_method(name, **extra):
    """Exercise the actual environment equations without importing the simulator modules."""
    import ast
    import math
    source = ast.parse((Path(__file__).parent/'env.py').read_text())
    cls = next(node for node in source.body if isinstance(node, ast.ClassDef))
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = dict(torch=torch, math=math, **extra)
    exec(compile(ast.Module(body=[method], type_ignores=[]), 'env.py', 'exec'), namespace)
    return namespace[name]


def test_all_40_physics_intervals_count_once():
    from types import SimpleNamespace as NS
    step = environment_method('step')
    env = NS(device='cpu', cfg=NS(decimation=40), physics_dt=.005, _sim_step_counter=0,
             episode_length_buf=torch.zeros(2, dtype=torch.long), common_step_counter=0,
             reset_terminated=torch.zeros(2, dtype=torch.bool), reset_time_outs=torch.zeros(2, dtype=torch.bool),
             step_energy=torch.zeros(2), extras={})
    env._pre_physics_step = lambda _: env.step_energy.zero_()
    env._measured_power = lambda: torch.full((2,), float(env._sim_step_counter))
    env._apply_action = lambda: None
    env.scene = NS(write_data_to_sim=lambda: None, update=lambda _: None)
    env.sim = NS(step=lambda **_: None)
    env._get_dones = lambda: (torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool))
    env._get_rewards = lambda: env.step_energy.clone()
    env._episode_records = lambda: []
    env._get_observations = lambda: {}
    _, reward, _, _, _ = step(env, torch.zeros(2, 2))
    # Exact integral of P(t)=200*t on [0, .2], with P samples 0,...,40.
    torch.testing.assert_close(reward, torch.full((2,), 4.0))
    assert env._sim_step_counter == 40
    assert env.episode_length_buf.tolist() == [1, 1]


def test_deadline_is_terminal_and_goal_wins_at_last_step():
    from types import SimpleNamespace as NS
    done = environment_method('_get_dones', terrain_height=lambda _: torch.tensor([2., 0.]))
    env = NS(nav=NAV_CFG, device='cpu', geometry=NS(world_x_bounds=(-.8, 4.8), world_y_bounds=(-.8, 4.8)),
             scene=NS(env_origins=torch.zeros(2, 3)),
             robot=NS(data=NS(root_pos_w=torch.tensor([[0., 0., 2.6], [0., 0., .6]]),
                              projected_gravity_b=torch.tensor([[0., 0., -1.], [0., 0., -1.]]))),
             episode_length_buf=torch.tensor([200, 200]), deadline_steps=torch.tensor([200, 200]), _distance=lambda: torch.tensor([.4, 2.]))
    terminated, truncated = done(env)
    assert terminated.tolist() == [True, True]
    assert not truncated.any()
    assert env.success.tolist() == [True, False]
    assert env.timeout.tolist() == [False, True]
    assert not env.fallen.any()  # equal clearance despite different world elevation


def test_contact_groups_material_weighting_and_airborne_mask():
    from types import SimpleNamespace as NS
    contact = PrivilegedContacts.__new__(PrivilegedContacts)
    contact.coefficients = torch.zeros(1, 4, 49, 2)
    contact.coefficients[:, 0, 0] = torch.tensor([1., .8])
    contact.coefficients[:, 0, 1] = torch.tensor([.4, .2])
    sensors = []
    for i in range(4):
        forces = torch.zeros(1, 1, 49, 3)
        points = torch.full_like(forces, torch.nan)
        if i == 0:
            forces[0, 0, :2, 2] = torch.tensor([100., 50.])
            points[0, 0, :2] = 0
        sensors.append(NS(data=NS(force_matrix_w=forces, contact_pos_w=points)))
    env = NS(num_envs=1, foot_sensors=sensors, foot_ids=[0, 1, 2, 3],
             robot=NS(data=NS(body_link_pos_w=torch.zeros(1, 4, 3),
                              body_link_lin_vel_w=torch.tensor([[[.2, .1, 0.]]]*4).reshape(1, 4, 3),
                              body_ang_vel_w=torch.zeros(1, 4, 3))))
    result = contact.extract(env, torch.zeros(1))
    assert result.shape == (1, 24)
    torch.testing.assert_close(result[0, :2], torch.tensor([.8, .6]))
    assert torch.equal(result[0, 8:12], torch.tensor([1., 0., 0., 0.]))
    torch.testing.assert_close(result[0, 12:16], torch.tensor([150., 0., 0., 0.]))
    torch.testing.assert_close(result[0, 16:18], torch.tensor([.2, .1]))
    assert not result[0, 18:].any()


def test_general_map_reward_components_and_energy_weight():
    from types import SimpleNamespace as NS
    reward = environment_method('_get_rewards')
    env = NS(nav=NAV_CFG, step_dt=.2,
             _distance=lambda: torch.tensor([.4, 2.1]),
             previous_distance=torch.tensor([.6, 2.0]), previous_xy=torch.zeros(2, 2),
             robot=NS(data=NS(root_pos_w=torch.zeros(2, 3))),
             scene=NS(env_origins=torch.zeros(2, 3)),
             path_length=torch.zeros(2), right_path_m=torch.zeros(2), left_path_m=torch.zeros(2),
             episode_energy=torch.zeros(2), step_energy=torch.tensor([20., 50.]),
             power_ema=torch.zeros(2), fallen=torch.tensor([False, True]),
             out_of_bounds=torch.zeros(2, dtype=torch.bool), timeout=torch.zeros(2, dtype=torch.bool),
             success=torch.tensor([True, False]), episode_nav_return=torch.zeros(2),
             episode_cost=torch.zeros(2), extras={})
    result = reward(env)
    torch.testing.assert_close(env.extras['reward_components'],
                               torch.tensor([[151.98, .2], [-301.02, .5]]))
    torch.testing.assert_close(result, torch.tensor([151.78, -301.52]))
    assert PPO_CFG.timesteps == 100000
