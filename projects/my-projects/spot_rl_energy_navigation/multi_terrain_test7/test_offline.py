"""CPU checks without launching Isaac Sim or loading/downloading DINO weights.

Run from my-projects:
PYTHONDONTWRITEBYTECODE=1 python3 spot_rl_energy_navigation/multi_terrain_test7/test_offline.py
"""

import ast
import math
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import cv2
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from multi_terrain_test7.actions import navigation_commands
from multi_terrain_test7.config import (
    ACTION_LOW, ACTION_HIGH, FEATURE_DIM, NAV_CFG, OBSERVATION_SIZE, SAC_CFG,
    LOCAL_CELLS, ROWS, COLS, VISUAL_SIZE, TERRAIN_LABELS,
)
from multi_terrain_test7.layouts import training_conditions
from multi_terrain_test7.visual_features import FeatureField, VisualObservation
from multi_terrain_test7.networks import policy_kwargs
from multi_terrain_test7.evaluation import PeriodicEvaluation, summarize
from multi_terrain_test7.utils import replay_bytes


class SyntheticNavigation(gym.Env):
    """Tiny numeric interface for testing SAC updates; no simulation or DINO."""
    metadata = {"render_modes": []}

    def __init__(self):
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBSERVATION_SIZE,), np.float32)
        self.action_space = gym.spaces.Box(-1., 1., (2,), np.float32)
        self.steps = 0

    def observation(self):
        obs = np.full(OBSERVATION_SIZE, 0.01, np.float32)
        obs[:11] = np.arange(11, dtype=np.float32) / 10 + 0.01
        obs[11:].reshape(20, 385)[:, -1] = 1.
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        return self.observation(), {}

    def step(self, action):
        self.steps += 1
        done = self.steps == 2
        info = {
            "is_success": done, "episode_energy_j": float(self.steps),
            "path_length_m": float(self.steps), "final_distance_m": 0.4 if done else 3.,
        }
        return self.observation(), float(0.1 - abs(action[1])), done, False, info


class TestSAC(unittest.TestCase):
    def test_two_branches_gradients_replay_eval_and_checkpoint_roundtrip(self):
        torch.set_num_threads(2)
        env = DummyVecEnv([SyntheticNavigation for _ in range(32)])
        model = SAC(
            "MlpPolicy", env, buffer_size=64, batch_size=8, learning_starts=0,
            train_freq=(1, "step"), gradient_steps=1, ent_coef="auto",
            policy_kwargs=policy_kwargs(), device="cpu", seed=0,
        )
        obs = env.reset()
        extractor = model.actor.features_extractor
        with torch.no_grad():
            self.assertEqual(extractor(torch.from_numpy(obs[:2])).shape, (2, 1152))
        self.assertIsNot(extractor, model.critic.features_extractor)
        self.assertIsNot(model.critic.features_extractor, model.critic_target.features_extractor)
        self.assertEqual(len(model.critic.q_networks), 2)
        visual_before = extractor.visual_branch[0].weight[:2].detach().clone()
        state_before = extractor.state_branch[0].weight.detach().clone()
        # Numeric smoke test, not a robot training run: two vector steps/updates.
        with tempfile.TemporaryDirectory() as directory:
            periodic = PeriodicEvaluation(Path(directory), training_conditions(), 32, seed=10000)
            model.learn(total_timesteps=64, callback=periodic)
            records = (Path(directory) / "evaluations.csv").read_text().splitlines()
            self.assertEqual(len(records), 1 + 2 * 32)
        self.assertEqual(model._n_updates, 2)
        self.assertTrue(model.replay_buffer.full)
        self.assertFalse(torch.equal(visual_before, extractor.visual_branch[0].weight[:2]))
        self.assertFalse(torch.equal(state_before, extractor.state_branch[0].weight))
        self.assertTrue(torch.isfinite(model.log_ent_coef).all())
        action = model.predict(obs, deterministic=True)[0]
        self.assertEqual(action.shape, (32, 2))
        self.assertTrue(np.isfinite(action).all())
        self.assertTrue((np.abs(action) <= 1.).all())
        q_values = model.critic(torch.from_numpy(obs[:2]), torch.from_numpy(action[:2]))
        self.assertEqual([q.shape for q in q_values], [(2, 1), (2, 1)])
        replay_position = model.replay_buffer.pos
        replay_before = model.replay_buffer.observations.copy()
        with tempfile.TemporaryDirectory() as directory:
            callback = PeriodicEvaluation(Path(directory), training_conditions(), 64, seed=10000)
            callback.init_callback(model)
            callback.num_timesteps = model.num_timesteps
            callback.evaluate_now(phase="final")
            self.assertEqual(model.num_timesteps, 64)
            self.assertEqual(model.replay_buffer.pos, replay_position)
            np.testing.assert_array_equal(model.replay_buffer.observations, replay_before)
            np.testing.assert_array_equal(model._last_obs, env.reset())
            path = Path(directory) / "policy_best.zip"
            self.assertTrue(path.is_file())
            restored = SAC.load(path, device="cpu")
            np.testing.assert_array_equal(restored.predict(obs, deterministic=True)[0], action)
        self.assertEqual(replay_bytes(100000, 32), 6_170_800_000)
        env.close()

    def test_best_selection_requires_both_maps_before_energy(self):
        def rows(first, second, energy):
            return [{"layout": name, "success": int(i < n), "energy_j": energy,
                     "final_distance_m": 3.}
                    for name, n in (("a", first), ("b", second)) for i in range(4)]
        _, one_sided = summarize(rows(4, 0, 1.))
        _, balanced = summarize(rows(2, 2, 1000.))
        _, balanced_efficient = summarize(rows(2, 2, 500.))
        self.assertGreater(balanced, one_sided)
        self.assertGreater(balanced_efficient, balanced)


class TestOffline(unittest.TestCase):
    def test_commands_stop_turn_advance_and_forbid_reverse_or_strafe(self):
        actions = torch.tensor([[-1., 0.], [-1., 1.], [1., -1.], [-3., 4.], [0., 0.]])
        commands = navigation_commands(actions)
        torch.testing.assert_close(commands, torch.tensor([
            [0., 0., 0.], [0., 0., 1.5], [1.6, 0., -1.5], [0., 0., 1.5], [0.8, 0., 0.],
        ]))
        self.assertTrue(torch.equal(actions[3], torch.tensor([-3., 4.])))
        with self.assertRaises(ValueError):
            navigation_commands(torch.zeros(2, 3))
        with self.assertRaises(ValueError):
            navigation_commands(torch.tensor([[float("nan"), 0.]]))

    def test_raw_features_are_averaged_without_projection_or_clipping(self):
        field = FeatureField()
        a = np.arange(FEATURE_DIM, dtype=np.float32) - 100.
        b = a + 4.
        points = np.array([[0.1, 0.1, 0.], [0.2, 0.2, 0.], [9., 9., 0.]])
        field.insert(points, np.stack([a, b, a * 10]))
        raw, valid = field.finish()
        self.assertEqual(raw.shape, (12, 12, 384))
        self.assertEqual(valid.sum(), 1)
        np.testing.assert_array_equal(raw[2, 2], (a + b) / 2)
        # A newly created grid has no memory of a preceding observation.
        self.assertFalse(FeatureField().finish()[1].any())

    def test_front_samples_rotate_with_robot_and_keep_validity(self):
        raw = np.empty((ROWS, COLS, FEATURE_DIM), dtype=np.float32)
        for row in range(ROWS):
            for col in range(COLS):
                raw[row, col] = row * 100 + col
        valid = np.ones((ROWS, COLS), dtype=bool)
        sampler = VisualObservation()
        forward = sampler.local(np.zeros(2), 0., raw, valid).reshape(20, 385)
        turned = sampler.local(np.zeros(2), math.pi / 2, raw, valid).reshape(20, 385)
        self.assertEqual(forward[:, -1].sum(), 18)
        self.assertEqual(turned[:, -1].sum(), 18)
        np.testing.assert_array_equal(forward[[0, 4]], 0.)
        # Sample 2 is directly 0.5 m ahead: +X at yaw 0, +Y at yaw pi/2.
        np.testing.assert_array_equal(forward[2, :384], raw[3, 2])
        np.testing.assert_array_equal(turned[2, :384], raw[2, 3])
        valid[3, 2] = False
        missing = sampler.local(np.zeros(2), 0., raw, valid).reshape(20, 385)
        np.testing.assert_array_equal(missing[2], 0.)

    def test_local_layouts_reward_and_import_isolation(self):
        actual = training_conditions()
        self.assertEqual(len(actual), 32)
        self.assertEqual([c.layout.costly_side for c in actual], ["right"] * 16 + ["left"] * 16)
        for condition in actual:
            self.assertEqual(condition.start_xy, (0., 0.))
            self.assertEqual(condition.goal_xy, (4., 4.))
            self.assertTrue(condition.layout.usd.is_file())
        self.assertEqual((NAV_CFG.progress_reward_per_m, NAV_CFG.energy_penalty_per_j,
                          NAV_CFG.time_penalty_per_step, NAV_CFG.success_bonus,
                          NAV_CFG.failure_penalty), (3., 1.1, 0.01, 60., 25.))
        self.assertEqual(NAV_CFG.max_lateral_m_s, 0.)
        self.assertEqual(SAC_CFG.timesteps, 100_000)
        self.assertEqual(SAC_CFG.timesteps % SAC_CFG.num_envs, 0)
        forbidden = ("binary_choice", "full_grid_energy_choice", "multi_terrain_binary_",
                     "multi_terrain_test5", "multi_terrain_test6", "rl_config", "zed_sensor", "batched_spot", "spot_terrain_gridmap")
        for path in HERE.glob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    self.assertFalse(node.module.startswith(forbidden), str(path))
                elif isinstance(node, ast.Import):
                    for name in node.names:
                        self.assertFalse(name.name.startswith(forbidden), str(path))

    def test_raw_vision_microbatches_and_current_frame_only(self):
        # Execute the actual class with fake camera/DINO interfaces, avoiding Isaac imports.
        module = ast.parse((HERE / "vision.py").read_text())
        node = next(n for n in module.body if isinstance(n, ast.ClassDef)
                    and n.name == "RawBatchedDino")
        batch_calls = []
        def processor(images, **kwargs):
            batch_calls.append(len(images))
            return {"pixel_values": torch.zeros(len(images), 3, 2, 2)}
        def model(pixel_values):
            features = torch.arange(384, dtype=torch.float32).expand(len(pixel_values), 1370, 384)
            return SimpleNamespace(last_hidden_state=features)
        def extractor(**kwargs):
            return SimpleNamespace(processor=processor, model=model, device="cpu",
                                   n_patches=37, feature_dim=384)
        visible = [True]
        def ground_points(camera, env_id):
            if visible[0]:
                # +X at 0.5 m: the central nearest sample.
                return np.array([[0.5, 0., 0.]]), np.arange(1369) == 0
            return np.empty((0, 3)), np.zeros(1369, dtype=bool)
        scope = dict(time=time, torch=torch, np=np, cv2=cv2,
                     DinoFeatureExtractor=extractor, FEATURE_DIM=FEATURE_DIM,
                     VisualObservation=VisualObservation, FeatureField=FeatureField,
                     ground_points=ground_points)
        exec(compile(ast.Module(body=[node], type_ignores=[]), "vision.py", "exec"), scope)
        camera = SimpleNamespace(data=SimpleNamespace(output={
            "rgb": torch.zeros(5, 4, 4, 4, dtype=torch.uint8),
        }))
        vision = scope["RawBatchedDino"]("cpu", batch_size=4)
        output, valid = vision.extract(camera, torch.zeros(5, 3), torch.zeros(5, 2), torch.zeros(5))
        self.assertEqual(output.shape, (5, 7700))
        self.assertEqual(batch_calls, [4, 1])
        self.assertEqual(valid.tolist(), [1] * 5)
        np.testing.assert_array_equal(output[0].reshape(20, 385)[2, :384], np.arange(384))
        visible[0] = False
        empty, valid = vision.extract(camera, torch.zeros(5, 3), torch.zeros(5, 2), torch.zeros(5))
        self.assertFalse(bool(empty.any()))
        self.assertFalse(bool(valid.any()))

    def test_environment_observation_and_reward_with_stationary_sensor_fixtures(self):
        # Exercise the actual changed environment hooks with identity poses,
        # without constructing a simulator or substituting the reward formula.
        module = ast.parse((HERE / "env.py").read_text())
        node = next(n for n in module.body if isinstance(n, ast.ClassDef))
        methods = [n for n in node.body if isinstance(n, ast.FunctionDef)
                   and n.name in {"_get_observations", "_get_rewards", "_distance", "_pre_physics_step"}]
        scope = dict(torch=torch, OBSERVATION_SIZE=OBSERVATION_SIZE,
                     TERRAIN_LABELS=TERRAIN_LABELS, navigation_commands=navigation_commands,
                     yaw_quat=lambda q: q, quat_apply_inverse=lambda q, v: v)
        exec(compile(ast.Module(body=methods, type_ignores=[]), "env.py", "exec"), scope)
        data = SimpleNamespace(
            root_pos_w=torch.zeros(2, 3), root_quat_w=torch.tensor([[1., 0., 0., 0.]] * 2),
            root_lin_vel_b=torch.tensor([[0.4, 0.2, 0.]] * 2),
            root_ang_vel_b=torch.zeros(2, 3), projected_gravity_b=torch.tensor([[0., 0., -1.]] * 2),
        )
        env = SimpleNamespace(
            robot=SimpleNamespace(data=data), goal_w=torch.tensor([[4., 4., 0.]] * 2),
            nav=NAV_CFG, num_envs=2, device="cpu", physics_dt=0.005,
            scene=SimpleNamespace(env_origins=torch.zeros(2, 3)), camera=object(),
            previous_command=torch.zeros(2, 3), step_energy=torch.zeros(2),
            episode_energy=torch.zeros(2), path_length=torch.zeros(2), previous_xy=torch.zeros(2, 2),
            success=torch.zeros(2, dtype=torch.bool), unsafe=torch.zeros(2, dtype=torch.bool),
            fallen=torch.zeros(2, dtype=torch.bool), out_of_bounds=torch.zeros(2, dtype=torch.bool),
            last_visual_ms=0., last_dino_ms=0., terrain_entries=torch.zeros(2, 7, dtype=torch.bool),
            vision=SimpleNamespace(last_visual_ms=0., last_dino_ms=0.,
                extract=lambda *args: (torch.zeros(2, VISUAL_SIZE), torch.zeros(2))),
            _measured_power=lambda: torch.full((2,), 100.),
        )
        env._distance = lambda: scope["_distance"](env)
        env.previous_distance = env._distance() + 0.1
        scope["_pre_physics_step"](env, torch.tensor([[0., -0.5]] * 2))
        reward = scope["_get_rewards"](env)
        torch.testing.assert_close(reward, torch.full((2,), -0.26))
        observation = scope["_get_observations"](env)["policy"]
        self.assertEqual(observation.shape, (2, 7711))
        self.assertTrue(bool(torch.isfinite(observation).all()))
        torch.testing.assert_close(observation[:, 3], torch.full((2,), 0.2))
        torch.testing.assert_close(observation[:, 8:11], torch.tensor([[0.5, 0., -0.5]] * 2))
        torch.testing.assert_close(env.extras["command_vy_m_s"], torch.zeros(2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
