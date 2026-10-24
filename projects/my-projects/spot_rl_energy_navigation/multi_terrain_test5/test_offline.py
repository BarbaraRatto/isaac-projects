"""CPU checks without launching Isaac Sim or loading/downloading DINO weights.

Run from my-projects:
PYTHONDONTWRITEBYTECODE=1 python3 spot_rl_energy_navigation/multi_terrain_test5/test_offline.py
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
from stable_baselines3.common.policies import ActorCriticPolicy

from binary_choice.config import NAV_CFG_BINARY
from multi_terrain_test5.actions import navigation_commands
from multi_terrain_test5.config import (
    ACTION_LOW, ACTION_HIGH, FEATURE_DIM, NAV_CFG, OBSERVATION_SIZE, PPO_CFG,
    LOCAL_CELLS, ROWS, COLS, VISUAL_SIZE, TERRAIN_LABELS,
)
from multi_terrain_test5.layouts import training_conditions
from multi_terrain_test5.visual_features import FeatureField, VisualObservation


class TestOffline(unittest.TestCase):
    def test_commands_stop_turn_advance_and_forbid_reverse_or_strafe(self):
        actions = torch.tensor([[0., 0.], [0., 1.], [1., -1.], [-3., 4.]])
        commands = navigation_commands(actions)
        torch.testing.assert_close(commands, torch.tensor([
            [0., 0., 0.], [0., 0., 1.5], [1.6, 0., -1.5], [0., 0., 1.5],
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

    def test_layouts_and_rewards_match_test4_without_mutation(self):
        from multi_terrain_binary_test4.layouts import training_conditions as test4_conditions
        actual, reference = training_conditions(), test4_conditions()
        self.assertEqual(len(actual), 32)
        for a, b in zip(actual, reference):
            self.assertEqual((a.layout.usd, a.layout.grid, a.start_xy, a.goal_xy),
                             (b.layout.usd, b.layout.grid, b.start_xy, b.goal_xy))
            self.assertTrue(a.layout.usd.is_file())
        for key in ("progress_reward_per_m", "energy_penalty_per_j", "time_penalty_per_step",
                    "success_bonus", "failure_penalty", "physics_dt",
                    "physics_steps_per_action", "max_episode_seconds"):
            self.assertEqual(getattr(NAV_CFG, key), getattr(NAV_CFG_BINARY, key))
        self.assertEqual(NAV_CFG.max_lateral_m_s, 0.)
        self.assertEqual(NAV_CFG_BINARY.max_lateral_m_s, 0.9)

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
        scope = dict(time=time, Path=Path, torch=torch, np=np, cv2=cv2,
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

    def test_policy_shapes_action_bounds_and_save_load(self):
        torch.set_num_threads(2)
        obs_space = gym.spaces.Box(-np.inf, np.inf, (OBSERVATION_SIZE,), np.float32)
        action_space = gym.spaces.Box(np.array(ACTION_LOW, np.float32),
                                      np.array(ACTION_HIGH, np.float32))
        policy = ActorCriticPolicy(obs_space, action_space, lambda _: PPO_CFG.learning_rate,
                                   net_arch=list(PPO_CFG.net_arch), activation_fn=torch.nn.Tanh)
        layers = [(m.in_features, m.out_features) for m in policy.mlp_extractor.policy_net
                  if isinstance(m, torch.nn.Linear)]
        self.assertEqual(layers, [(7711, 512), (512, 256), (256, 128)])
        observation = np.zeros((3, OBSERVATION_SIZE), np.float32)
        action, _ = policy.predict(observation, deterministic=True)
        self.assertEqual(action.shape, (3, 2))
        for row in action:
            self.assertTrue(action_space.contains(row))
        self.assertEqual(policy.predict_values(torch.from_numpy(observation)).shape, (3, 1))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pth"
            policy.save(path)
            restored = ActorCriticPolicy.load(path, device="cpu")
            np.testing.assert_array_equal(restored.predict(observation, deterministic=True)[0], action)
        self.assertEqual(LOCAL_CELLS, 20)
        self.assertEqual(VISUAL_SIZE, 7700)

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
            last_visual_ms=0., last_dino_ms=0., terrain_entries=torch.zeros(2, 7, dtype=torch.bool),
            vision=SimpleNamespace(last_visual_ms=0., last_dino_ms=0.,
                extract=lambda *args: (torch.zeros(2, VISUAL_SIZE), torch.zeros(2))),
            _measured_power=lambda: torch.full((2,), 100.),
        )
        env._distance = lambda: scope["_distance"](env)
        env.previous_distance = env._distance() + 0.1
        scope["_pre_physics_step"](env, torch.tensor([[0.5, -0.5]] * 2))
        reward = scope["_get_rewards"](env)
        torch.testing.assert_close(reward, torch.full((2,), 0.28))
        observation = scope["_get_observations"](env)["policy"]
        self.assertEqual(observation.shape, (2, 7711))
        self.assertTrue(bool(torch.isfinite(observation).all()))
        torch.testing.assert_close(observation[:, 3], torch.full((2,), 0.2))
        torch.testing.assert_close(observation[:, 8:11], torch.tensor([[0.5, 0., -0.5]] * 2))
        torch.testing.assert_close(env.extras["command_vy_m_s"], torch.zeros(2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
