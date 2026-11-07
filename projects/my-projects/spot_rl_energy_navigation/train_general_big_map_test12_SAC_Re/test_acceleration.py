"""Equivalence and recovery checks without launching the simulator."""
from dataclasses import replace
from pathlib import Path
import math
import sys

import cv2
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_general_big_map_test12_SAC_Re.config import SAC_CFG, OBSERVATION_SIZE, LOCAL_DINO
from train_general_big_map_test12_SAC_Re.fast_rgb import DeviceRGB
from train_general_big_map_test12_SAC_Re.missions import place_missions
from train_general_big_map_test12_SAC_Re.replay import ReplayBuffer
from train_general_big_map_test12_SAC_Re.sac import SAC, materialize_metrics
from train_general_big_map_test12_SAC_Re.test_offline import synthetic_observations


@pytest.fixture(scope='module')
def processor():
    transformers = pytest.importorskip('transformers')
    return transformers.AutoImageProcessor.from_pretrained(str(LOCAL_DINO), local_files_only=True)


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_device_pixels_and_normalization_exact(processor, device):
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('CUDA hardware unavailable')
    rng = np.random.default_rng(42)
    random = rng.integers(0, 256, (2, 360, 640, 3), dtype=np.uint8)
    edge = np.zeros((1, 360, 640, 3), dtype=np.uint8)
    edge[:, ::2, ::2] = 255
    rgb = torch.as_tensor(np.concatenate([random, edge]), device=device)
    implementation = DeviceRGB(processor, device)
    for first in range(len(rgb)):
        assert implementation.verify(rgb[first:first+1])['normalized_max_error'] == 0


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_legacy_npz_tensor_replay_wrap_save_and_cpu_reload(tmp_path, device):
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('CUDA hardware unavailable')
    original = ReplayBuffer(5)
    a, c = synthetic_observations(4)
    for offset in (0, 4):
        a[:, 0] = torch.arange(offset, offset+4)
        c[:, :OBSERVATION_SIZE] = a
        original.add(a, c, torch.zeros(4, 2), a[:, 0], torch.tensor([True, False]*2), a+1, c+1)
    # The exact old np.savez format, not the new writer.
    np.savez(tmp_path/'legacy.npz', capacity=5, size=5, position=3, total=8,
             **original.arrays)
    tensor = ReplayBuffer(5, device=device, tensor_storage=True)
    tensor.load(tmp_path/'legacy.npz')
    for key in original.arrays:
        np.testing.assert_array_equal(original.arrays[key], tensor.arrays[key].cpu().numpy())
    batch = tensor.sample(12, device)
    torch.testing.assert_close(batch['next_actor'], batch['actor']+1)
    tensor.add(a.to(device), c.to(device), torch.zeros(4, 2, device=device),
               a[:, 0].to(device), torch.ones(4, device=device, dtype=torch.bool),
               (a+1).to(device), (c+1).to(device))
    tensor.save(tmp_path/'new.npz')
    loaded = ReplayBuffer(5)
    loaded.load(tmp_path/'new.npz')
    assert loaded.total == 12 and loaded.position == 2
    for key in loaded.arrays:
        np.testing.assert_array_equal(loaded.arrays[key], tensor.arrays[key].cpu().numpy())


def test_batched_placement_matches_original_scalar_formula():
    missions = [dict(start=[i*.1, i*.3, 0., i*.05], goal=[i*.4, i*.2, 0., i*.02],
                     yaw=i*.3, deadline_steps=123+i) for i in range(11)]
    origins = torch.arange(33).reshape(11, 3).float()
    clearance = torch.full((11,), .52)
    original = torch.randn(11, 13)
    expected, goals = original.clone(), origins.clone()
    for i, m in enumerate(missions):
        expected[i, :2] = origins[i, :2]+torch.tensor(m['start'][:2])
        expected[i, 2] = origins[i, 2]+m['start'][3]+clearance[i]+.02
        expected[i, 3:7] = torch.tensor([math.cos(m['yaw']/2), 0., 0., math.sin(m['yaw']/2)])
        expected[i, 7:] = 0
        goals[i, :2] = origins[i, :2]+torch.tensor(m['goal'][:2])
        goals[i, 2] = origins[i, 2]+m['goal'][3]
    actual, actual_goals, deadlines = place_missions(original, origins, clearance, missions)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_goals, goals, rtol=0, atol=0)
    assert deadlines.tolist() == [123+i for i in range(11)]


def test_deferred_metrics_preserve_sac_update():
    first = SAC(settings=replace(SAC_CFG, batch_size=4))
    second = SAC(settings=first.cfg)
    second.load_state_dict(first.state_dict())
    a, c = synthetic_observations(4)
    buffer = ReplayBuffer(8)
    buffer.add(a, c, torch.zeros(4, 2), torch.ones(4), torch.ones(4, dtype=torch.bool), a, c)
    batch = buffer.sample(4, 'cpu')
    torch.manual_seed(7)
    expected = first.update(batch)
    torch.manual_seed(7)
    actual = materialize_metrics(second.update(batch, defer_metrics=True))
    assert actual == expected
    for key, value in first.policy.state_dict().items():
        torch.testing.assert_close(second.policy.state_dict()[key], value, rtol=0, atol=0)
