"""Numerical checks for the navigation-only heading cost."""
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_general_big_map_test5.rewards import orientation_penalty


def test_orientation_dead_zone_symmetry_and_wrap():
    degrees = torch.tensor([0., 90., 110., 120., 150., 180., -120., -180., 240., 360.])
    actual = orientation_penalty(torch.deg2rad(degrees), torch.ones(10, dtype=torch.bool), .2)
    expected = -.4*torch.tensor([0., 0., 0., 1/7, 4/7, 1., 1/7, 1., 1/7, 0.])
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)


def test_orientation_time_scaling_and_alignment_exclusion():
    error = torch.tensor([torch.pi, torch.pi])
    navigation = torch.tensor([False, True])
    one_second = orientation_penalty(error, navigation, 1.)
    torch.testing.assert_close(one_second, torch.tensor([0., -2.]))
    torch.testing.assert_close(5*orientation_penalty(error, navigation, .2), one_second)
