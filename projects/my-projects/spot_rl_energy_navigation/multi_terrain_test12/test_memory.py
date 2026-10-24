"""Numerical memory regression checks; no Isaac Sim or model inference."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch

from multi_terrain_test12.memory import EpisodeMap, MapBank
from multi_terrain_test12.perception import cell_indices, local_cells, world_to_local
from multi_terrain_test12.config import OBSERVATION_SIZE, CRITIC_SIZE
from multi_terrain_test12.artifacts import pca_compatible


EMPTY = np.empty((0, 3))
FEATURES = np.empty((0, 32))


def cloud():
    x, y = np.meshgrid(np.linspace(.34, .66, 17), np.linspace(-.16, .16, 17))
    return np.column_stack((x.ravel(), y.ravel(), .2*x.ravel()-.1*y.ravel()))


def test_memory_moves_behind_and_rotates_slopes_and_height():
    m = EpisodeMap()
    base = np.array([100., 200., .5])
    pts = cloud()+[100., 200., 0.]
    patch = np.array([[100.5, 200., .1]])
    m.update(patch, np.ones((1, 32)), pts, base, 0.)
    assert m.query(base, 0., 0.).shape == (81, 40)
    new_base = base+[1., 0., .1]
    # No new data; keep known terrain, aged by 10 seconds.
    m.update(EMPTY, FEATURES, EMPTY, new_base, 10.)
    result = m.query(new_base, 0., 10.)
    cell = cell_indices(np.array([[-.5, 0., 0.]]))[0]
    np.testing.assert_allclose(result[cell, :32], 1.)
    np.testing.assert_allclose(result[cell, 32:36], [-.5, np.arctan(.2), np.arctan(-.1), 0.], atol=1e-6)
    np.testing.assert_allclose(result[cell, 36:], [1, 1, .25, .25])
    rotated = m.query(base, np.pi/2, 10.)
    cell = cell_indices(np.array([[0., -.5, 0.]]))[0]
    np.testing.assert_allclose(rotated[cell, 33:35], [np.arctan(-.1), np.arctan(-.2)], atol=1e-6)
    assert rotated[:, 36].sum() == 1
    assert not rotated[40].any()  # centre not observed: no ground-truth fill


def test_dense_moments_preserve_roughness_and_do_not_count_revisits_twice():
    m = EpisodeMap()
    p = cloud()
    p[:, 2] += np.random.default_rng(4).normal(0, .01, len(p))
    base = np.array([0., 0., .5])
    m.update(EMPTY, FEATURES, p, base, 0.)
    expected = local_cells(EMPTY, FEATURES, world_to_local(p, base, 0.))
    result = m.query(base, 0., 0.)
    np.testing.assert_allclose(result[:, 32:38], expected[:, 32:38], atol=1e-6)
    count = sum(r[2] for r in m.geometry.values())
    m.update(EMPTY, FEATURES, p, base, 1.)
    assert sum(r[2] for r in m.geometry.values()) == count
    assert m.query(base, 0., 1.)[:, 39].max() == 0
    # Single-point observation cannot fabricate a plane.
    single = EpisodeMap()
    single.update(EMPTY, FEATURES, p[:1], base, 0.)
    assert single.query(base, 0., 0.)[:, 37].sum() == 0


def test_partial_reset_and_independent_modalities():
    bank = MapBank()
    base = np.zeros(3)
    for i in range(2):
        bank.observe(i, np.array([[.5, 0., 0.]]), np.full((1, 32), i+1.), cloud(), base, 0., 0.)
    bank.reset([0])
    a = bank.observe(0, EMPTY, FEATURES, EMPTY, base, 0., 0.)
    b = bank.observe(1, EMPTY, FEATURES, EMPTY, base, 0., 2.)
    assert not a.any()
    assert b[:, 36].sum() == 1
    assert b[:, 38].max() == .05
    # Geometry-only refresh must not refresh the age of the old RGB evidence.
    b = bank.observe(1, EMPTY, FEATURES, cloud(), base, 0., 4.)
    assert np.isclose(b[:, 38].max(), .1)
    assert b[:, 39].max() == 0.


def test_age_masks_and_pca_transfer_contract():
    from multi_terrain_test12.networks import ObservationNormalizer
    raw = torch.zeros(2, OBSERVATION_SIZE)
    cells = raw[:, 15:].view(2, 81, 40)
    cells[..., :36] = 123.
    cells[..., 38:] = .75
    cells[:, 40, 36:38] = 1
    out = ObservationNormalizer().actor(raw)[:, 15:].view(2, 81, 40)
    assert not out[:, :40].any()
    torch.testing.assert_close(out[:, 40, 38:], torch.full((2, 2), .75))
    identity = dict(dino={'weights': 'abc'}, input_size=518, resize='x', terrain='old')
    assert pca_compatible(identity, dict(identity, terrain='new'))
    assert not pca_compatible(identity, dict(identity, dino={'weights': 'changed'}))
    assert (OBSERVATION_SIZE, CRITIC_SIZE) == (3255, 3279)
