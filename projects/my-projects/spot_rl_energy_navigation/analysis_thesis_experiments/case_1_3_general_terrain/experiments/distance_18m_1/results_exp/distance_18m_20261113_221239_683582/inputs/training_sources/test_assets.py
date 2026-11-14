"""Frozen test9 map and PCA transfer invariants, without simulation."""
from pathlib import Path
import json
import sys
from collections import Counter
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_general_big_map_test9_SAC.layouts import map_metadata
from train_general_big_map_test9_SAC.config import DEFAULT_PCA
from train_general_big_map_test9_SAC.artifacts import source_identity
from train_general_big_map_test9_SAC.calibration import validate_calibration
from train_general_big_map_test9_SAC.perception import FrozenPCA


def test_counts_ice_and_agreed_swaps():
    grid = map_metadata()['grid']
    assert set(Counter(x for row in grid for x in row).values()) == {7}
    ice = [(r, c) for r, row in enumerate(grid) for c, name in enumerate(row) if name == 't2_slippery']
    assert len(ice) == 7
    assert all(max(abs(r-a), abs(c-b)) > 1 for i, (r, c) in enumerate(ice) for a, b in ice[i+1:])
    assert grid[4][4] == 't2_slippery'
    assert grid[6][5] == 't5_large_rocks'
    assert grid[2][0] == 't2_slippery'
    assert grid[3][5] == 't4_fine_gravel'


def test_original_pca_provenance_preserved_and_transfer_is_exact():
    pca = FrozenPCA.load(DEFAULT_PCA)
    identity = source_identity()
    assert pca.metadata['identity']['terrain'] != identity['terrain']
    validate_calibration(pca, identity)
    altered = FrozenPCA(pca.mean.copy(), pca.components.copy(), pca.explained.copy(), pca.metadata)
    altered.mean[0] += 1
    with pytest.raises(ValueError, match='not calibrated'):
        validate_calibration(altered, identity)
    bad_identity = dict(identity, terrain={'real_terrains_test9':'changed'})
    with pytest.raises(ValueError, match='not calibrated'):
        validate_calibration(pca, bad_identity)
