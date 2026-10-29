"""Offline checks for coverage, data separation and the automatic launcher."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from train_general_big_map_test6_SAC.calibration import (view_plan, point_labels, Reservoir,
                                                    validate_calibration, validation_metrics)
from train_general_big_map_test6_SAC.config import PCA_DATASET, TERRAIN_NAMES
from train_general_big_map_test6_SAC.layouts import map_metadata
from train_general_big_map_test6_SAC.perception import FrozenPCA


def test_real_map_view_coverage_split_and_ice_safety():
    with np.load(HERE/'artifacts/spawn_bank.npz') as bank:
        points, evaluation = bank['points'], bank['evaluation']
    plan = view_plan(points, evaluation, map_metadata())
    assert plan == view_plan(points, evaluation, map_metadata())
    assert len(plan) == 7*(96+24)
    ids = {}
    for split, expected in [('train', 96), ('validation', 24)]:
        views = [v for v in plan if v['split'] == split]
        assert np.bincount([v['target_material'] for v in views]).tolist() == [expected]*7
        ids[split] = {v['point_id'] for v in views}
        assert not evaluation[list(ids[split])].any()
        assert not (points[list(ids[split]), 4] == 1).any()
    assert ids['train'].isdisjoint(ids['validation'])


def test_material_mapping_rejects_boundaries_and_outside():
    meta = map_metadata()
    centers = np.array([[c*6., r*3.] for r in range(7) for c in range(7)])
    expected = [TERRAIN_NAMES.index(x) for row in meta['grid'] for x in row]
    np.testing.assert_array_equal(point_labels(centers, meta), expected)
    np.testing.assert_array_equal(point_labels(np.array([[-3., 0.], [39., 2.], [0., 20.], [3.01, 0.]]), meta), [-1]*4)


def test_reservoir_balances_all_materials_and_refuses_missing_coverage():
    r = Reservoir(1100, 1)
    rng = np.random.default_rng(2)
    for _ in range(10):
        y = np.repeat(np.arange(7), 150)
        r.add(rng.normal(size=(len(y), 384)).astype(np.float32), y)
    x, y = r.balanced(1000)
    assert x.shape == (7700, 384)
    np.testing.assert_array_equal(np.bincount(y), [1100]*7)
    r.samples[6] = []
    with pytest.raises(ValueError, match='Insufficient PCA coverage'):
        r.balanced(1000)


def test_validation_and_legacy_pca_rejection():
    identity = {'map': 'current', 'dino': 'fixed'}
    metadata = dict(dataset=PCA_DATASET, identity=identity, materials=list(TERRAIN_NAMES),
                    fitted_per_terrain=1000, validation_counts=[250]*7)
    pca = FrozenPCA(np.zeros(384), np.eye(384)[:32], np.ones(32)/32, metadata)
    validate_calibration(pca, identity)
    x = np.zeros((140, 384), dtype=np.float32)
    x[:, 0] = np.arange(140)%20
    report = validation_metrics(pca, x, np.repeat(np.arange(7), 20))
    assert report['overall']['variance_retained'] == pytest.approx(1.)
    with pytest.raises(ValueError, match='not calibrated'):
        validate_calibration(pca, {'map': 'different'})
    old = FrozenPCA.load(HERE/'artifacts/pca32.npz')
    with pytest.raises(ValueError, match='not calibrated'):
        validate_calibration(old, old.metadata['identity'])


@pytest.mark.parametrize('arguments, failure, expected', [
    (['--timesteps', '64', '--device', 'cuda:1', '--pca', '/tmp/new pca.npz'], False, ['prepare_pca.py', 'train.py']),
    ([], True, ['prepare_pca.py']),
    (['--resume', '/tmp/checkpoint.pt'], False, ['train.py']),
    (['--help'], False, ['train.py']),
])
def test_launcher_sequence_and_failure_stop(tmp_path, arguments, failure, expected):
    launcher = tmp_path/'fake_isaaclab'
    launcher.write_text('''#!/usr/bin/env python3
import json, os, sys
with open(os.environ['CALL_LOG'], 'a') as f: f.write(json.dumps(sys.argv[1:])+'\\n')
if os.environ['FAIL_PREP'] == '1' and sys.argv[2].endswith('prepare_pca.py'): sys.exit(12)
''')
    launcher.chmod(0o755)
    log = tmp_path/'calls.jsonl'
    env = dict(os.environ, GENERAL_MAP_ISAACLAB=str(launcher), CALL_LOG=str(log), FAIL_PREP=str(int(failure)))
    result = subprocess.run(['bash', str(HERE/'train.sh'), *arguments], env=env, capture_output=True)
    assert (result.returncode != 0) == failure, result.stderr
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert [Path(call[1]).name for call in calls] == expected
    if '--pca' in arguments:
        assert calls[0][calls[0].index('--output')+1] == '/tmp/new pca.npz'
        assert calls[0][-2:] == ['--output', '/tmp/new pca.npz']
        assert 'cuda:1' in calls[0]
