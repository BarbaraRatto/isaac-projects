"""Portable two-terrain input verification before installation; no simulator needed."""
import importlib.util
import json
from pathlib import Path

import pytest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('test12_verify',HERE/'verify.py')
verify=importlib.util.module_from_spec(spec); spec.loader.exec_module(verify)
NAME='train_general_big_map_test12_SAC_Re'


def fixture(tmp):
    exp=tmp/'spot_rl_energy_navigation'/NAME
    paths=['IsaacRobotics/assets/spot.usd','IsaacRobotics/policies/spot/models/spot_policy.pt','.cloud_assets/ZED_X.usdc',
           'spot_rl_energy_navigation/'+NAME+'/train.py','spot_rl_energy_navigation/'+NAME+'/maps/lab.usda',
           'spot_rl_energy_navigation/'+NAME+'/materials.json']
    for relative in paths:
        path=tmp/relative; path.parent.mkdir(parents=True,exist_ok=True); path.write_text('fixture')
    manifest=dict(identity=dict(dino={},missions={},terrain={'lab':verify.sha256(exp/'maps/lab.usda')}),
                  pca_source='artifacts/pca32_lab.npz',pca_status='pending_new_lab_calibration',
                  pca_lifecycle='calibrate_on_first_training',terrain_files={'lab':'maps/lab.usda'},
                  runtime_input_files={'materials.json':verify.sha256(exp/'materials.json')})
    (exp/'artifacts').mkdir(); path=exp/'artifacts/manifest.json'; path.write_text(json.dumps(manifest))
    return exp,path,manifest


def test_pending_pca_and_usda_are_explicitly_declared(tmp_path):
    exp,path,manifest=fixture(tmp_path)
    verify.verify_files(tmp_path,experiment=NAME)
    manifest.pop('pca_lifecycle'); path.write_text(json.dumps(manifest))
    with pytest.raises(FileNotFoundError,match='PCA'):
        verify.verify_files(tmp_path,experiment=NAME)


def test_ready_pca_required_and_runtime_hash_enforced(tmp_path):
    exp,path,manifest=fixture(tmp_path)
    manifest['pca_status']='ready'; path.write_text(json.dumps(manifest))
    with pytest.raises(FileNotFoundError,match='PCA'):
        verify.verify_files(tmp_path,experiment=NAME)
    (exp/'artifacts/pca32_lab.npz').write_text('fixture')
    verify.verify_files(tmp_path,experiment=NAME)
    (exp/'materials.json').write_text('changed')
    with pytest.raises(ValueError,match='changed frozen input'):
        verify.verify_files(tmp_path,experiment=NAME)


def test_pending_does_not_allow_pca_path_escape(tmp_path):
    exp,path,manifest=fixture(tmp_path)
    manifest['pca_source']='../../outside.npz'; path.write_text(json.dumps(manifest))
    with pytest.raises(FileNotFoundError,match='PCA'):
        verify.verify_files(tmp_path,experiment=NAME)
