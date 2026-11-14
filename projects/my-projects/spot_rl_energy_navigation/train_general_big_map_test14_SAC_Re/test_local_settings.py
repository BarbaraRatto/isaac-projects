"""Local training settings and evaluation-count contract."""
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from train_general_big_map_test14_SAC_Re.config import SAC_CFG, MISSION_CFG
from train_general_big_map_test14_SAC_Re.layouts import training_conditions


def test_local_defaults_preserve_update_ratio_and_evaluation_sample_count():
    assert SAC_CFG.timesteps==600000
    assert SAC_CFG.num_envs==len(training_conditions())==32
    assert SAC_CFG.gradient_steps==4
    assert SAC_CFG.gradient_steps/SAC_CFG.num_envs==16/128
    assert SAC_CFG.eval_episodes_per_env==4
    assert SAC_CFG.num_envs*SAC_CFG.eval_episodes_per_env==128
    assert MISSION_CFG.evaluation_intervals==(10000,10000,30000,30000)
    assert MISSION_CFG.spawn_excluded==MISSION_CFG.goal_excluded==('lab_blackice',)
