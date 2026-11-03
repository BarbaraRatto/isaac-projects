"""Failure injection for committed snapshots; CPU only, never launches Isaac Sim."""
from pathlib import Path
import signal
import os
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_general_big_map_test9_SAC import recovery
from train_general_big_map_test9_SAC.config import SAC_CFG
from train_general_big_map_test9_SAC.replay import ReplayBuffer
from train_general_big_map_test9_SAC.test_offline import pca_fixture


class Learner:
    cfg = SAC_CFG
    transitions = updates = 0

    def state_dict(self):
        return dict(transitions=self.transitions, updates=self.updates, weight=torch.tensor([3.]))


def advance(learner, replay, count):
    for array in replay.arrays.values():
        array.fill(0)
    learner.transitions = replay.total = count
    replay.size = min(count, replay.capacity)
    replay.position = count % replay.capacity


def save(manager, learner, replay):
    return manager.save(learner, replay, pca_fixture(), {'test':True}, None,
                        {'curriculum':{'level':2, 'streak':1}}, 'test')


def test_commit_and_retention_fallback_ignore_partial_generation(tmp_path):
    manager = recovery.RecoveryManager(tmp_path, reserve_gib=0)
    learner, replay = Learner(), ReplayBuffer(4)
    for count in (1, 2, 3):
        advance(learner, replay, count)
        save(manager, learner, replay)
    generations = recovery.generations(manager.root)
    assert len(generations) == 2
    (manager.root/'.pending_crash').mkdir()
    (manager.root/'.pending_crash'/'policy.pt').write_bytes(b'partial')
    assert recovery.resolve_resume(tmp_path) == generations[0]/'policy.pt'
    # Even a same-length corruption must be detected, not just missing files.
    with (generations[0]/'replay.npz').open('r+b') as stream:
        stream.write(b'xxxx')
    with pytest.warns(UserWarning, match='Skipping invalid'):
        assert recovery.resolve_resume(tmp_path) == generations[1]/'policy.pt'
    with pytest.raises(ValueError, match='Corrupt'):
        recovery.resolve_resume(generations[0]/'policy.pt')
    # A fresh commit retains the older GOOD copy, rather than the corrupted copy.
    advance(learner, replay, 4)
    newest = save(manager, learner, replay)
    assert generations[1].exists()
    assert recovery.resolve_resume(newest) == newest/'policy.pt'


def test_failed_write_preserves_previous_complete_snapshot(tmp_path, monkeypatch):
    manager = recovery.RecoveryManager(tmp_path, reserve_gib=0)
    learner, replay = Learner(), ReplayBuffer(4)
    advance(learner, replay, 2)
    good = save(manager, learner, replay)
    def fail(path):
        Path(path).write_bytes(b'incomplete replay')
        raise OSError('injected disk failure')
    monkeypatch.setattr(replay, 'save', fail)
    with pytest.raises(OSError, match='injected'):
        save(manager, learner, replay)
    assert recovery.resolve_resume(tmp_path) == good/'policy.pt'
    assert not list(manager.root.glob('.pending_*'))


def test_mismatch_and_insufficient_disk_never_replace_last_good(tmp_path, monkeypatch):
    manager = recovery.RecoveryManager(tmp_path, reserve_gib=0)
    learner, replay = Learner(), ReplayBuffer(4)
    advance(learner, replay, 2)
    good = save(manager, learner, replay)
    learner.transitions = 3
    with pytest.raises(ValueError, match='mismatched'):
        save(manager, learner, replay)
    learner.transitions = 2
    monkeypatch.setattr(recovery.shutil, 'disk_usage', lambda _: SimpleNamespace(free=1))
    with pytest.raises(OSError, match='free space'):
        save(manager, learner, replay)
    assert recovery.resolve_resume(tmp_path) == good/'policy.pt'


def test_final_alias_is_portable_and_model_only_resume_rejected(tmp_path):
    manager = recovery.RecoveryManager(tmp_path, reserve_gib=0)
    learner, replay = Learner(), ReplayBuffer(4)
    advance(learner, replay, 3)
    good = save(manager, learner, replay)
    state = torch.load(good/'policy.pt', weights_only=False)
    assert state['training_state']['curriculum'] == {'level':2, 'streak':1}
    state['training_state']['complete_snapshot'] = str(good.relative_to(tmp_path))
    torch.save(state, tmp_path/'policy_final.pt')
    assert recovery.resolve_resume(tmp_path/'policy_final.pt') == good/'policy.pt'
    state['training_state']['complete_snapshot'] = None
    torch.save(state, tmp_path/'policy_latest.pt')
    with pytest.raises(ValueError, match='model-only'):
        recovery.resolve_resume(tmp_path/'policy_latest.pt')


def test_signals_defer_stop_until_safe_boundary():
    previous = signal.getsignal(signal.SIGTERM)
    with recovery.graceful_signals() as stop:
        os.kill(os.getpid(), signal.SIGTERM)
        assert stop.signal == signal.SIGTERM  # no async exception inside update/write
    assert signal.getsignal(signal.SIGTERM) == previous
