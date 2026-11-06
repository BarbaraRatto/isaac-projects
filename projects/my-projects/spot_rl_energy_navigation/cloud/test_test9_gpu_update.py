"""Installer transaction and rollback tests, no GPU/software installation."""
import importlib.util
import json
from pathlib import Path
import sys
import subprocess

import pytest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location('test9_update_installer', HERE/'install_test9_gpu_update.py')
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


def fixture(tmp_path):
    project, update = tmp_path/'project', tmp_path/'update'
    project.mkdir()
    update.mkdir()
    base = {}
    files = {}
    prefix = 'spot_rl_energy_navigation/train_general_big_map_test9_SAC/'
    for name, old, new in [(prefix+'train.py', 'old = True\n', 'old = False\n'),
                           (prefix+'new.py', None, 'created = True\n')]:
        target = project/name
        target.parent.mkdir(parents=True, exist_ok=True)
        if old is not None:
            target.write_text(old)
            base[name] = installer.sha(target)
        payload = update/'payload'/name
        payload.parent.mkdir(parents=True, exist_ok=True)
        payload.write_text(new)
        files[name] = dict(old_sha256=base.get(name), new_sha256=installer.sha(payload))
    bundle = project/'spot_rl_energy_navigation/cloud/bundle-manifest.json'
    bundle.parent.mkdir(parents=True)
    bundle.write_text(json.dumps(dict(primary_experiment='train_general_big_map_test9_SAC', files=base)))
    (update/'update.json').write_text(json.dumps(dict(id='test', files=files, base_files=base)))
    return project, update, files


def test_install_updates_checksums_and_rollback_preserves_runs(tmp_path):
    project, update, files = fixture(tmp_path)
    run = project/'runs'
    run.mkdir()
    (run/'checkpoint.pt').write_bytes(b'leave unchanged')
    installer.check(update, project)
    installer.install(update, project)
    for name, record in files.items():
        assert installer.sha(project/name) == record['new_sha256']
    bundle = json.loads((project/'spot_rl_energy_navigation/cloud/bundle-manifest.json').read_text())
    assert all(bundle['files'][name] == record['new_sha256'] for name, record in files.items())
    installer.restore(project, Path(bundle['updates'][0]['backup']))
    assert (project/next(iter(files))).read_text() == 'old = True\n'
    assert not (project/list(files)[1]).exists()
    assert (run/'checkpoint.pt').read_bytes() == b'leave unchanged'


def test_reject_unknown_edits_and_corrupt_payload(tmp_path):
    project, update, files = fixture(tmp_path)
    target = project/next(iter(files))
    target.write_text('user edit\n')
    with pytest.raises(ValueError, match='Unexpected local edits'):
        installer.check(update, project)
    target.write_text('old = True\n')
    (update/'payload'/next(iter(files))).write_text('corrupt\n')
    with pytest.raises(ValueError, match='Corrupt update'):
        installer.check(update, project)


def test_failed_install_restores_all_originals(tmp_path, monkeypatch):
    project, update, files = fixture(tmp_path)
    original_copy = installer.atomic_copy
    def fail(source, target):
        if source == update/'payload'/list(files)[1]:
            raise OSError('injected install error')
        original_copy(source, target)
    monkeypatch.setattr(installer, 'atomic_copy', fail)
    with pytest.raises(OSError, match='injected'):
        installer.install(update, project)
    assert (project/next(iter(files))).read_text() == 'old = True\n'
    assert not (project/list(files)[1]).exists()


def test_resume_preserves_old_training_parameters():
    sys.path.insert(0, str(HERE.parent))
    from train_general_big_map_test9_SAC.resume_gpu import resume_command
    arguments = dict(device='cuda:0', timesteps=1500000, num_envs=128, gradient_steps=16,
                     seed=12, dino_batch_size=128, torch_threads=4,
                     eval_every=10000, checkpoint_every=10000, recovery_minutes=30)
    command = resume_command(Path('/project'), Path('/runs'), Path('/runs/old'), {'arguments': arguments})
    for key, value in arguments.items():
        flag = '--'+key.replace('_', '-')
        assert command[command.index(flag)+1] == str(value)
    assert command[command.index('--rgb-backend')+1] == 'cuda'
    assert command[command.index('--replay-device')+1] == 'cuda:0'


def test_installer_refuses_a_running_training_process():
    process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(10)',
                                '/fake/train_general_big_map_test9_SAC/train.py'])
    try:
        with pytest.raises(RuntimeError, match='Training still active'):
            installer.assert_stopped()
    finally:
        process.terminate()
        process.wait(timeout=5)
