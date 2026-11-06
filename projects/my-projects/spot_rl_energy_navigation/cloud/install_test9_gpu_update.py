"""Install/rollback the small test9 update using only standard Python.

Refuses active training, unknown base code, modified assets, or corrupt payloads.
Copies originals to a separate backup before replacing any project file.
Never touches runs, recovery snapshots, assets, dependencies or env.sh.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_stopped():
    active = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            if any(arg.endswith(b'train_general_big_map_test9_SAC/train.py')
                   for arg in path.read_bytes().split(b'\0')):
                active.append(path.parent.name)
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            pass
    if active:
        raise RuntimeError('Training still active (PIDs '+', '.join(active)+
                           '). Ctrl+C inside its tmux, then wait for complete save and shell prompt.')


def safe(project, name):
    path = project/name
    if not name or not path.resolve().is_relative_to(project.resolve()):
        raise ValueError('Invalid update path: '+name)
    return path


def atomic_copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.'+target.name+'.', dir=target.parent)
    try:
        with os.fdopen(fd, 'wb') as stream:
            stream.write(source.read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, source.stat().st_mode & 0o777)
        os.replace(temporary, target)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def check(update, project):
    manifest = json.loads((update/'update.json').read_text())
    bundle_file = project/'spot_rl_energy_navigation/cloud/bundle-manifest.json'
    bundle = json.loads(bundle_file.read_text())
    if bundle.get('primary_experiment') != 'train_general_big_map_test9_SAC':
        raise ValueError('Target must be the original test9 cloud package')
    for name, record in manifest['files'].items():
        payload = safe(update/'payload', name)
        if not payload.is_file() or sha(payload) != record['new_sha256']:
            raise ValueError('Corrupt update payload: '+name)
        if name.endswith('.py'):
            compile(payload.read_text(), str(payload), 'exec')
        target = safe(project, name)
        actual = sha(target) if target.is_file() else None
        if actual not in (record['old_sha256'], record['new_sha256']):
            raise ValueError('Unexpected local edits / wrong base: '+name)
    # Verify all ORIGINAL package assets and code, not just the replaced files.
    # Payload files may already be installed (idempotent install).
    for name, expected in manifest['base_files'].items():
        path = safe(project, name)
        allowed = {expected}
        if name in manifest['files']:
            allowed.add(manifest['files'][name]['new_sha256'])
        if not path.is_file() or sha(path) not in allowed:
            raise ValueError('Base package missing/changed: '+name)
    return manifest, bundle_file, bundle


def restore(project, backup):
    record = json.loads((backup/'backup.json').read_text())
    for name, expected in record['original'].items():
        target = safe(project, name)
        if expected is None:
            if target.exists():
                target.unlink()
        else:
            source = safe(backup/'files', name)
            if sha(source) != expected:
                raise ValueError('Corrupt backup: '+name)
            atomic_copy(source, target)


def install(update, project):
    assert_stopped()
    manifest, bundle_file, bundle = check(update, project)
    backup = project/'spot_rl_energy_navigation/cloud/update-backups'/(
        'test9-gpu-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    backup.mkdir(parents=True)
    names = [*manifest['files'], str(bundle_file.relative_to(project))]
    original = {}
    for name in names:
        path = safe(project, name)
        original[name] = sha(path) if path.is_file() else None
        if path.is_file():
            destination = backup/'files'/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    (backup/'backup.json').write_text(json.dumps(dict(original=original), indent=2)+'\n')
    try:
        for name in manifest['files']:
            atomic_copy(update/'payload'/name, project/name)
            bundle['files'][name] = manifest['files'][name]['new_sha256']
        bundle.setdefault('updates', []).append(dict(id=manifest['id'], backup=str(backup)))
        temporary = backup/'new-bundle-manifest.json'
        temporary.write_text(json.dumps(bundle, indent=2)+'\n')
        atomic_copy(temporary, bundle_file)
        check(update, project)
    except BaseException:
        restore(project, backup)
        raise
    print('[UPDATE] Installed; originals backed up at: '+str(backup))
    print('[UPDATE] Runs, maps, PCA, DINO and software environment preserved.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project', type=Path)
    parser.add_argument('--check', action='store_true', help='read-only checks; safe while training continues')
    parser.add_argument('--rollback', type=Path, help='restore a backup printed during installation')
    args = parser.parse_args()
    project = args.project.resolve()
    update = Path(__file__).resolve().parent
    if args.rollback:
        assert_stopped()
        restore(project, args.rollback.resolve())
        print('[UPDATE] Rollback complete.')
    elif args.check:
        check(update, project)
        print('[UPDATE] CHECK PASSED (no project files changed).')
    else:
        install(update, project)


if __name__ == '__main__':
    main()
