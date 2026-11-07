"""Durable, paired SAC checkpoints. No simulator imports.

One writer per training run. A complete generation is committed by a directory
rename only after replay, policy and manifest have been flushed. Temporary
directories are never considered for resume. A directory resume can fall back
to an older verified generation; an explicit checkpoint fails if corrupted.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import uuid
import warnings

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train_general_big_map_test13_SAC_Re.artifacts import save_checkpoint


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def fsync_directory(path):
    fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name('.'+path.name+'.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def check_space(directory, replay_bytes, reserve_gib=1.):
    # Full replay plus conservative room for Adam/targets/policy and a disk reserve.
    required = replay_bytes + 512*1024**2 + int(reserve_gib*1024**3)
    free = shutil.disk_usage(directory).free
    if free < required:
        raise OSError(f'Not enough free space for a NEW complete recovery snapshot: '
                      f'{free/1024**3:.2f} GiB free, {required/1024**3:.2f} GiB required. '
                      'Existing snapshots have been preserved. Free space or use a larger persistent volume.')


def verify_generation(directory):
    directory = Path(directory)
    manifest = json.loads((directory/'manifest.json').read_text())
    if manifest.get('schema') != 1 or manifest.get('complete') is not True:
        raise ValueError('Not a complete recovery snapshot')
    if set(manifest.get('files', {})) != {'policy.pt', 'replay.npz'}:
        raise ValueError('Incomplete recovery manifest')
    for name, expected in manifest['files'].items():
        path = directory/name
        if not path.is_file() or path.stat().st_size != expected['bytes'] or sha256(path) != expected['sha256']:
            raise ValueError(f'Corrupt or missing recovery file: {path}')
    # Validate lightweight replay metadata without loading the multi-GB arrays.
    with np.load(directory/'replay.npz', allow_pickle=False) as stored:
        actual = {key: int(stored[key]) for key in ('capacity', 'size', 'position', 'total')}
    if actual != manifest['replay'] or actual['total'] != manifest['transitions']:
        raise ValueError('Recovery manifest/replay counters differ')
    if (actual['size'] != min(actual['total'], actual['capacity']) or
            actual['position'] != actual['total'] % actual['capacity']):
        raise ValueError('Invalid replay ring counters')
    return manifest


def generations(root):
    return sorted((p for p in Path(root).glob('snapshot_*') if p.is_dir()), reverse=True)


def resolve_resume(path):
    """Accept a run, its recovery directory, a complete generation, or a .pt file."""
    path = Path(path).expanduser().resolve()
    if path.is_file():
        if path.name == 'policy.pt' and (path.parent/'manifest.json').is_file():
            verify_generation(path.parent)
            return path
        state = torch.load(path, map_location='cpu', weights_only=False)
        complete = (state.get('training_state') or {}).get('complete_snapshot')
        if not complete:
            raise ValueError('This is a model-only checkpoint. Resume from the RUN DIRECTORY '
                             '(automatic verified fallback) or a complete recovery/snapshot_*/policy.pt.')
        directory = (path.parent/complete).resolve()
        manifest = verify_generation(directory)
        if manifest['transitions'] != state['learner']['transitions']:
            raise ValueError('Final checkpoint points to a different recovery generation')
        return directory/'policy.pt'
    if (path/'manifest.json').is_file() and path.name.startswith('snapshot_'):
        verify_generation(path)
        return path/'policy.pt'
    root = path/'recovery' if (path/'recovery').is_dir() else path
    failures = []
    for directory in generations(root):
        try:
            verify_generation(directory)
            return directory/'policy.pt'
        except (OSError, ValueError, KeyError) as exc:
            warnings.warn(f'Skipping invalid recovery snapshot {directory.name}: {exc}')
            failures.append(str(exc))
    raise ValueError(f'No complete verified snapshot in {root}. '+ '; '.join(failures))


class RecoveryManager:
    def __init__(self, run, interval_seconds=1800., keep=2, reserve_gib=1.):
        if interval_seconds <= 0 or keep < 2 or reserve_gib < 0:
            raise ValueError('Recovery interval must be positive, keep >= 2, reserve >= 0')
        self.run = Path(run)
        self.root = self.run/'recovery'
        self.root.mkdir(exist_ok=True)
        self.interval_seconds, self.keep, self.reserve_gib = interval_seconds, keep, reserve_gib
        self.last_saved = time.monotonic()

    def due(self):
        return time.monotonic()-self.last_saved >= self.interval_seconds

    def save(self, learner, replay, pca, identity, best_score, training_state, reason):
        if replay.total != learner.transitions:
            raise ValueError('Refusing to save mismatched learner/replay counters')
        check_space(self.root, replay.nbytes, self.reserve_gib)
        name = f'snapshot_{learner.transitions:012d}_{time.time_ns()}'
        temporary = self.root/('.pending_'+uuid.uuid4().hex)
        temporary.mkdir()
        started = time.monotonic()
        try:
            replay.save(temporary/'replay.npz')
            state = dict(training_state, replay_snapshot=dict(file='replay.npz', transitions=learner.transitions),
                         complete_snapshot='.')
            save_checkpoint(temporary/'policy.pt', learner, pca, identity, best_score, state)
            files = {}
            for filename in ('policy.pt', 'replay.npz'):
                path = temporary/filename
                with path.open('rb') as stream:
                    os.fsync(stream.fileno())
                files[filename] = dict(bytes=path.stat().st_size, sha256=sha256(path))
            manifest = dict(schema=1, complete=True, transitions=learner.transitions,
                            updates=learner.updates, reason=reason,
                            created_utc=datetime.now(timezone.utc).isoformat(), files=files,
                            replay=dict(capacity=replay.capacity, size=replay.size,
                                        position=replay.position, total=replay.total))
            atomic_json(temporary/'manifest.json', manifest)
            verify_generation(temporary)
            destination = self.root/name
            os.replace(temporary, destination)
            fsync_directory(self.root)
            record = dict(snapshot=name, transitions=learner.transitions, reason=reason,
                          created_utc=manifest['created_utc'], seconds=time.monotonic()-started,
                          checkpoint=f'recovery/{name}/policy.pt',
                          save_bytes=sum(v['bytes'] for v in files.values()))
            atomic_json(self.root/'latest.json', record)
            atomic_json(self.run/'recovery_status.json', record)
            self.last_saved = time.monotonic()
            # Only prune after a complete new generation exists. Retain the previous
            # verified generation even if a newer, older-on-disk candidate is corrupt.
            valid = []
            for candidate in generations(self.root):
                try:
                    if candidate != destination:
                        verify_generation(candidate)
                    valid.append(candidate)
                except (OSError, ValueError, KeyError):
                    continue
            for old in valid[self.keep:]:
                shutil.rmtree(old)
            fsync_directory(self.root)
            print(f'[RECOVERY] Complete: {destination} ({learner.transitions:,} transitions; '
                  f'{record["seconds"]:.1f}s). Keep the entire RUN DIRECTORY on persistent storage.', flush=True)
            return destination
        except BaseException:
            # An actual kill may leave .pending_*; readers ignore these.
            if temporary.exists():
                shutil.rmtree(temporary)
            raise


class StopRequest:
    def __init__(self):
        self.signal = None

    def __call__(self, signum, frame):
        self.signal = signum
        print(f'[RECOVERY] Signal {signum}: stopping at the next safe training boundary, then saving.', flush=True)


@contextmanager
def graceful_signals():
    """Never interrupt halfway through a replay insertion or optimizer update."""
    request = StopRequest()
    originals = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in originals:
        signal.signal(sig, request)
    try:
        yield request
    finally:
        for sig, handler in originals.items():
            signal.signal(sig, handler)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Verify a full recovery snapshot without simulation; prints its checkpoint path.')
    parser.add_argument('path', type=Path)
    args = parser.parse_args()
    print(resolve_resume(args.path))


if __name__ == '__main__':
    main()
