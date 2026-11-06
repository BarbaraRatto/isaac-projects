"""Build a small, baseline-bound code update; never rewrites the original archive."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import tempfile

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
EXPERIMENT = 'spot_rl_energy_navigation/train_general_big_map_test9_SAC'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=HERE/'dist/spot-test9-sac-cloud.tar.gz')
    parser.add_argument('--output', type=Path, default=HERE/'dist/spot-test9-gpu-update.tar.gz')
    parser.add_argument('--validation', type=Path)
    args = parser.parse_args()
    with tarfile.open(args.base) as archive:
        member = next(m for m in archive if m.name.endswith('/cloud/bundle-manifest.json'))
        base = json.load(archive.extractfile(member))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='test9-gpu-package-') as directory:
        root = Path(directory)/'spot-test9-gpu-update'
        root.mkdir()
        files = {}
        candidates = [*sorted((PROJECT/EXPERIMENT).iterdir()), HERE/'TEST9_GPU_UPDATE.md']
        for path in candidates:
            if not path.is_file() or path.suffix not in ('.py', '.sh', '.md'):
                continue
            name = str(path.relative_to(PROJECT))
            new = digest(path.read_bytes())
            old = base['files'].get(name)
            if new == old:
                continue
            destination = root/'payload'/name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            files[name] = dict(old_sha256=old, new_sha256=new)
        shutil.copy2(HERE/'install_test9_gpu_update.py', root/'install_update.py')
        shutil.copy2(HERE/'TEST9_GPU_UPDATE.md', root/'LEGGIMI.md')
        if args.validation:
            shutil.copy2(args.validation, root/'local_validation.json')
        (root/'update.json').write_text(json.dumps(dict(
            schema=1, id='test9-gpu-20261104', files=files,
            base_files=base['files'],
            unchanged='Reward, policy architecture, curriculum, scaler, assets and SAC settings.'), indent=2)+'\n')
        temporary = args.output.with_suffix('.tmp')
        with tarfile.open(temporary, 'w:gz', compresslevel=6) as archive:
            archive.add(root, arcname=root.name)
        temporary.replace(args.output)
    (args.output.with_suffix(args.output.suffix+'.sha256')).write_text(
        digest(args.output.read_bytes())+'  '+args.output.name+'\n')
    print(json.dumps(dict(archive=str(args.output), bytes=args.output.stat().st_size,
                         changed_files=len(files), sha256=digest(args.output.read_bytes())), indent=2))


if __name__ == '__main__':
    main()
