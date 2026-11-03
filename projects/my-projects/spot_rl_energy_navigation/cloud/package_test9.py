#!/usr/bin/env python3
"""Create ONE complete test9 cloud archive, with portable USD assets and textures.

Use Isaac Sim python.sh (pxr required), without launching SimulationApp.
No previous runs, model checkpoints, private environment files or credentials.
"""
import argparse
import json
from pathlib import Path
import shutil
import tarfile
import tempfile

import package
from portable_test9 import (EXPERIMENT, make_map_portable, copy_camera, verify_usd_assets)
from verify import verify_files, verify_bundle

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
TOP = 'spot-test9-sac-cloud'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'dist/spot-test9-sac-cloud.tar.gz')
    parser.add_argument('--zed', type=Path,
                        default=Path('/home/isaac/isaac-sim/assets/Assets/Isaac/5.1/Isaac/Sensors/Stereolabs/ZED_X/ZED_X.usdc'))
    args = parser.parse_args()
    textures = make_map_portable(PROJECT)
    files = package.collect_files([EXPERIMENT], args.zed)
    for name in textures:
        files[name] = PROJECT/name
    # The top-level guide gives only test9 commands; older guides remain available.
    files['LEGGIMI_TEST9.md'] = HERE/'CLOUD_TEST9.md'
    files['start_test9.sh'] = HERE/'start_test9.sh'
    files['monitor_test9.sh'] = HERE/'monitor_test9.sh'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    required = sum(p.stat().st_size for p in files.values())*3 + 100*1024**2
    if min(shutil.disk_usage(tempfile.gettempdir()).free, shutil.disk_usage(args.output.parent).free) < required:
        raise OSError(f'Need approximately {required/1024**3:.1f} GiB free for staging/archive/verification')
    temporary_archive = args.output.with_suffix('.gz.tmp')
    try:
        with tempfile.TemporaryDirectory(prefix='spot-test9-bundle-') as directory:
            project = Path(directory)/TOP
            for name, source in files.items():
                target = project/name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            copy_camera(args.zed, project/'.cloud_assets/ZED_X.usdc')
            # Confirm assets remain resolvable after relocation, with no source-PC paths.
            count = verify_usd_assets(project)
            verify_files(project, zed=project/'.cloud_assets/ZED_X.usdc', experiment=EXPERIMENT)
            manifest = dict(schema=1, primary_experiment=EXPERIMENT, experiments=[EXPERIMENT],
                            files={p.relative_to(project).as_posix(): package.sha256(p)
                                   for p in sorted(project.rglob('*')) if p.is_file()})
            manifest_path = project/'spot_rl_energy_navigation/cloud/bundle-manifest.json'
            manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
            verify_bundle(project)
            with tarfile.open(temporary_archive, 'w:gz', compresslevel=3) as archive:
                archive.add(project, arcname=TOP)
            # Read every archive payload and compare against the staged checksums.
            import hashlib
            with tarfile.open(temporary_archive) as archive:
                names = set()
                for member in archive:
                    if not member.isfile():
                        if not member.isdir():
                            raise ValueError('Archive contains a link or special file')
                        continue
                    relative = Path(member.name).relative_to(TOP).as_posix()
                    if relative in names or '..' in Path(relative).parts:
                        raise ValueError('Unsafe or duplicate archive entry')
                    names.add(relative)
                    if relative in manifest['files']:
                        h = hashlib.sha256()
                        with archive.extractfile(member) as stream:
                            for block in iter(lambda: stream.read(1024*1024), b''):
                                h.update(block)
                        if h.hexdigest() != manifest['files'][relative]:
                            raise ValueError(f'Archive payload mismatch: {relative}')
                if names != set(manifest['files']) | {'spot_rl_energy_navigation/cloud/bundle-manifest.json'}:
                    raise ValueError('Archive file inventory differs from manifest')
            temporary_archive.replace(args.output)
            checksum = package.sha256(args.output)
            args.output.with_suffix(args.output.suffix+'.sha256').write_text(checksum+'  '+args.output.name+'\n')
            report = dict(archive=args.output.name, sha256=checksum, bytes=args.output.stat().st_size,
                          files=len(manifest['files']), relocated_usd_asset_references=count,
                          primary_experiment=EXPERIMENT, contains_terrain_textures=True,
                          package_integrity_verified=True, simulation_launched=False,
                          installation_on_cloud_tested=False)
            args.output.with_suffix('.verification.json').write_text(json.dumps(report, indent=2)+'\n')
            print(f'Ready: {args.output}\nSize: {args.output.stat().st_size/1024**2:.1f} MiB\nSHA256: {checksum}')
    finally:
        temporary_archive.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
