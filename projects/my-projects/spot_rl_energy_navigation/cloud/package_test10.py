#!/usr/bin/env python3
"""Create ONE complete test10 cloud archive, with portable USD assets and textures.

Use Isaac Sim python.sh (pxr required), without launching SimulationApp.
No previous runs, model checkpoints, private environment files or credentials.
"""
import argparse
import ast
import json
from pathlib import Path
import shutil
import tarfile
import tempfile

import package
from portable_test9 import copy_camera, verify_usd_assets

EXPERIMENT = "train_general_big_map_test10_SAC"
from verify import verify_files, verify_bundle

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
TOP = 'spot-test10-sac-cloud'


def verify_experiment_sources(project):
    """Catch missing sibling modules without importing Isaac or starting Kit."""
    nav = Path(project)/'spot_rl_energy_navigation'
    root = nav/EXPERIMENT
    for path in root.rglob('*.py'):
        source = path.read_text()
        compile(source, str(path), 'exec')
        for node in ast.walk(ast.parse(source)):
            modules = ([node.module] if isinstance(node, ast.ImportFrom) else
                       [alias.name for alias in node.names] if isinstance(node, ast.Import) else [])
            for module in modules:
                if not module or not module.startswith(EXPERIMENT+'.'):
                    continue
                target = nav.joinpath(*module.split('.'))
                if not target.with_suffix('.py').is_file() and not target.is_dir():
                    raise ValueError(f'Missing experiment module {module}, imported by {path.name}')
    if list((root/'artifacts').glob('frozen_scalers.*')):
        raise ValueError('Fresh bundle must calibrate its own scalers, not ship fitted values')


def terrain_textures(project):
    """Read-only dependency discovery; changing map bytes would invalidate identity."""
    from pxr import Usd, Sdf
    project = Path(project).resolve()
    experiment = project/'spot_rl_energy_navigation'/EXPERIMENT
    path = experiment/'maps/real_terrains_test10.usd'
    manifest = json.loads((experiment/'artifacts/manifest.json').read_text())
    if package.sha256(path) != manifest['identity']['terrain']['real_terrains_test10']:
        raise ValueError('Frozen test10 map was changed')
    textures = {}
    stage = Usd.Stage.Open(str(path))
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            value = attr.Get()
            if not isinstance(value, Sdf.AssetPath) or value.path == 'OmniPBR.mdl':
                continue
            resolved = Path(value.resolvedPath).resolve() if value.resolvedPath else None
            if (Path(value.path).is_absolute() or resolved is None or not resolved.is_file()
                    or not resolved.is_relative_to(project/'terrain_generator/textures')):
                raise ValueError(f'Nonportable or missing terrain texture: {value.path}')
            textures[resolved.relative_to(project).as_posix()] = package.sha256(resolved)
    if len(textures) != 10:
        raise ValueError(f'Expected 10 terrain images, found {len(textures)}')
    return textures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'dist/spot-test10-sac-cloud.tar.gz')
    parser.add_argument('--zed', type=Path,
                        default=Path('/home/isaac/isaac-sim/assets/Assets/Isaac/5.1/Isaac/Sensors/Stereolabs/ZED_X/ZED_X.usdc'))
    args = parser.parse_args()
    textures = terrain_textures(PROJECT)
    files = package.collect_files([EXPERIMENT], args.zed)
    # Deliver only this experiment and its installer; no old cloud launchers,
    # updates, machine settings or previously fitted scaler artifacts.
    cloud_files = {'setup.sh', 'run.sh', 'verify.py', 'requirements.txt', 'constraints.txt',
                   'package.py', 'portable_test9.py', 'package_test10.py', 'CLOUD_TEST10.md',
                   'setup_test10.sh', 'start_test10.sh', 'monitor_test10.sh', 'test_test10_cloud.py'}
    files = {n: p for n, p in files.items()
             if not n.startswith('spot_rl_energy_navigation/cloud/') or p.name in cloud_files}
    for name in list(files):
        if (Path(name).suffix in {'.lock', '.pending'}
                or ('artifacts' in Path(name).parts and Path(name).name.startswith('frozen_scalers.'))):
            del files[name]
    for name in textures:
        files[name] = PROJECT/name
    # The top-level guide gives only test10 commands; older guides remain available.
    files['LEGGIMI_TEST10.md'] = HERE/'CLOUD_TEST10.md'
    files['setup_test10.sh'] = HERE/'setup_test10.sh'
    files['start_test10.sh'] = HERE/'start_test10.sh'
    files['monitor_test10.sh'] = HERE/'monitor_test10.sh'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    required = sum(p.stat().st_size for p in files.values())*3 + 100*1024**2
    if min(shutil.disk_usage(tempfile.gettempdir()).free, shutil.disk_usage(args.output.parent).free) < required:
        raise OSError(f'Need approximately {required/1024**3:.1f} GiB free for staging/archive/verification')
    temporary_archive = args.output.with_suffix('.gz.tmp')
    try:
        with tempfile.TemporaryDirectory(prefix='spot-test10-bundle-') as directory:
            project = Path(directory)/TOP
            for name, source in files.items():
                target = project/name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            copy_camera(args.zed, project/'.cloud_assets/ZED_X.usdc')
            # Confirm assets remain resolvable after relocation, with no source-PC paths.
            # Add texture hashes without altering canonical USD bytes or identity.
            asset_manifest = project/'spot_rl_energy_navigation'/EXPERIMENT/'artifacts/manifest.json'
            data = json.loads(asset_manifest.read_text())
            data['portable_texture_sha256'] = textures
            asset_manifest.write_text(json.dumps(data, indent=2)+'\n')
            count = verify_usd_assets(project, EXPERIMENT, 'real_terrains_test10')
            verify_experiment_sources(project)
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
                          installation_on_cloud_tested=False,
                          real_scaler_calibration='automatic at first fresh training launch',
                          includes_only_test10=True)
            args.output.with_suffix('.verification.json').write_text(json.dumps(report, indent=2)+'\n')
            print(f'Ready: {args.output}\nSize: {args.output.stat().st_size/1024**2:.1f} MiB\nSHA256: {checksum}')
    finally:
        temporary_archive.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
