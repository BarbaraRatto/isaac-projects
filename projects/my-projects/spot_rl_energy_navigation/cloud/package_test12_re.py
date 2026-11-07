#!/usr/bin/env python3
"""Create ONE complete test12 SAC Re ZIP, with portable robot/camera/DINO and flat lab map.

Use Isaac Sim python.sh (pxr required), without launching SimulationApp.
No previous runs, model checkpoints, private environment files or credentials.
"""
import argparse
import ast
import json
from pathlib import Path
import shutil
import zipfile
import tempfile

import package
from portable_test9 import copy_camera, verify_usd_assets

EXPERIMENT = "train_general_big_map_test12_SAC_Re"
from verify import verify_files, verify_bundle

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
TOP = 'spot-test12-sac-re-cloud'


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
    if list((root/'artifacts').glob('pca*.npz')):
        raise ValueError('Fresh bundle must calibrate its own lab PCA, not ship fitted values')



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=HERE/'dist/spot-test12-sac-re-cloud.zip')
    parser.add_argument('--zed', type=Path,
                        default=Path('/home/isaac/isaac-sim/assets/Assets/Isaac/5.1/Isaac/Sensors/Stereolabs/ZED_X/ZED_X.usdc'))
    args = parser.parse_args()
    files = package.collect_files([EXPERIMENT], args.zed)
    # Deliver only this experiment and its installer; no old cloud launchers,
    # updates, machine settings or previously fitted scaler artifacts.
    cloud_files = {'setup.sh', 'run.sh', 'verify.py', 'requirements.txt', 'constraints.txt',
                   'package.py', 'portable_test9.py', 'package_test12_re.py', 'CLOUD_TEST12_RE.md',
                   'setup_test12_re.sh', 'start_test12_re.sh', 'monitor_test12_re.sh', 'test_test12_re_cloud.py', 'test_test12_re_inputs.py', 'COMANDI_TEST12_RE.md'}
    files = {n: p for n, p in files.items()
             if not n.startswith('spot_rl_energy_navigation/cloud/') or p.name in cloud_files}
    for name in list(files):
        if (Path(name).suffix in {'.lock', '.pending'}
                or ('artifacts' in Path(name).parts and Path(name).name.startswith('frozen_scalers.'))):
            del files[name]
    # The top-level guide gives only test12_re commands; older guides remain available.
    files['LEGGIMI_TEST12_RE.md'] = HERE/'COMANDI_TEST12_RE.md'
    files['setup_test12_re.sh'] = HERE/'setup_test12_re.sh'
    files['start_test12_re.sh'] = HERE/'start_test12_re.sh'
    files['monitor_test12_re.sh'] = HERE/'monitor_test12_re.sh'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    required = sum(p.stat().st_size for p in files.values())*3 + 100*1024**2
    if min(shutil.disk_usage(tempfile.gettempdir()).free, shutil.disk_usage(args.output.parent).free) < required:
        raise OSError(f'Need approximately {required/1024**3:.1f} GiB free for staging/archive/verification')
    temporary_archive = args.output.with_suffix('.zip.tmp')
    try:
        with tempfile.TemporaryDirectory(prefix='spot-test12_re-bundle-') as directory:
            project = Path(directory)/TOP
            for name, source in files.items():
                target = project/name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            copy_camera(args.zed, project/'.cloud_assets/ZED_X.usdc')
            # Confirm assets remain resolvable after relocation, with no source-PC paths.
            count = verify_usd_assets(project, EXPERIMENT, 'lab_15x15', '.usda')
            verify_experiment_sources(project)
            verify_files(project, zed=project/'.cloud_assets/ZED_X.usdc', experiment=EXPERIMENT)
            manifest = dict(schema=1, primary_experiment=EXPERIMENT, experiments=[EXPERIMENT],
                            files={p.relative_to(project).as_posix(): package.sha256(p)
                                   for p in sorted(project.rglob('*')) if p.is_file()})
            manifest_path = project/'spot_rl_energy_navigation/cloud/bundle-manifest.json'
            manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
            verify_bundle(project)
            with zipfile.ZipFile(temporary_archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
                for path in sorted(project.rglob('*')):
                    if path.is_file():
                        archive.write(path, arcname=TOP+'/'+path.relative_to(project).as_posix())
            # Re-read every payload and check content hashes and the exact inventory.
            import hashlib
            with zipfile.ZipFile(temporary_archive) as archive:
                names=set()
                for member in archive.infolist():
                    relative=Path(member.filename).relative_to(TOP).as_posix()
                    if relative in names or '..' in Path(relative).parts:
                        raise ValueError('Unsafe or duplicate archive entry')
                    names.add(relative)
                    if relative in manifest['files']:
                        digest=hashlib.sha256()
                        with archive.open(member) as stream:
                            for block in iter(lambda:stream.read(1024*1024),b''):
                                digest.update(block)
                        if digest.hexdigest()!=manifest['files'][relative]:
                            raise ValueError('Archive payload mismatch: '+relative)
                if names != set(manifest['files']) | {'spot_rl_energy_navigation/cloud/bundle-manifest.json'}:
                    raise ValueError('Archive file inventory differs from manifest')
            temporary_archive.replace(args.output)
            checksum = package.sha256(args.output)
            args.output.with_suffix(args.output.suffix+'.sha256').write_text(checksum+'  '+args.output.name+'\n')
            report = dict(archive=args.output.name, sha256=checksum, bytes=args.output.stat().st_size,
                          files=len(manifest['files']), relocated_usd_asset_references=count,
                          primary_experiment=EXPERIMENT, contains_terrain_textures=False,
                          package_integrity_verified=True, simulation_launched=False,
                          installation_on_cloud_tested=False,
                          pca_calibration='automatic at first fresh training launch',
                          includes_only_test12_re=True)
            args.output.with_suffix('.verification.json').write_text(json.dumps(report, indent=2)+'\n')
            print(f'Ready: {args.output}\nSize: {args.output.stat().st_size/1024**2:.1f} MiB\nSHA256: {checksum}')
    finally:
        temporary_archive.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
