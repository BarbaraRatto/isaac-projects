"""USD portability helpers for the test9 bundle; run with Isaac Sim's python.sh.

Only asset paths are changed. The canonical portable map is used locally AND
in the archive, so cloud checkpoints retain exactly the local asset identity.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

EXPERIMENT = 'train_general_big_map_test9_SAC'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def make_map_portable(project):
    from pxr import Usd, Sdf
    project = Path(project).resolve()
    experiment = project/'spot_rl_energy_navigation'/EXPERIMENT
    path = experiment/'maps/real_terrains_test9.usd'
    manifest_path = experiment/'artifacts/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    before = digest(path)
    if manifest['identity']['terrain']['real_terrains_test9'] != before:
        raise ValueError('Map hash changed before portability conversion; refusing to bless unknown changes')
    stage = Usd.Stage.Open(str(path))
    edits, textures = [], {}
    for prim in stage.Traverse():
        for attr in prim.GetAttributes():
            value = attr.Get()
            if not isinstance(value, Sdf.AssetPath) or value.path == 'OmniPBR.mdl':
                continue
            source = Path(value.resolvedPath) if value.resolvedPath else (path.parent/value.path).resolve()
            if not source.is_file() or not source.is_relative_to(project/'terrain_generator/textures'):
                raise ValueError(f'Unexpected/missing map dependency: {value.path}')
            relative = os.path.relpath(source, path.parent)
            textures[source.relative_to(project).as_posix()] = digest(source)
            if value.path != relative:
                edits.append((attr, relative))
    if len(textures) != 10:
        raise ValueError(f'Expected 10 terrain texture images, found {len(textures)}')
    if edits:
        if any((experiment/'runs').glob('**/*.pt')):
            raise ValueError('Existing test9 checkpoints: do not change map identity during a run')
        for attr, relative in edits:
            attr.Set(Sdf.AssetPath(relative))
        stage.GetRootLayer().Save()
    after = digest(path)
    metadata_path = experiment/'artifacts/map_metadata.json'
    metadata = json.loads(metadata_path.read_text())
    metadata['usd_sha256'] = after
    write_json(metadata_path, metadata)
    identity = manifest['identity']
    identity['terrain']['real_terrains_test9'] = after
    identity['missions']['map_metadata.json'] = digest(metadata_path)
    manifest['pca_transfer']['target_identity'] = identity
    manifest['portable_texture_sha256'] = textures
    manifest.setdefault('portability', dict(source_map_sha256=before,
                        changes='Only texture asset paths changed to project-relative references; geometry/material values unchanged.'))
    write_json(manifest_path, manifest)
    report_path = experiment/'artifacts/map_validation.json'
    if report_path.is_file():
        report = json.loads(report_path.read_text())
        report.update(target_sha256=after, portable_texture_paths=True)
        write_json(report_path, report)
    return textures


def copy_camera(source, destination):
    """Copy exact camera geometry/optics plus its local MDL and PNG dependencies."""
    from pxr import Usd, Sdf
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    original = Usd.Stage.Open(str(source))
    stage = Usd.Stage.Open(str(destination))
    references = []
    for prim in original.Traverse():
        for attr in prim.GetAttributes():
            value = attr.Get()
            if not isinstance(value, Sdf.AssetPath):
                continue
            if not value.resolvedPath:
                raise ValueError(f'Unresolved camera dependency: {value.path}')
            mdl = Path(value.resolvedPath)
            if mdl.suffix != '.mdl':
                raise ValueError(f'Unsupported camera dependency: {mdl}')
            copied = destination.parent/'materials'/mdl.name
            copied.parent.mkdir(exist_ok=True)
            shutil.copy2(mdl, copied)
            references.append(copied)
            # This material imports OmniPBR from the simulator and 3 relative PNGs.
            for texture in re.findall(r'texture_2d\("([^"\n]+)"', mdl.read_text()):
                src, dst = (mdl.parent/texture).resolve(), (copied.parent/texture).resolve()
                if not src.is_relative_to(mdl.parent) or not dst.is_relative_to(copied.parent):
                    raise ValueError('Camera material dependency escapes its directory')
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                references.append(dst)
            stage.GetAttributeAtPath(attr.GetPath()).Set(Sdf.AssetPath(os.path.relpath(copied, destination.parent)))
    stage.GetRootLayer().Save()
    return references


def verify_usd_assets(project):
    """All non-runtime USD asset references must resolve inside the relocated bundle."""
    from pxr import Usd, Sdf
    project = Path(project).resolve()
    paths = [project/'spot_rl_energy_navigation'/EXPERIMENT/'maps/real_terrains_test9.usd',
             project/'IsaacRobotics/assets/spot.usd', project/'.cloud_assets/ZED_X.usdc']
    count = 0
    for path in paths:
        stage = Usd.Stage.Open(str(path))
        for layer in stage.GetUsedLayers():
            if not layer.anonymous and not Path(layer.realPath).resolve().is_relative_to(project):
                raise ValueError(f'External USD layer: {layer.realPath}')
        for prim in stage.Traverse():
            for attr in prim.GetAttributes():
                value = attr.Get()
                if not isinstance(value, Sdf.AssetPath) or value.path == 'OmniPBR.mdl':
                    continue
                if Path(value.path).is_absolute() or not value.resolvedPath:
                    raise ValueError(f'Nonportable/missing USD asset: {value.path}')
                resolved = Path(value.resolvedPath).resolve()
                if not resolved.is_relative_to(project) or not resolved.is_file():
                    raise ValueError(f'USD asset resolves outside bundle: {resolved}')
                count += 1
    for mdl in (project/'.cloud_assets').rglob('*.mdl'):
        for texture in re.findall(r'texture_2d\("([^"\n]+)"', mdl.read_text()):
            path = (mdl.parent/texture).resolve()
            if not path.is_relative_to(project) or not path.is_file():
                raise ValueError(f'Missing camera material texture: {texture}')
    return count
