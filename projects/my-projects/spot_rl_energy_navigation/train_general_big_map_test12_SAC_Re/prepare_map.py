"""Build the explicit user-defined 5x5 lab map, endpoint bank and runtime drag metadata offline."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from pxr import Usd, UsdGeom

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from train_general_big_map_test12_SAC_Re.config import MISSION_CFG, TERRAIN_NAMES, experiment_signature
from train_general_big_map_test12_SAC_Re.lab_geometry import materials, material_pair, flat_box


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2)+'\n')


def layout(seed=None):
    """User layout, top input row at high Y; internal rows start at low Y."""
    spec=json.loads((HERE/'layout_spec.json').read_text())
    rows=spec['rows_top_to_bottom']
    if (spec['orientation']!='first_row_is_top_high_y' or spec['cell_size_m']!=3.
            or len(rows)!=5 or any(len(row)!=5 for row in rows)
            or any(x not in ('g','v') for row in rows for x in row)
            or sum(x=='v' for row in rows for x in row)!=8):
        raise ValueError('Expected the requested 5x5 layout with eight green cells')
    names={'g':TERRAIN_NAMES[0],'v':TERRAIN_NAMES[1]}
    return [[names[x] for x in row] for row in reversed(rows)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse', action='store_true')
    args = parser.parse_args()
    target = HERE/'maps/lab_15x15.usda'
    manifest_path = HERE/'artifacts/manifest.json'
    if manifest_path.exists():
        if not args.reuse:
            raise FileExistsError('Frozen map already exists; do not regenerate during a run')
        manifest = json.loads(manifest_path.read_text())
        if manifest['materials_sha256'] != sha(HERE/'materials.json'):
            raise ValueError('Frozen materials changed; create a separate experiment')
        if manifest['layout_spec_sha256'] != sha(HERE/'layout_spec.json'):
            raise ValueError('Layout specification differs from the frozen experiment')
        for name,expected in manifest['identity']['missions'].items():
            if sha(HERE/'artifacts'/name) != expected:
                raise ValueError('Mission artifact changed: '+name)
        if sha(target) != manifest['identity']['terrain']['lab_15x15']:
            raise ValueError('Terrain changed')
        print('[MAP] Compatible frozen lab map already exists.')
        return
    if target.exists():
        raise FileExistsError('Partial map output exists; inspect before rebuilding')
    data = materials()
    grid = layout()
    stage = Usd.Stage.CreateNew(str(target))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, '/World').GetPrim())
    UsdGeom.Xform.Define(stage, '/World/Grid')
    pairs = {name: material_pair(stage, name, cfg) for name,cfg in data.items()}
    regions = []
    records = []
    rng = np.random.default_rng(927361)
    for r,row in enumerate(grid):
        for c,name in enumerate(row):
            material = 'grey' if name==TERRAIN_NAMES[0] else 'green'
            bounds = (3*c,3*r,3*(c+1),3*(r+1))
            flat_box(stage, f'Cell_{r}_{c}_{name}', bounds, pairs[material], data[material]['color_srgb'])
            regions.append(dict(material=material,bounds_xy_m=list(bounds)))
            xy = rng.uniform(np.array(bounds[:2])+MISSION_CFG.cell_margin_m,
                             np.array(bounds[2:])-MISSION_CFG.cell_margin_m, size=(100,2))
            for x,y in xy:
                records.append([x,y,0.,0.,TERRAIN_NAMES.index(name),r,c])
    stage.GetRootLayer().Save()
    rows = np.asarray(records, dtype=np.float64)
    evaluation = np.zeros(len(rows), dtype=bool)
    for tile in range(25):
        evaluation[np.arange(tile*100,(tile+1)*100,5)] = True
    np.savez_compressed(HERE/'artifacts/spawn_bank.npz', points=rows,evaluation=evaluation)
    metadata = dict(grid=grid,cell_width_m=3.,cell_height_m=3.,bounds_min=[0.,0.,-.1],
                    bounds_max=[15.,15.,0.],usd_sha256=sha(target),layout_source='layout_spec.json',
                    layout_spec_sha256=sha(HERE/'layout_spec.json'),
                    points=len(rows),train_points=int((~evaluation).sum()),evaluation_points=int(evaluation.sum()),
                    local_checks='flat cells; endpoints at least 0.65 m from edges',reachability_checked=False)
    write(HERE/'artifacts/map_metadata.json',metadata)
    drag = dict(schema=2,materials=data,surface_regions={target.name:regions},maps_sha256={target.name:sha(target)},
                runtime_drag=dict(model='F_world_xy=-c*v_world_xy',selection='base XY inside green',
                                  body='Spot body, at COM',alignment=True,artificial_energy_addition=False))
    write(HERE/'artifacts/drag_manifest.json',drag)
    from train_general_big_map_test12_SAC_Re.artifacts import source_identity
    identity=source_identity()
    write(manifest_path,dict(signature=experiment_signature(),identity=identity,layout_source='layout_spec.json',
                            layout_spec_sha256=sha(HERE/'layout_spec.json'),
                            materials_sha256=sha(HERE/'materials.json'),
                            pca_status='pending_new_lab_calibration',
                            pca_lifecycle='calibrate_on_first_training',
                            pca_source='artifacts/pca32_lab.npz',
                            terrain_files={'lab_15x15':'maps/lab_15x15.usda'},
                            runtime_input_files={name:sha(HERE/name) for name in
                                ('materials.json','layout_spec.json','artifacts/drag_manifest.json','surface_drag.py')}))
    print('[MAP] Created 15x15 m, 17 grey + 8 green cells. Rows below: high Y first.')
    for row in reversed(grid):
        print(' '.join('v' if name==TERRAIN_NAMES[1] else 'g' for name in row))


if __name__=='__main__':
    main()
