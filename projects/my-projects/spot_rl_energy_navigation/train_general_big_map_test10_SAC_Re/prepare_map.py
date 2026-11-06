"""Build the fixed seeded 5x5 lab map, endpoint bank and runtime drag metadata offline."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from pxr import Usd, UsdGeom

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from train_general_big_map_test10_SAC_Re.config import MISSION_CFG, TERRAIN_NAMES, experiment_signature
from train_general_big_map_test10_SAC_Re.lab_geometry import materials, material_pair, flat_box


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2)+'\n')


def layout(seed):
    """Seeded sample, requiring both isolated green tiles and adjacent green tiles."""
    rng = np.random.default_rng(seed)
    for _ in range(10000):
        green = set(map(int, rng.choice(25, 8, replace=False)))
        degrees = []
        for tile in green:
            r,c = divmod(tile,5)
            degrees.append(sum(0<=r+dr<5 and 0<=c+dc<5 and (r+dr)*5+c+dc in green
                               for dr,dc in ((1,0),(-1,0),(0,1),(0,-1))))
        if 0 in degrees and max(degrees)>0:
            return [[TERRAIN_NAMES[int(r*5+c in green)] for c in range(5)] for r in range(5)]
    raise RuntimeError('No mixed isolated/adjacent layout found')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
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
        if manifest['map_seed'] != args.seed:
            raise ValueError('Map seed differs from frozen experiment')
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
    grid = layout(args.seed)
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
                    bounds_max=[15.,15.,0.],usd_sha256=sha(target),map_seed=args.seed,
                    points=len(rows),train_points=int((~evaluation).sum()),evaluation_points=int(evaluation.sum()),
                    local_checks='flat cells; endpoints at least 0.65 m from edges',reachability_checked=False)
    write(HERE/'artifacts/map_metadata.json',metadata)
    drag = dict(schema=2,materials=data,surface_regions={target.name:regions},maps_sha256={target.name:sha(target)},
                runtime_drag=dict(model='F_world_xy=-c*v_world_xy',selection='base XY inside green',
                                  body='Spot body, at COM',alignment=True,artificial_energy_addition=False))
    write(HERE/'artifacts/drag_manifest.json',drag)
    identity = dict(terrain={'lab_15x15':sha(target)},missions={name:sha(HERE/'artifacts'/name)
                         for name in ('spawn_bank.npz','map_metadata.json')})
    write(manifest_path,dict(signature=experiment_signature(),identity=identity,map_seed=args.seed,
                            materials_sha256=sha(HERE/'materials.json'),pca_status='pending_new_lab_calibration'))
    print('[MAP] Created 15x15 m, 17 grey + 8 green cells. Rows below: high Y first.')
    for row in reversed(grid):
        print(' '.join('G' if name==TERRAIN_NAMES[1] else '.' for name in row))


if __name__=='__main__':
    main()
