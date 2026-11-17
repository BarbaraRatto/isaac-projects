"""Rearrange frozen test13 terrain cells, without SimulationApp or regeneration.

Run using Isaac's USD Python:
  /home/isaac/isaaclab/2.3.0/_isaac_sim/python.sh \
    spot_rl_energy_navigation/multi_terrain_test13/create_test_maps.py
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

from pxr import Gf, Sdf, Usd, UsdGeom, Vt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from multi_terrain_test13.layouts import ROCKS, TEST_LAYOUTS, Condition
from multi_terrain_test13.terrain import read_terrain_codes


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(layout, source, overwrite=False):
    if layout.usd.exists() and not overwrite:
        raise FileExistsError(f'{layout.usd} already exists; use --overwrite deliberately')
    temporary = layout.usd.with_name(layout.usd.stem+'.building.usd')
    if temporary.exists():
        raise FileExistsError(temporary)
    donors = defaultdict(list)
    for cell in source.GetPrimAtPath('/World/Grid').GetChildren():
        donors[cell.GetName().split('_', 3)[3]].append(cell)
    target = Usd.Stage.CreateNew(str(temporary))
    UsdGeom.SetStageUpAxis(target, UsdGeom.GetStageUpAxis(source))
    UsdGeom.SetStageMetersPerUnit(target, UsdGeom.GetStageMetersPerUnit(source))
    if not Sdf.CopySpec(source.GetRootLayer(), '/World', target.GetRootLayer(), '/World'):
        raise RuntimeError('Could not copy frozen source World')
    target.SetDefaultPrim(target.GetPrimAtPath('/World'))
    for cell in list(target.GetPrimAtPath('/World/Grid').GetChildren()):
        target.RemovePrim(cell.GetPath())
    used = defaultdict(int)
    provenance = []
    for row, terrains in enumerate(layout.grid):
        for col, terrain in enumerate(terrains):
            donor = donors[terrain][used[terrain] % len(donors[terrain])]
            used[terrain] += 1
            sr, sc = map(int, donor.GetName().split('_')[1:3])
            src = donor.GetPath()
            dst = Sdf.Path(f'/World/Grid/Cell_{row}_{col}_{terrain}')
            if not Sdf.CopySpec(source.GetRootLayer(), src, target.GetRootLayer(), dst):
                raise RuntimeError(f'Could not copy {src}')
            for prim in Usd.PrimRange(target.GetPrimAtPath(dst)):
                for relation in prim.GetRelationships():
                    relation.SetTargets([p.ReplacePrefix(src, dst) if p.HasPrefix(src) else p
                                         for p in relation.GetTargets()])
                for attribute in prim.GetAttributes():
                    connections = attribute.GetConnections()
                    if connections:
                        attribute.SetConnections([p.ReplacePrefix(src, dst) if p.HasPrefix(src) else p
                                                  for p in connections])
                if prim.IsA(UsdGeom.Mesh):
                    mesh = UsdGeom.Mesh(prim)
                    # Frozen source points already contain the cell placement.
                    # Translation preserves every height, face and horizontal scale.
                    offset = Gf.Vec3f(2*(col-sc), 2*(row-sr), 0)
                    points = Vt.Vec3fArray([p+offset for p in mesh.GetPointsAttr().Get()])
                    mesh.GetPointsAttr().Set(points)
                    mesh.GetExtentAttr().Set(UsdGeom.PointBased.ComputeExtent(points))
            provenance.append(dict(row=row, col=col, terrain=terrain, donor=str(src)))
    target.GetRootLayer().Save()
    # Validate before making the new USD available to playback.
    from dataclasses import replace
    read_terrain_codes(Condition(0, replace(layout, usd=temporary)).case)
    for prim in target.Traverse():
        for relation in prim.GetRelationships():
            for path in relation.GetTargets():
                if path.IsAbsolutePath() and not target.GetPrimAtPath(path.GetPrimPath()):
                    raise RuntimeError(f'Broken relationship: {prim.GetPath()} -> {path}')
    temporary.replace(layout.usd)
    print(f'[USD] {layout.usd}', flush=True)
    return dict(name=layout.name, file=layout.usd.name, sha256=sha256(layout.usd),
                grid=layout.grid, cells=provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    for layout in TEST_LAYOUTS:
        if layout.usd.exists() and not args.overwrite:
            parser.error(f'{layout.usd} exists; use --overwrite to rebuild all three maps')
    source_hash = sha256(ROCKS.usd)
    source = Usd.Stage.Open(str(ROCKS.usd))
    records = [generate(layout, source, args.overwrite) for layout in TEST_LAYOUTS]
    manifest = dict(source=ROCKS.usd.name, source_sha256=source_hash,
                    cell_size_m=2, rows='first row y=0; then y=2 and y=4',
                    start_xy=[0, 0], goal_xy=[4, 4], maps=records)
    (HERE/'maps/new_maps_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
