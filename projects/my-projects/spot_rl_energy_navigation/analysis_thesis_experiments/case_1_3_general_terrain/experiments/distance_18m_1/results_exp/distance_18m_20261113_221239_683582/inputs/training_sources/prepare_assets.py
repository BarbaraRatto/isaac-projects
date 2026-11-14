"""Build test9 once, offline with Isaac Sim python.sh (no SimulationApp).

Copy existing cells, preserving their meshes/materials. Never regenerate noise.
The source experiment is needed only to rebuild, not to train or evaluate.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from train_general_big_map_test9_SAC.config import experiment_signature
from train_general_big_map_test9_SAC.artifacts import source_identity
from train_general_big_map_test9_SAC.calibration import pca_digest
from train_general_big_map_test9_SAC.perception import FrozenPCA, file_sha256

# User-facing coordinates: row 1 at bottom, columns increase along +X.
SWAPS = [((2, 4), (1, 1)), ((6, 2), (4, 6)),
         ((7, 6), (5, 5)), ((4, 6), (3, 1))]
SOURCE_SHA256 = '3656ff462e51ef6935e3d572834965a39bdf9345f9ecdfc1af509a665903925c'


def build_map(source, output):
    if file_sha256(source) != SOURCE_SHA256:
        raise ValueError('Source map differs from the agreed test7 snapshot')
    original = Usd.Stage.Open(str(source))
    cells = {(int(p.GetName().split('_')[1])+1, int(p.GetName().split('_')[2])+1): p
             for p in original.GetPrimAtPath('/World/Grid').GetChildren()}
    permutation = {key: key for key in cells}
    for a, b in SWAPS:
        permutation[a], permutation[b] = permutation[b], permutation[a]
    original.GetRootLayer().Export(str(output))
    result = Usd.Stage.Open(str(output))
    for cell in list(result.GetPrimAtPath('/World/Grid').GetChildren()):
        result.RemovePrim(cell.GetPath())
    mapping = {}
    for destination, source_position in sorted(permutation.items()):
        prim = cells[source_position]
        r, c = destination
        terrain = prim.GetName().split('_', 3)[3]
        target = Sdf.Path(f'/World/Grid/Cell_{r-1}_{c-1}_{terrain}')
        Sdf.CopySpec(original.GetRootLayer(), prim.GetPath(), result.GetRootLayer(), target)
        mapping[prim.GetPath()] = target
        if destination != source_position:
            moved = result.GetPrimAtPath(target)
            moved.SetTypeName('Xform')
            delta = Gf.Vec3d((c-source_position[1])*6., (r-source_position[0])*3., 0.)
            UsdGeom.Xformable(moved).AddTranslateOp().Set(delta)
    # CopySpec does not relocate references to sibling prims/materials.
    def relocate(path):
        for old, new in mapping.items():
            if path.HasPrefix(old):
                return path.ReplacePrefix(old, new)
        return path
    for prim in result.Traverse():
        for rel in prim.GetRelationships():
            targets = rel.GetTargets()
            if targets:
                rel.SetTargets([relocate(t) for t in targets])
        for attr in prim.GetAttributes():
            connections = attr.GetConnections()
            if connections:
                attr.SetConnections([relocate(t) for t in connections])
    # Verify world geometry is the original translated rigidly, with identical topology.
    ca, cb = UsdGeom.XformCache(), UsdGeom.XformCache()
    for old, new in mapping.items():
        a = UsdGeom.Mesh(original.GetPrimAtPath(old.AppendPath('terrain/mesh')))
        b = UsdGeom.Mesh(result.GetPrimAtPath(new.AppendPath('terrain/mesh')))
        for getter in ('GetPointsAttr', 'GetFaceVertexIndicesAttr', 'GetFaceVertexCountsAttr'):
            np.testing.assert_array_equal(getattr(a, getter)().Get(), getattr(b, getter)().Get())
        old_rc = tuple(int(v) for v in old.name.split('_')[1:3])
        new_rc = tuple(int(v) for v in new.name.split('_')[1:3])
        delta = np.array([(new_rc[1]-old_rc[1])*6., (new_rc[0]-old_rc[0])*3., 0.])
        points = np.column_stack((np.asarray(a.GetPointsAttr().Get()), np.ones(len(a.GetPointsAttr().Get()))))
        world_a = points @ np.asarray(ca.GetLocalToWorldTransform(a.GetPrim()))
        world_b = points @ np.asarray(cb.GetLocalToWorldTransform(b.GetPrim()))
        np.testing.assert_allclose(world_b[:, :3], world_a[:, :3]+delta, atol=1e-6)
    result.GetRootLayer().Save()
    return {str(old): str(new) for old, new in mapping.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=HERE.parent/'train_general_big_map_test7_SAC')
    args = parser.parse_args()
    output = HERE/'maps/real_terrains_test9.usd'
    if output.exists() or (HERE/'artifacts/spawn_bank.npz').exists():
        raise FileExistsError('Frozen test9 outputs already exist; do not rebuild during training')
    mapping = build_map(args.source/'maps/real_terrains_v2.usd', output)
    from train_general_big_map_test9_SAC.prepare_map import main as prepare_bank
    prepare_bank()
    meta = json.loads((HERE/'artifacts/map_metadata.json').read_text())
    ice = [(r, c) for r, row in enumerate(meta['grid']) for c, name in enumerate(row) if name == 't2_slippery']
    assert len(ice) == 7
    assert all(max(abs(r-a), abs(c-b)) > 1 for i, (r, c) in enumerate(ice) for a, b in ice[i+1:])
    for name in ('pca32_big_map.npz', 'pca32_big_map.json', 'pca32_big_map.collection.json'):
        shutil.copy2(args.source/'artifacts'/name, HERE/'artifacts'/name)
    pca_path = HERE/'artifacts/pca32_big_map.npz'
    pca = FrozenPCA.load(pca_path)
    identity = source_identity()
    manifest = dict(identity=identity, signature=experiment_signature(), pca_status='transferred_frozen',
                    pca_sha256=file_sha256(pca_path), pca_source='artifacts/pca32_big_map.npz',
                    map_source_sha256=SOURCE_SHA256, cell_mapping=mapping, swaps_one_based=SWAPS,
                    pca_transfer=dict(calibration_identity=pca.metadata['identity'],
                                      target_identity=identity, array_sha256=pca_digest(pca),
                                      source='train_general_big_map_test7_SAC/artifacts/pca32_big_map.npz',
                                      reason='Same seven terrain geometries/materials, only translated/permuted. '
                                             'PCA metadata and validation metrics describe the ORIGINAL calibration map.'))
    (HERE/'artifacts/manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    from cloud.portable_test9 import make_map_portable
    make_map_portable(HERE.parents[1])
    print('Test9 map, mission bank and explicit frozen-PCA transfer ready; no simulation was launched.')


if __name__ == '__main__':
    main()
