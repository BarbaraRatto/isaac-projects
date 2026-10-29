"""Offline USD inspection and local spawn checks. No SimulationApp or reachability search.

Run with Isaac Sim's python.sh, not ordinary Python. Existing outputs are protected.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
from scipy.spatial import cKDTree
from pxr import Usd, UsdGeom

HERE = Path(__file__).resolve().parent

class Surface:
    def __init__(self, mesh, matrix):
        pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        # USD row-vector transform, including any authored parent transforms.
        transform = np.asarray(matrix)
        pts = np.column_stack((pts, np.ones(len(pts)))) @ transform
        pts = pts[:, :3]
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        if not np.all(counts == 3):
            raise ValueError('Expected triangle meshes')
        triangles = pts[np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)]
        a, b = triangles[:, 1, :2]-triangles[:, 0, :2], triangles[:, 2, :2]-triangles[:, 0, :2]
        area = a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0]
        self.triangles = triangles[(np.abs(area)>1e-12) & (triangles[:, :, 2].min(1)>-0.01)]
        if not len(self.triangles):
            raise ValueError('No terrain top faces')
        self.tree = cKDTree(self.triangles[:, :, :2].mean(1))

    def height(self, xy):
        # Nearest-centroid search is conservative: accept only explicit containing triangles.
        # Missing hits reject the candidate; this is not used for robot observations.
        k = min(128, len(self.triangles))
        _, ids = self.tree.query(xy, k=k)
        if k == 1: ids = ids[:, None]
        tri = self.triangles[ids]
        a, b = tri[:, :, 1, :2]-tri[:, :, 0, :2], tri[:, :, 2, :2]-tri[:, :, 0, :2]
        q = xy[:, None, :]-tri[:, :, 0, :2]
        det = a[..., 0]*b[..., 1]-a[..., 1]*b[..., 0]
        u = (q[..., 0]*b[..., 1]-q[..., 1]*b[..., 0])/det
        v = (a[..., 0]*q[..., 1]-a[..., 1]*q[..., 0])/det
        inside = (u>=-1e-7) & (v>=-1e-7) & (u+v<=1+1e-7)
        z = tri[:, :, 0, 2]+u*(tri[:, :, 1, 2]-tri[:, :, 0, 2])+v*(tri[:, :, 2, 2]-tri[:, :, 0, 2])
        return np.max(np.where(inside, z, -np.inf), axis=1)


def main():
    import sys
    sys.path.insert(0, str(HERE.parent))
    from train_general_big_map_test6_SAC.config import MISSION_CFG, TERRAIN_NAMES
    output = HERE/'artifacts/spawn_bank.npz'
    if output.exists():
        raise FileExistsError('Frozen spawn bank already exists; do not rebuild during a run')
    path = HERE/'maps/real_terrains_v2.usd'
    stage = Usd.Stage.Open(str(path))
    cache = UsdGeom.XformCache()
    boxes = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    grid = [['']*7 for _ in range(7)]
    records = []
    rng = np.random.default_rng(927361)
    # Conservative square footprint, sampled every 11 cm, for any initial yaw.
    v = np.linspace(-MISSION_CFG.footprint_radius_m, MISSION_CFG.footprint_radius_m, 11)
    ox, oy = np.meshgrid(v, v)
    footprint = np.column_stack((ox.ravel(), oy.ravel()))
    for cell in stage.GetPrimAtPath('/World/Grid').GetChildren():
        _, r, c, name = cell.GetName().split('_', 3)
        r, c = int(r), int(c); grid[r][c] = name
        if name in MISSION_CFG.spawn_excluded: continue
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(str(cell.GetPath())+'/terrain/mesh'))
        surface = Surface(mesh, cache.GetLocalToWorldTransform(mesh.GetPrim()))
        bbox = boxes.ComputeWorldBound(cell).ComputeAlignedBox()
        lo, hi = np.asarray(bbox.GetMin())[:2], np.asarray(bbox.GetMax())[:2]
        accepted = []
        for _ in range(12):
            xy = rng.uniform(lo+MISSION_CFG.cell_margin_m, hi-MISSION_CFG.cell_margin_m, size=(30, 2))
            heights = surface.height((xy[:, None]+footprint).reshape(-1, 2)).reshape(30, -1)
            valid = np.isfinite(heights).all(1) & ((heights.max(1)-heights.min(1))<=MISSION_CFG.max_spawn_relief_m)
            for q, z in zip(xy[valid], heights[valid]):
                accepted.append([*q, z.min(), z.max(), TERRAIN_NAMES.index(name), r, c])
            if len(accepted)>=100: break
        if len(accepted)<20:
            raise RuntimeError(f'Too few local safe points in {cell.GetName()}: {len(accepted)}')
        records.extend(accepted[:100])
        print(cell.GetName(), 'valid points', min(len(accepted),100), flush=True)
    rows = np.asarray(records, dtype=np.float64)
    # Disjoint spatial candidates for train/evaluation (no claim of held-out map generalization).
    evaluation = np.zeros(len(rows), dtype=bool)
    for r in range(7):
        for c in range(7):
            ids = np.flatnonzero((rows[:, 5]==r)&(rows[:, 6]==c))
            evaluation[ids[::5]] = True
    bounds = boxes.ComputeWorldBound(stage.GetPrimAtPath('/World/Grid')).ComputeAlignedBox()
    meta = dict(grid=grid, cell_width_m=6.0, cell_height_m=3.0,
                bounds_min=list(bounds.GetMin()), bounds_max=list(bounds.GetMax()),
                usd_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                points=len(rows), train_points=int((~evaluation).sum()), evaluation_points=int(evaluation.sum()),
                local_checks='11x11 footprint samples, 0.55 m half-width, relief <= 0.16 m; not a traversal guarantee',
                reachability_checked=False)
    np.savez_compressed(output, points=rows, evaluation=evaluation)
    (HERE/'artifacts/map_metadata.json').write_text(json.dumps(meta, indent=2)+'\n')

if __name__ == '__main__': main()
