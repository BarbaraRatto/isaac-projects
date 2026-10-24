"""Validate the frozen map without modifying or regenerating it."""
from functools import lru_cache
from pxr import Usd, UsdGeom, UsdPhysics
from train_general_big_map_test1.layouts import map_metadata, TRAIN_LAYOUTS

@lru_cache(maxsize=1)
def validate_terrain():
    meta = map_metadata()
    stage = Usd.Stage.Open(str(TRAIN_LAYOUTS[0].usd))
    grid = stage.GetPrimAtPath('/World/Grid')
    if len(list(grid.GetChildren())) != 49:
        raise ValueError('Expected the frozen 7x7 terrain')
    for r, row in enumerate(meta['grid']):
        for c, name in enumerate(row):
            prim = stage.GetPrimAtPath(f'/World/Grid/Cell_{r}_{c}_{name}/terrain/mesh')
            if not prim or not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
                raise ValueError(f'Missing terrain collision mesh {r}, {c}')
    return meta
