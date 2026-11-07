"""Validate the frozen lab map and the runtime resistance configuration."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
from train_general_big_map_test11_SAC_Re.layouts import map_metadata, TRAIN_LAYOUTS


@lru_cache(maxsize=1)
def validate_terrain():
    meta = map_metadata()
    path = TRAIN_LAYOUTS[0].usd
    root = Path(__file__).resolve().parent
    spec_hash=hashlib.sha256((root/'layout_spec.json').read_bytes()).hexdigest()
    if meta['layout_spec_sha256']!=spec_hash:
        raise ValueError('Layout specification changed after map creation')
    drag = json.loads((root/'artifacts/drag_manifest.json').read_text())
    if drag['maps_sha256'][path.name] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError('Terrain and drag manifest differ')
    if drag['materials'] != json.loads((root/'materials.json').read_text()):
        raise ValueError('Material settings changed after map creation')
    if sum(name=='lab_green' for row in meta['grid'] for name in row)!=8:
        raise ValueError('Expected exactly 8 green cells')
    stage = Usd.Stage.Open(str(path))
    if len(list(stage.GetPrimAtPath('/World/Grid').GetChildren())) != 25:
        raise ValueError('Expected the frozen 5x5 lab map')
    for r,row in enumerate(meta['grid']):
        for c,name in enumerate(row):
            prim=stage.GetPrimAtPath(f'/World/Grid/Cell_{r}_{c}_{name}/terrain/mesh')
            if not prim or not prim.IsA(UsdGeom.Mesh) or not prim.HasAPI(UsdPhysics.CollisionAPI):
                raise ValueError(f'Missing terrain collision mesh {r}, {c}')
            physical,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
            api=UsdPhysics.MaterialAPI(physical.GetPrim())
            if abs(api.GetStaticFrictionAttr().Get()-1.)>1e-6 or abs(api.GetDynamicFrictionAttr().Get()-.8)>1e-6:
                raise ValueError('Both lab materials must use asphalt friction')
    return meta
