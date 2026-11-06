"""Create flat lab materials and small comparison boards offline, without Isaac Sim."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

HERE = Path(__file__).resolve().parent
TRIANGLES = [(4,5,6),(4,6,7),(0,2,1),(0,3,2),(0,1,5),(0,5,4),
             (1,2,6),(1,6,5),(2,3,7),(2,7,6),(3,0,4),(3,4,7)]


def materials():
    data = json.loads((HERE/'materials.json').read_text())
    if set(data) != {'grey', 'green'}:
        raise ValueError('Expected exactly grey and green materials')
    for name, cfg in data.items():
        if (not all(math.isfinite(cfg[k]) for k in ('dynamic_friction','static_friction','restitution','roughness'))
                or not 0 <= cfg['dynamic_friction'] <= cfg['static_friction']
                or not 0<=cfg['restitution']<=1 or not 0<=cfg['roughness']<=1):
            raise ValueError('Invalid friction: '+name)
        if len(cfg['color_srgb']) != 3 or not all(0 <= x <= 1 for x in cfg['color_srgb']):
            raise ValueError('Invalid RGB: '+name)
        stiffness = cfg.get('compliant_contact_stiffness', 0.)
        damping = cfg.get('compliant_contact_damping', 0.)
        if not all(math.isfinite(x) and x >= 0 for x in (stiffness, damping)):
            raise ValueError('Invalid contact stiffness/damping: '+name)
        if (damping > 0 and stiffness == 0) or (stiffness > 0 and cfg['restitution'] != 0):
            raise ValueError('Compliant contact requires positive stiffness and zero restitution: '+name)
        drag = cfg.get('horizontal_drag_ns_per_m', 0.)
        if not math.isfinite(drag) or drag < 0:
            raise ValueError('Invalid horizontal drag: '+name)
    return data


def linear_color(rgb):
    return Gf.Vec3f(*[x/12.92 if x <= .04045 else ((x+.055)/1.055)**2.4 for x in rgb])


def material_pair(stage, name, cfg):
    visual = UsdShade.Material.Define(stage, '/World/Looks/'+name)
    shader = UsdShade.Shader.Define(stage, '/World/Looks/'+name+'/Surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(linear_color(cfg['color_srgb']))
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(cfg['roughness'])
    shader.CreateInput('metallic', Sdf.ValueTypeNames.Float).Set(0.)
    visual.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    physics = UsdShade.Material.Define(stage, '/World/PhysicsMaterials/'+name)
    api = UsdPhysics.MaterialAPI.Apply(physics.GetPrim())
    api.CreateStaticFrictionAttr(cfg['static_friction'])
    api.CreateDynamicFrictionAttr(cfg['dynamic_friction'])
    api.CreateRestitutionAttr(cfg['restitution'])
    # Author PhysX schema tokens without importing/starting Omni extensions.
    physics.GetPrim().AddAppliedSchema('PhysxMaterialAPI')
    for key in ('frictionCombineMode', 'restitutionCombineMode'):
        physics.GetPrim().CreateAttribute('physxMaterial:'+key, Sdf.ValueTypeNames.Token).Set('average')
    # Force-based spring (N/m) and damper (N*s/m); zero stiffness keeps grey rigid.
    for key, field in (('compliantContactStiffness', 'compliant_contact_stiffness'),
                       ('compliantContactDamping', 'compliant_contact_damping')):
        physics.GetPrim().CreateAttribute('physxMaterial:'+key, Sdf.ValueTypeNames.Float).Set(cfg.get(field, 0.))
    physics.GetPrim().CreateAttribute('physxMaterial:compliantContactAccelerationSpring',
                                     Sdf.ValueTypeNames.Bool).Set(False)
    return visual, physics


def flat_box(stage, name, bounds, pair, color):
    x0,y0,x1,y1 = bounds
    points = [(x0,y0,-.1),(x1,y0,-.1),(x1,y1,-.1),(x0,y1,-.1),
              (x0,y0,0),(x1,y0,0),(x1,y1,0),(x0,y1,0)]
    mesh = UsdGeom.Mesh.Define(stage, '/World/Terrain/'+name)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in points]))
    mesh.CreateFaceVertexCountsAttr([3]*len(TRIANGLES))
    mesh.CreateFaceVertexIndicesAttr([i for face in TRIANGLES for i in face])
    mesh.CreateSubdivisionSchemeAttr('none')
    mesh.CreateExtentAttr(UsdGeom.PointBased.ComputeExtent(mesh.GetPointsAttr().Get()))
    mesh.CreateDisplayColorAttr([linear_color(color)])
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('none')
    binding = UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
    binding.Bind(pair[0])
    binding.Bind(pair[1], materialPurpose='physics')


def build(output, surfaces, data):
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    world = UsdGeom.Xform.Define(stage, '/World')
    stage.SetDefaultPrim(world.GetPrim())
    pairs = {name:material_pair(stage,name,cfg) for name,cfg in data.items()}
    for index, (material, bounds) in enumerate(surfaces):
        flat_box(stage, f'{material}_{index}', bounds, pairs[material], data[material]['color_srgb'])
    stage.GetRootLayer().Save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=HERE/'maps')
    args = parser.parse_args()
    data = materials()
    specs = {
        'grey_tile.usda': [('grey',(-1.5,-1.5,1.5,1.5))],
        'green_tile.usda': [('green',(-1.5,-1.5,1.5,1.5))],
        'grey_flat.usda': [('grey',(-2,-3,10,3))],
        'green_flat.usda': [('green',(-2,-3,10,3))],
        'comparison.usda': [('grey',(-2,-3,10,0)),('green',(-2,0,10,3))],
    }
    hashes = {}
    for name, surfaces in specs.items():
        path = args.output_dir/name
        # Regeneration is explicit and deterministic; previous experiments untouched.
        if path.exists():path.unlink()
        build(path, surfaces, data)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        print('[TERRAIN] '+str(path))
    manifest = dict(schema=2,materials=data,friction_combine_mode='average',
                    surface_regions={name:[dict(material=material,bounds_xy_m=list(bounds))
                                           for material,bounds in surfaces] for name,surfaces in specs.items()},
                    runtime_drag=dict(model='F_world_xy = -c * base_velocity_world_xy',
                                      region_selection='base XY inside tile, not individual foot contacts',
                                      applied_to='root body center of mass; zero vertical force and zero torque',
                                      requires_runtime_script=True),
                    unit_cell_m=[3,3],comparison_size_m=[12,6],surface_top_z_m=0.,
                    future_training_map=dict(size_m=[15,15],rows=5,cols=5,green_cells=8,grey_cells=17,
                                             layout='not generated yet; mix isolated and adjacent green cells'),
                    manual_spawns=dict(grey=[0,0,.75],green=[0,0,.75],comparison=[0,-1.5,.75]),
                    maps_sha256=hashes,simulation_launched=False)
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':main()
