"""Offline geometry, binding and integration checks; no simulator or GPU."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import pytest
from pxr import Usd,UsdGeom,UsdPhysics,UsdShade

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from create_terrains import build,materials
from energy_metrics import TraversalMeter,write_report
from surface_drag import SurfaceDrag


def test_drag_opposes_world_motion_and_clears_when_leaving_green():
    import torch
    from types import SimpleNamespace
    calls=[]
    robot=SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.tensor([[0.,1.,.5]]),
        root_lin_vel_w=torch.tensor([[.9,-.2,.3]])),
        set_external_force_and_torque=lambda **kw:calls.append(kw))
    drag=SurfaceDrag(HERE/'maps/manifest.json',HERE/'maps/comparison.usda')
    force=drag.apply(robot)
    assert force.tolist()[0]==pytest.approx([-81.,18.,0.])
    assert float((force*robot.data.root_lin_vel_w).sum())<0
    assert calls[-1]['is_global'] is True and calls[-1]['body_ids']==[0]
    assert not bool(calls[-1]['torques'].any())
    robot.data.root_lin_vel_w.zero_()
    assert not bool(drag.apply(robot).any())
    robot.data.root_lin_vel_w[:]=torch.tensor([[-.9,0.,0.]])
    assert drag.apply(robot)[0,0]==pytest.approx(81.)
    robot.data.root_pos_w[:,1]=-1.
    assert not bool(drag.apply(robot).any())
    assert not bool(calls[-1]['forces'].any())
    robot.data.root_pos_w[:,0]=12.
    robot.data.root_pos_w[:,1]=1.
    assert not bool(drag.apply(robot).any())


def test_drag_respects_translated_energy_boards_and_grey_control():
    import torch
    from types import SimpleNamespace
    robot=SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.tensor([[0.,5.,.5]]),
        root_lin_vel_w=torch.tensor([[.9,0.,0.]])),
        set_external_force_and_torque=lambda **kw:None)
    green=SurfaceDrag(HERE/'maps/manifest.json',HERE/'maps/green_flat.usda',offset_xy=(0.,5.))
    grey=SurfaceDrag(HERE/'maps/manifest.json',HERE/'maps/grey_flat.usda',offset_xy=(0.,5.))
    assert green.apply(robot)[0,0]==pytest.approx(-81.)
    assert not bool(grey.apply(robot).any())
    robot.data.root_pos_w[:,1]=-5.
    assert not bool(green.apply(robot).any())


def test_authored_maps_have_exact_flat_collision_and_material_bindings():
    manifest=json.loads((HERE/'maps/manifest.json').read_text())
    expected={'grey_tile.usda':1,'green_tile.usda':1,'grey_flat.usda':1,'green_flat.usda':1,'comparison.usda':2}
    for name,count in expected.items():
        path=HERE/'maps'/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['maps_sha256'][name]
        stage=Usd.Stage.Open(str(path));assert stage.GetDefaultPrim().GetPath().pathString=='/World'
        assert UsdGeom.GetStageUpAxis(stage)=='Z' and UsdGeom.GetStageMetersPerUnit(stage)==1.
        meshes=[p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
        assert len(meshes)==count
        for prim in meshes:
            mesh=UsdGeom.Mesh(prim);points=list(mesh.GetPointsAttr().Get())
            assert sorted({round(float(v[2]),5) for v in points})==[-.1,0.]
            assert UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
            material=prim.GetName().split('_')[0];cfg=manifest['materials'][material]
            binding=UsdShade.MaterialBindingAPI(prim)
            visual,_=binding.ComputeBoundMaterial()
            physical,_=binding.ComputeBoundMaterial(materialPurpose='physics')
            assert str(visual.GetPath())=='/World/Looks/'+material
            api=UsdPhysics.MaterialAPI(physical.GetPrim())
            assert api.GetStaticFrictionAttr().Get()==pytest.approx(cfg['static_friction'])
            assert api.GetDynamicFrictionAttr().Get()==pytest.approx(cfg['dynamic_friction'])
            assert api.GetRestitutionAttr().Get()==0
            for key,field in (('compliantContactStiffness','compliant_contact_stiffness'),
                              ('compliantContactDamping','compliant_contact_damping')):
                assert physical.GetPrim().GetAttribute('physxMaterial:'+key).Get()==pytest.approx(cfg.get(field,0.))
            assert physical.GetPrim().GetAttribute('physxMaterial:compliantContactAccelerationSpring').Get() is False
            assert physical.GetPrim().GetAttribute('physxMaterial:frictionCombineMode').Get()=='average'
            assert 'PhysxMaterialAPI' in physical.GetPrim().GetMetadata('apiSchemas').GetAppliedItems()
            assert not UsdPhysics.RigidBodyAPI(prim)
            # Top triangles face up; they are coplanar at z=0.
            indices=mesh.GetFaceVertexIndicesAttr().Get()
            for a,b,c in (indices[:3],indices[3:6]):
                p,q,r=points[a],points[b],points[c]
                assert p[2]==q[2]==r[2]==0
                assert (q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0])>0
        assert all(not str(attr.Get()).startswith('/home/') for p in stage.Traverse() for attr in p.GetAttributes())


def test_constant_power_gate_excludes_standing_approach_and_finish():
    meter=TraversalMeter(1.,7.)
    meter.update((0,0),(0,0),100,100,10)
    for i in range(100):meter.update((i/10,0),((i+1)/10,0),100,100,.1)
    assert meter.completed
    assert meter.energy_j==pytest.approx(600)
    assert meter.seconds==pytest.approx(6)
    assert meter.path_m==pytest.approx(6)
    assert meter.forward_m==pytest.approx(6)


def test_fractional_gates_integrate_linearly_varying_power():
    meter=TraversalMeter(1.,3.)
    meter.update((0,0),(4,0),10,30,2)
    assert meter.completed and meter.energy_j==pytest.approx(20)
    assert meter.seconds==1 and meter.forward_m==2


def test_backtracking_counts_energy_and_extra_path_after_start():
    meter=TraversalMeter(1.,3.)
    meter.update((0,0),(2,0),10,10,2)
    meter.update((2,0),(.5,0),10,10,1.5)
    meter.update((.5,0),(3,0),10,10,2.5)
    assert meter.completed and meter.energy_j==pytest.approx(50)
    assert meter.forward_m==pytest.approx(2) and meter.path_m==pytest.approx(5)


def test_failed_trials_are_excluded_and_energy_difference_is_paired(tmp_path):
    records=[dict(material='grey',repeat=1,command_speed_m_s=.9,reason='completed',j_per_forward_m=100,mean_forward_speed_m_s=.89),
             dict(material='green',repeat=1,command_speed_m_s=.9,reason='completed',j_per_forward_m=130,mean_forward_speed_m_s=.86),
             dict(material='grey',repeat=2,command_speed_m_s=.9,reason='completed',j_per_forward_m=90,mean_forward_speed_m_s=.9),
             dict(material='green',repeat=2,command_speed_m_s=.9,reason='persistent_tilt',j_per_forward_m='',mean_forward_speed_m_s='')]
    with (tmp_path/'trials.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=records[0]);writer.writeheader();writer.writerows(records)
    result=write_report(tmp_path)
    pair=result['paired_comparisons'][0]
    assert pair['valid_pairs']==1 and pair['mean_green_minus_grey_j_per_m']==30
    assert pair['mean_green_percent_increase']==pytest.approx(30)
    green=next(g for g in result['groups'] if g['material']=='green')
    assert green['completed']==1 and green['attempts']==2


def test_comparison_surface_rectangles_are_adjacent_without_gaps():
    stage=Usd.Stage.Open(str(HERE/'maps/comparison.usda'))
    grey=UsdGeom.Mesh(stage.GetPrimAtPath('/World/Terrain/grey_0')).GetExtentAttr().Get()
    green=UsdGeom.Mesh(stage.GetPrimAtPath('/World/Terrain/green_1')).GetExtentAttr().Get()
    assert list(grey[0])==pytest.approx([-2,-3,-.1])
    assert list(grey[1])==pytest.approx([10,0,0])
    assert list(green[0])==pytest.approx([-2,0,-.1])
    assert list(green[1])==pytest.approx([10,3,0])


def test_runtime_terrain_placement_uses_single_translation_operation():
    from measure_energy import load_test_areas
    from pxr import Gf
    stage=Usd.Stage.CreateInMemory()
    calls=[]
    def create_prim(path,kind,usd_path,translation):
        prim=stage.DefinePrim(path,kind)
        prim.GetReferences().AddReference(usd_path)
        # Isaac Sim create_prim -> XFormPrim already authors translate/orient/scale.
        UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(*translation))
        calls.append(path)
        return prim
    centres=load_test_areas(create_prim)
    assert len(calls)==2
    for material,y in centres.items():
        prim=stage.GetPrimAtPath('/World/Lab/'+material)
        ops=UsdGeom.Xformable(prim).GetOrderedXformOps()
        assert len(ops)==1 and ops[0].GetOpType()==UsdGeom.XformOp.TypeTranslate
        assert list(ops[0].Get())==pytest.approx([0,y,0])
        bounds=UsdGeom.BBoxCache(0,['default']).ComputeWorldBound(prim).ComputeAlignedRange()
        assert bounds.GetMin()[1]==pytest.approx(y-3)
        assert bounds.GetMax()[1]==pytest.approx(y+3)
