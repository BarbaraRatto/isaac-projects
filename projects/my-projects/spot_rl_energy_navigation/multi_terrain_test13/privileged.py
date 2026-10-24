"""Training-only contact features; material coefficients read from the composed USD."""
import numpy as np
import torch

from multi_terrain_test13.config import FOOT_NAMES, CONTACT_THRESHOLD_N


def combine_friction(a, b):
    """PhysX combine precedence: average < min < multiply < max."""
    modes = ('average', 'min', 'multiply', 'max')
    mode = modes[max(modes.index(a[2]), modes.index(b[2]))]
    x, y = np.asarray(a[:2]), np.asarray(b[:2])
    return {'average': lambda: (x+y)/2, 'min': lambda: np.minimum(x, y),
            'multiply': lambda: x*y, 'max': lambda: np.maximum(x, y)}[mode]()


def bound_material(prim):
    from pxr import UsdPhysics, UsdShade
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial('physics')
    if not material or not material.GetPrim().HasAPI(UsdPhysics.MaterialAPI):
        raise RuntimeError(f'No explicit physical material on {prim.GetPath()}')
    api = UsdPhysics.MaterialAPI(material.GetPrim())
    mode = material.GetPrim().GetAttribute('physxMaterial:frictionCombineMode').Get() or 'average'
    return (float(api.GetStaticFrictionAttr().Get()), float(api.GetDynamicFrictionAttr().Get()), str(mode))


class PrivilegedContacts:
    def __init__(self, env):
        from pxr import Usd, UsdPhysics
        stage = env.sim.stage
        coefficients = []
        for condition in env.conditions:
            prefix = f'/World/envs/env_{condition.env_id}'
            terrain = []
            for row in range(3):
                for col in range(3):
                    name = condition.case.grid[row][col]
                    prim = stage.GetPrimAtPath(f'{prefix}/Warehouse/Grid/Cell_{row}_{col}_{name}/terrain/mesh')
                    if not prim.IsValid() or not prim.HasAPI(UsdPhysics.CollisionAPI):
                        raise RuntimeError(f'Expected a collision mesh at {prim.GetPath()}')
                    terrain.append(bound_material(prim))
            feet = []
            for foot in FOOT_NAMES:
                root = stage.GetPrimAtPath(f'{prefix}/Spot/{foot}')
                colliders = [p for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
                             if p.HasAPI(UsdPhysics.CollisionAPI)]
                materials = [bound_material(p) for p in colliders]
                if not materials or any(m != materials[0] for m in materials):
                    raise RuntimeError(f'{foot}: expected uniform physical material across foot colliders')
                feet.append([combine_friction(materials[0], mat) for mat in terrain])
            coefficients.append(feet)
        self.coefficients = torch.tensor(np.asarray(coefficients), device=env.device, dtype=torch.float32)

    def extract(self, env, yaw):
        force_matrices = []
        point_matrices = []
        for sensor in env.foot_sensors:
            force = sensor.data.force_matrix_w
            if force.shape != (env.num_envs, 1, 9, 3):
                raise RuntimeError(f'Expected 9 terrain contact groups per foot, got {force.shape}')
            force_matrices.append(force[:, 0])
            point_matrices.append(sensor.data.contact_pos_w[:, 0])
        forces = torch.stack(force_matrices, dim=1)  # N, four feet, nine actual contact groups, xyz
        points = torch.stack(point_matrices, dim=1)
        magnitude = forces.norm(dim=-1)
        total = magnitude.sum(-1)
        contact = total > CONTACT_THRESHOLD_N
        weights = magnitude / total.clamp_min(1e-6)[..., None]
        # Multiple simultaneous materials: normal-force-weighted effective coefficients.
        friction = (self.coefficients * weights[..., None]).sum(2)
        d = env.robot.data
        pos = d.body_link_pos_w[:, env.foot_ids, None, :]
        linear = d.body_link_lin_vel_w[:, env.foot_ids, None, :]
        angular = d.body_ang_vel_w[:, env.foot_ids, None, :]
        delta = torch.where(torch.isfinite(points), points, pos) - pos
        velocity = linear + torch.cross(angular.expand_as(delta), delta, dim=-1)
        normals = forces / magnitude.clamp_min(1e-6)[..., None]
        tangent = velocity - (velocity*normals).sum(-1, keepdim=True)*normals
        tangent = (tangent*weights[..., None]).sum(2)
        c, s = yaw.cos()[:, None], yaw.sin()[:, None]
        slip = torch.stack((c*tangent[..., 0]+s*tangent[..., 1],
                            -s*tangent[..., 0]+c*tangent[..., 1]), dim=-1)
        friction *= contact[..., None]
        slip *= contact[..., None]
        total *= contact
        return torch.cat((friction.flatten(1), contact.float(), total, slip.flatten(1)), dim=1)


def terrain_height(env):
    """Collision-surface height below the base. Used only for termination, never actor perception."""
    import carb
    from omni.physx import get_physx_scene_query_interface
    query = get_physx_scene_query_interface()
    bases = env.robot.data.root_pos_w.detach().cpu().numpy()
    heights = []
    for i, base in enumerate(bases):
        hits = []
        prefix = f'/World/envs/env_{i}/Warehouse/'
        def report(hit):
            if str(hit.collision).startswith(prefix):
                hits.append(float(hit.position[2]))
            return True
        query.raycast_all(carb.Float3(float(base[0]), float(base[1]), float(base[2]+0.05)),
                          carb.Float3(0, 0, -1), 5.0, report)
        heights.append(max(hits) if hits else float('-inf'))
    return torch.tensor(heights, device=env.device)
