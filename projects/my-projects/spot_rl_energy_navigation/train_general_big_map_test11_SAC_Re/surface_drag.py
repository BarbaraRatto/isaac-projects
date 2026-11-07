"""Runtime-only horizontal resistance; not a USD material property or energy bonus."""
import hashlib
import json
from pathlib import Path


class SurfaceDrag:
    def __init__(self, manifest_path, terrain_path, offset_xy=(0., 0.)):
        manifest_path, terrain_path = Path(manifest_path), Path(terrain_path)
        manifest = json.loads(manifest_path.read_text())
        if hashlib.sha256(terrain_path.read_bytes()).hexdigest() != manifest['maps_sha256'][terrain_path.name]:
            raise ValueError('Terrain differs from drag manifest; regenerate terrains')
        self.regions = []
        for region in manifest['surface_regions'][terrain_path.name]:
            coefficient = manifest['materials'][region['material']].get('horizontal_drag_ns_per_m', 0.)
            x0,y0,x1,y1 = region['bounds_xy_m']
            self.regions.append((coefficient, (x0+offset_xy[0], y0+offset_xy[1],
                                               x1+offset_xy[0], y1+offset_xy[1])))

    def apply(self, robot, origins=None):
        """Fill wrench buffers BEFORE controller.forward() writes them to physics.

        Root body index 0 is the articulation base. With positions omitted,
        Isaac Lab applies the force at the body's center of mass. Recompute
        zero force outside green each step, including after respawn.
        """
        import torch
        xy = robot.data.root_pos_w[:, :2]
        if origins is not None:
            xy = xy-origins[:,:2]
        velocity = robot.data.root_lin_vel_w
        coefficient = torch.zeros_like(xy[:, 0])
        for c,(x0,y0,x1,y1) in self.regions:
            inside = (xy[:,0]>=x0) & (xy[:,0]<x1) & (xy[:,1]>=y0) & (xy[:,1]<y1)
            coefficient = torch.where(inside, c, coefficient)
        forces = torch.zeros_like(velocity)
        forces[:,:2] = -coefficient[:,None] * velocity[:,:2]
        robot.set_external_force_and_torque(forces=forces[:,None,:],
                                           torques=torch.zeros_like(forces[:,None,:]),
                                           body_ids=[0], is_global=True)
        return forces
