"""Batched equivalent of the reference drag, preserving region and force semantics."""
import torch
from train_general_big_map_test12_SAC_Re.surface_drag import SurfaceDrag


class BatchedSurfaceDrag(SurfaceDrag):
    def __init__(self, manifest_path, terrain_path):
        super().__init__(manifest_path,terrain_path)
        self.tables={}

    def apply(self,robot,origins=None):
        xy=robot.data.root_pos_w[:,:2]
        if origins is not None:
            xy=xy-origins[:,:2]
        key=(xy.device,xy.dtype)
        if key not in self.tables:
            self.tables[key]=(
                torch.tensor([b for c,b in self.regions],device=xy.device,dtype=xy.dtype),
                torch.tensor([c for c,b in self.regions],device=xy.device,dtype=xy.dtype))
        bounds,coefficients=self.tables[key]
        inside=((xy[:,None]>=bounds[None,:,:2]) & (xy[:,None]<bounds[None,:,2:])).all(-1)
        selected=torch.where(inside,torch.arange(len(bounds),device=xy.device)+1,0).amax(1)
        c=torch.where(selected>0,coefficients[(selected-1).clamp_min(0)],0.)
        force=torch.zeros_like(robot.data.root_lin_vel_w)
        force[:,:2]=-c[:,None]*robot.data.root_lin_vel_w[:,:2]
        robot.set_external_force_and_torque(forces=force[:,None],torques=torch.zeros_like(force[:,None]),
                                           body_ids=[0],is_global=True)
        return force
