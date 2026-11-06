"""Torch sparse episode maps, batched across robots; no terrain or label access.

All map arithmetic/storage stays on the selected device. Float64 retains the
CPU reference's centered moments and precision at translated scene origins.
Keys encode env + signed XY bins; range overflow is rejected, never wrapped.
RGB resizing/normalization remains the original preprocessing in vision.py.
"""
import torch
from train_general_big_map_test10_SAC.config import (
    LOCAL_CELLS, CELL_SIZE, MEMORY_RESOLUTION_M, MEMORY_AGE_SECONDS,
    GEOMETRY_MIN_POINTS, GEOMETRY_MIN_XY_EIGENVALUE, LOOK_X_M, LOOK_Y_M,
    RESOLUTION_M, DEPTH_STRIDE,
)

_SIDE = 1 << 20
_OFFSET = _SIDE // 2
_AREA = _SIDE * _SIDE


def _sum(values, ids, count):
    out = values.new_zeros((count, *values.shape[1:]))
    return out.index_add_(0, ids, values)


def _local(points, bases, yaw):
    d = points-bases
    c, s = yaw.cos(), yaw.sin()
    return torch.stack((c*d[:, 0]+s*d[:, 1], -s*d[:, 0]+c*d[:, 1], d[:, 2]), -1)


def _cells(points):
    ix = torch.floor((points[:, 0]-(LOOK_X_M[0]-RESOLUTION_M/2))/RESOLUTION_M).long()
    iy = torch.floor((points[:, 1]-(LOOK_Y_M[0]-RESOLUTION_M/2))/RESOLUTION_M).long()
    valid = (ix >= 0) & (ix < len(LOOK_X_M)) & (iy >= 0) & (iy < len(LOOK_Y_M))
    return ix*len(LOOK_Y_M)+iy, valid


class _Store:
    def __init__(self, device, shape):
        self.keys = torch.empty(0, device=device, dtype=torch.int64)
        self.centers = torch.empty((0, 3), device=device, dtype=torch.float64)
        self.data = torch.empty((0, *shape), device=device, dtype=torch.float64)
        self.weight = torch.empty(0, device=device, dtype=torch.float64)
        self.times = torch.empty(0, device=device, dtype=torch.float64)

    def reset(self, reset_mask):
        keep = ~reset_mask[self.keys // _AREA]
        for name in ('keys', 'centers', 'data', 'weight', 'times'):
            setattr(self, name, getattr(self, name)[keep])

    def put(self, keys, centers, data, weight, times):
        # Incoming keys are unique. Overwrite, never accumulate repeated visits.
        if not len(keys):
            return
        if not len(self.keys):
            self.keys, self.centers, self.data, self.weight, self.times = keys, centers, data, weight, times
            return
        rows = torch.searchsorted(self.keys, keys)
        exists = (rows < len(self.keys)) & (self.keys[rows.clamp_max(len(self.keys)-1)] == keys)
        for name, values in (('centers', centers), ('data', data), ('weight', weight), ('times', times)):
            getattr(self, name)[rows[exists]] = values[exists]
        missing = ~exists
        # Sorting is only necessary when new spatial bins are discovered.
        new_keys = keys[missing]
        if len(new_keys):
            merged = torch.cat((self.keys, new_keys))
            order = merged.argsort()
            self.keys = merged[order]
            for name, values in (('centers', centers), ('data', data), ('weight', weight), ('times', times)):
                setattr(self, name, torch.cat((getattr(self, name), values[missing]))[order])


class TorchMapBank:
    def __init__(self, device):
        self.device = torch.device(device)
        self.anchor = None
        self.initialized = None
        self.visual = _Store(self.device, (32,))
        self.geometry = _Store(self.device, (3, 3))

    @torch.no_grad()
    def reset(self, env_ids):
        if self.anchor is None:
            return
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        reset_mask = torch.zeros(len(self.anchor), device=self.device, dtype=torch.bool)
        reset_mask[ids] = True
        self.initialized[ids] = False
        self.visual.reset(reset_mask)
        self.geometry.reset(reset_mask)

    def _update(self, points, features, envs, times, store):
        if not len(points):
            return
        p = points.to(torch.float64)-self.anchor[envs]
        if not bool(torch.isfinite(p).all()) or (features is not None and not bool(torch.isfinite(features).all())):
            raise ValueError('Non-finite map input')
        xy = torch.floor(p[:, :2]/MEMORY_RESOLUTION_M).long()+_OFFSET
        if bool(((xy < 0) | (xy >= _SIDE)).any()):
            raise ValueError('GPU map exceeds signed spatial key range')
        encoded = envs*_AREA+xy[:, 0]*_SIDE+xy[:, 1]
        keys, inv, count = torch.unique(encoded, sorted=True, return_inverse=True, return_counts=True)
        weight = count.to(torch.float64)
        centers = _sum(p, inv, len(keys))/weight[:, None]
        if features is not None:
            data = _sum(features.to(torch.float64), inv, len(keys))/weight[:, None]
        else:
            delta = p-centers[inv]
            data = _sum(delta[:, :, None]*delta[:, None, :], inv, len(keys))/weight[:, None, None]
        store.put(keys, centers, data, weight, times[keys // _AREA])

    @torch.no_grad()
    def observe(self, patches, features, patch_envs, dense, dense_envs, bases, yaw, times):
        bases, yaw, times = bases.double(), yaw.double(), times.double()
        n = len(bases)
        if features.shape != (len(patches), 32):
            raise ValueError('Expected one PCA32 vector per patch point')
        if self.anchor is None:
            if n >= (1 << 23):
                raise ValueError('Too many environments for GPU map keys')
            self.anchor = torch.zeros_like(bases)
            self.initialized = torch.zeros(n, device=self.device, dtype=torch.bool)
        if len(self.anchor) != n:
            raise ValueError('Environment count changed in a live map bank')
        self.anchor[~self.initialized] = bases[~self.initialized]
        self.initialized.fill_(True)
        self._update(patches, features, patch_envs, times, self.visual)
        self._update(dense, None, dense_envs, times, self.geometry)
        return self.query(bases, yaw, times)

    @torch.no_grad()
    def query(self, bases, yaw, times):
        n = len(bases)
        out = torch.zeros((n*LOCAL_CELLS, CELL_SIZE), dtype=torch.float32, device=self.device)
        if self.anchor is None:
            return out.reshape(n, LOCAL_CELLS, CELL_SIZE)
        base = bases.double()-self.anchor
        yaw, times = yaw.double(), times.double()
        for visual, store in ((True, self.visual), (False, self.geometry)):
            if not len(store.keys):
                continue
            envs = store.keys // _AREA
            local = _local(store.centers, base[envs], yaw[envs])
            cells, valid = _cells(local)
            ids = envs[valid]*LOCAL_CELLS+cells[valid]
            pts, w = local[valid], store.weight[valid]
            data = store.data[valid]
            angles = yaw[envs[valid]]
            ages = ((times[envs[valid]]-store.times[valid])/MEMORY_AGE_SECONDS).clamp(0, 1)
            size = n*LOCAL_CELLS
            total = _sum(w, ids, size)
            denom = total.clamp_min(1)
            age = _sum(ages*w, ids, size)/denom
            if visual:
                out[:, :32] = (_sum(data*w[:, None], ids, size)/denom[:, None]).float()
                out[:, 36] = (total > 0).float()
                out[:, 38] = age.float()
                continue
            center = _sum(pts*w[:, None], ids, size)/denom[:, None]
            delta = pts-center[ids]
            c, s = angles.cos(), angles.sin()
            rotation = torch.zeros((len(ids), 3, 3), device=self.device, dtype=torch.float64)
            rotation[:, 0, 0] = c; rotation[:, 0, 1] = s
            rotation[:, 1, 0] = -s; rotation[:, 1, 1] = c; rotation[:, 2, 2] = 1
            covs = rotation @ data @ rotation.transpose(1, 2)
            covariance = _sum((covs+delta[:, :, None]*delta[:, None, :])*w[:, None, None], ids, size)/denom[:, None, None]
            # Closed-form symmetric 2x2 eigensystem/solve avoids per-cell CPU loops.
            a, b, d = covariance[:, 0, 0], covariance[:, 1, 0], covariance[:, 1, 1]
            minimum = .5*(a+d-torch.sqrt(((a-d)**2+4*b*b).clamp_min(0)))
            good = (total >= GEOMETRY_MIN_POINTS) & (minimum >= GEOMETRY_MIN_XY_EIGENVALUE)
            determinant = torch.where(good, a*d-b*b, torch.ones_like(a))
            xz, yz = covariance[:, 0, 2], covariance[:, 1, 2]
            slope_x = (d*xz-b*yz)/determinant
            slope_y = (a*yz-b*xz)/determinant
            normal = torch.stack((-slope_x, -slope_y, torch.ones_like(a)), 1)
            variance = torch.einsum('bi,bij,bj->b', normal, covariance, normal)/normal.square().sum(1)
            geometry = torch.stack((center[:, 2], slope_x.atan(), slope_y.atan(), variance.clamp_min(0).sqrt()), 1)
            out[:, 32:36] = torch.where(good[:, None], geometry, 0.).float()
            out[:, 37] = good.float()
            out[:, 39] = torch.where(good, age, 0.).float()
        if not bool(torch.isfinite(out).all()):
            raise ValueError('Non-finite persistent map')
        return out.reshape(n, LOCAL_CELLS, CELL_SIZE)


def backproject_batch(depth, intrinsic, positions, quaternions, bodies, *, patches=False, stride=DEPTH_STRIDE):
    """NumPy reference sampling/ROS quaternion convention, plus robot rejection.

    Returns compact points, environment indices, and original pixel/patch indices.
    Process small environment chunks to bound temporary body-distance tensors.
    """
    device = depth.device
    n, height, width = depth.shape
    if patches:
        u = ((torch.arange(37, device=device, dtype=torch.float64)+.5)*width/37).round().long().clamp(0, width-1)
        v = ((torch.arange(37, device=device, dtype=torch.float64)+.5)*height/37).round().long().clamp(0, height-1)
    else:
        u = torch.arange(stride//2, width, stride, device=device)
        v = torch.arange(stride//2, height, stride, device=device)
    vv, uu = torch.meshgrid(v, u, indexing='ij')
    uu, vv = uu.flatten(), vv.flatten()
    result, environments, indices = [], [], []
    for first in range(0, n, 8):
        last = min(n, first+8)
        sampled = depth[first:last, vv, uu]
        # Compare in sensor dtype, like NumPy (float32(0.2) is excluded).
        valid = torch.isfinite(sampled) & (sampled > .2) & (sampled < 15.)
        z = sampled.double()
        k = intrinsic[first:last].double()
        optical = torch.stack(((uu[None]-k[:, 0, 2, None])*z/k[:, 0, 0, None],
                               (vv[None]-k[:, 1, 2, None])*z/k[:, 1, 1, None], z), -1)
        w, x, y, zz = quaternions[first:last].double().unbind(1)
        rotation = torch.stack((1-2*(y*y+zz*zz), 2*(x*y-zz*w), 2*(x*zz+y*w),
                                2*(x*y+zz*w), 1-2*(x*x+zz*zz), 2*(y*zz-x*w),
                                2*(x*zz-y*w), 2*(y*zz+x*w), 1-2*(x*x+y*y)), 1).reshape(-1, 3, 3)
        world = optical @ rotation.transpose(1, 2)+positions[first:last, None].double()
        # Chunk pixels, not bodies: bounded broadcast and no per-link Python loop.
        for start in range(0, len(uu), 2048):
            stop = start+2048
            dist = world[:, start:stop, None]-bodies[first:last, None].double()
            valid[:, start:stop] &= (dist.square().sum(-1) > .08**2).all(-1)
        ei, pi = valid.nonzero(as_tuple=True)
        result.append(world[ei, pi]); environments.append(ei+first); indices.append(pi)
    return torch.cat(result), torch.cat(environments), torch.cat(indices)
