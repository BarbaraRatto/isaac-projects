"""Offline mission geometry, never exposed as policy observations or rewards.

Conservative 2-D torso-footprint clearance from ice/stairs and map boundaries.
Other materials retain their existing locomotion/geometry semantics; this is a
reference corridor, not a proof that the frozen walking controller succeeds.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.spatial import cKDTree
from train_general_big_map_test10_SAC.config import MISSION_CFG, TERRAIN_NAMES


class RouteGrid:
    def __init__(self, meta, resolution=None, clearance=None):
        self.meta = meta
        self.resolution = resolution or MISSION_CFG.planner_resolution_m
        self.clearance = MISSION_CFG.route_clearance_m if clearance is None else clearance
        self.origin = np.asarray(meta['bounds_min'][:2], dtype=float)
        self.extent = np.asarray(meta['bounds_max'][:2])-self.origin
        self.shape = tuple(np.ceil(self.extent[::-1]/self.resolution).astype(int))
        yy, xx = np.indices(self.shape)
        self.xy = self.origin+np.stack((xx.ravel()+.5, yy.ravel()+.5), 1)*self.resolution
        self.rectangles = []
        for r, row in enumerate(meta['grid']):
            for c, name in enumerate(row):
                if name in MISSION_CFG.spawn_excluded:
                    lo = self.origin+np.array([c*meta['cell_width_m'], r*meta['cell_height_m']])
                    hi = lo+np.array([meta['cell_width_m'],meta['cell_height_m']])
                    self.rectangles.append((lo, hi))
        self.free = self.valid(self.xy)
        ids = np.arange(len(self.xy)).reshape(self.shape)
        edges, lengths = [], []
        for dr, dc in ((0,1),(1,0),(1,1),(1,-1)):
            rows, cols = np.indices(self.shape)
            rr, cc = rows+dr, cols+dc
            good = (rr>=0)&(rr<self.shape[0])&(cc>=0)&(cc<self.shape[1])
            a = ids[good]; b = ids[rr[good],cc[good]]
            keep = self.free[a]&self.free[b]
            if dr and dc:
                # Never squeeze diagonally between blocked squares.
                keep &= self.free[ids[rows[good],cc[good]]]&self.free[ids[rr[good],cols[good]]]
            a,b = a[keep],b[keep]
            edges.extend((np.stack((a,b)),np.stack((b,a))))
            lengths.extend((np.full(len(a),self.resolution*np.hypot(dr,dc)),)*2)
        edge = np.concatenate(edges,axis=1); distance=np.concatenate(lengths)
        shape=(len(self.xy),)*2
        self.graph = coo_matrix((distance,edge),shape=shape).tocsr()
        codes = self.codes(self.xy)
        costs = np.asarray(MISSION_CFG.material_proxy)[codes]
        weighted = distance*(costs[edge[0]]+costs[edge[1]])/2
        self.energy_graph = coo_matrix((weighted,edge),shape=shape).tocsr()
        self.free_ids=np.flatnonzero(self.free)
        self.tree=cKDTree(self.xy[self.free])

    def codes(self, xy):
        cr=np.floor((np.asarray(xy)-self.origin)/[self.meta['cell_width_m'],self.meta['cell_height_m']]).astype(int)
        grid=np.array([[TERRAIN_NAMES.index(x) for x in row] for row in self.meta['grid']])
        return grid[np.clip(cr[...,1],0,len(grid)-1),np.clip(cr[...,0],0,len(grid[0])-1)]

    def valid(self, xy):
        xy=np.asarray(xy)
        good=((xy>=self.origin+self.clearance)&(xy<=self.origin+self.extent-self.clearance)).all(-1)
        for lo,hi in self.rectangles:
            good &= ~((xy>=lo-self.clearance)&(xy<=hi+self.clearance)).all(-1)
        return good

    def straight_clear(self, a, b, inflated=True):
        a=np.asarray(a);b=np.atleast_2d(b);delta=b-a
        hit=np.zeros(len(b),dtype=bool)
        margin=self.clearance if inflated else 0.
        for lo,hi in self.rectangles:
            entry=np.zeros(len(b));exit=np.ones(len(b));possible=np.ones(len(b),dtype=bool)
            for axis in range(2):
                stationary=np.abs(delta[:,axis])<1e-12
                possible &= ~stationary|((a[axis]>=lo[axis]-margin)&(a[axis]<=hi[axis]+margin))
                inverse=np.divide(1.,delta[:,axis],out=np.zeros(len(b)),where=~stationary)
                t0=(lo[axis]-margin-a[axis])*inverse;t1=(hi[axis]+margin-a[axis])*inverse
                entry=np.maximum(entry,np.where(stationary,-np.inf,np.minimum(t0,t1)))
                exit=np.minimum(exit,np.where(stationary,np.inf,np.maximum(t0,t1)))
            hit |= possible&(entry<=exit)
        return ~hit

    def snap(self, points):
        distance,index=self.tree.query(np.asarray(points)[:,:2])
        nodes=self.free_ids[index]
        good=(distance<=self.resolution*1.5)&self.valid(np.asarray(points)[:,:2])
        for i in np.flatnonzero(good):
            good[i]=self.straight_clear(points[i,:2],self.xy[nodes[i]])[0]
        return nodes, distance, good

    def path(self, predecessor, source, target, a, b):
        route=[int(target)]
        while route[-1]!=source:
            previous=int(predecessor[route[-1]])
            if previous<0 or len(route)>len(self.xy):return None
            route.append(previous)
        return np.vstack((a,self.xy[route[::-1]],b))


def route_side(path,a,b):
    direction=np.asarray(b)-a
    offset=path-np.asarray(a)
    signed=(direction[0]*offset[:,1]-direction[1]*offset[:,0])/np.linalg.norm(direction)
    index=np.argmax(np.abs(signed))
    return ('left' if signed[index]>0 else 'right'),float(np.abs(signed[index]))
