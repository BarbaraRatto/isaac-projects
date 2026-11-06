"""Final-cell reservoir and fit; simulator-independent and testable on CPU."""
import numpy as np
from train_general_big_map_test10_SAC.frozen_scalers import PHYSICAL


class CellReservoir:
    def __init__(self,capacity=4000,seed=123):
        self.capacity=capacity;self.rng=np.random.default_rng(seed)
        self.values=[[] for _ in range(8)]
        self.seen=np.zeros(8,dtype=int)
        self.views=[set() for _ in range(8)]

    def add(self,cells,labels,view_id):
        for code in range(8):
            selected=cells[(labels==code)&(cells[:,36]>.5),:32]
            if not len(selected):continue
            self.views[code].add(int(view_id))
            # Bound the contribution of one pose; correlations are not independent samples.
            selected=selected[self.rng.choice(len(selected),min(12,len(selected)),replace=False)]
            for value in selected:
                if not np.isfinite(value).all():raise ValueError('Non-finite cell feature')
                self.seen[code]+=1
                bucket=self.values[code]
                if len(bucket)<self.capacity:bucket.append(value.copy())
                else:
                    index=int(self.rng.integers(self.seen[code]))
                    if index<self.capacity:bucket[index]=value.copy()

    def ready(self,count,min_views):
        return (min(map(len,self.values[:7]))>=count and
                min(map(len,self.views[:7]))>=min_views and len(self.values[7])>=int(7*count*.25))

    def balanced(self,count):
        if min(map(len,self.values[:7]))<count:raise ValueError('Insufficient seven-material final cells')
        arrays=[];labels=[]
        for code in range(8):
            n=count if code<7 else int(7*count*.25) # mixed boundaries = 20% of final dataset
            if len(self.values[code])<n:raise ValueError('Insufficient mixed-boundary cells')
            ids=self.rng.choice(len(self.values[code]),n,replace=False)
            arrays.append(np.asarray(self.values[code])[ids]);labels.extend([code]*n)
        return np.concatenate(arrays),np.asarray(labels,dtype=np.int64)


def fit_scalers(train,validation,identity,pca_sha256,provenance):
    tx,ty=train;vx,vy=validation
    if tx.shape[1:]!=(32,) or vx.shape[1:]!=(32,) or not np.isfinite(tx).all() or not np.isfinite(vx).all():
        raise ValueError('Expected finite 32-D FINAL CELL features')
    mean=tx.astype(np.float64).mean(0);std=tx.astype(np.float64).std(0)
    floor=max(1e-3,.01*float(np.median(std)));scale=np.maximum(std,floor)
    normalized=(vx-mean)/scale
    fraction=float((np.abs(normalized)>10).mean())
    if fraction>.01:raise ValueError(f'Calibration clips too often on independent views: {fraction:.3%}')
    return dict(schema=1,mode='frozen_final_cells_before_training',goal_scale_m=10.,clip_normalized=10.,
                identity=identity,pca_sha256=pca_sha256,visual_mean=mean.tolist(),visual_scale=scale.tolist(),
                **PHYSICAL,calibration=dict(fit_source='final_valid_grid_cells',disjoint_viewpoints=True,
                    train_counts=[int((ty==j).sum()) for j in range(7)],
                    validation_counts=[int((vy==j).sum()) for j in range(7)],
                    mixed_train=int((ty==7).sum()),mixed_validation=int((vy==7).sum()),
                    validation_clip_fraction=fraction,
                    clip_by_material=[float((np.abs(normalized[vy==j])>10).mean()) for j in range(7)],
                    visual_std_floor=floor,**provenance))
