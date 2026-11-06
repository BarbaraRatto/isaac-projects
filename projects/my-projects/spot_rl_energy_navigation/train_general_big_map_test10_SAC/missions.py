"""Goal-conditioned mixed curriculum. Mission labels/routes NEVER enter observations."""
from dataclasses import dataclass, asdict
from pathlib import Path
import math
import numpy as np
import torch
from train_general_big_map_test10_SAC.config import MISSION_CFG,NAV_CFG

BANK=Path(__file__).resolve().parent/'artifacts/missions.npz'


def place_missions(root,origins,clearance,missions):
    rows=[[*m['start'][:4],*m['goal'][:4],math.cos(m['yaw']/2),math.sin(m['yaw']/2),m['deadline_steps']] for m in missions]
    if not rows:return root,origins.clone(),torch.empty(0,dtype=torch.long,device=root.device)
    data=torch.tensor(rows,dtype=root.dtype,device=root.device)
    root[:,:2]=origins[:,:2]+data[:,:2]
    root[:,2]=origins[:,2]+data[:,3]+clearance+.02
    root[:,3:7]=0;root[:,3]=data[:,8];root[:,6]=data[:,9];root[:,7:]=0
    goals=origins.clone();goals[:,:2]+=data[:,4:6];goals[:,2]+=data[:,7]
    return root,goals,data[:,10].long()


def evaluation_interval(stage,override=None):
    return int(override if override is not None else MISSION_CFG.evaluation_intervals[stage])


def category_schedule(stage,n=128):
    """Weighted fair order: a one-robot 9-episode play also sees different classes."""
    weights=np.asarray(MISSION_CFG.category_weights[stage]);counts=np.zeros(len(weights),dtype=int);result=[]
    # Largest remainders give exact counts at 128 while fair order interleaves categories.
    quotas=np.floor(weights*n).astype(int)
    for j in np.argsort(-(weights*n-quotas),kind='stable')[:n-int(quotas.sum())]:quotas[j]+=1
    for i in range(n):
        deficit=weights*(i+1)-counts
        deficit[counts>=quotas]=-np.inf
        j=int(np.argmax(deficit));counts[j]+=1;result.append(MISSION_CFG.categories[j])
    return result


@dataclass
class Curriculum:
    level:int=0
    streak:int=0

    def update(self,by_category):
        if self.level>=4:return False
        required=[name for name,w in zip(MISSION_CFG.categories,MISSION_CFG.category_weights[self.level]) if w>0]
        sufficient=all(name in by_category and by_category[name]['episodes']>0 and
                       by_category[name]['success_rate']>=(MISSION_CFG.promotion_direct if name=='direct' else MISSION_CFG.promotion_other)
                       for name in required)
        self.streak=self.streak+1 if sufficient else 0
        if self.streak>=MISSION_CFG.promotion_evaluations:
            self.level+=1;self.streak=0;return True
        return False

    def state_dict(self):return asdict(self)

    def load_state_dict(self,state):
        self.level,self.streak=int(state['level']),int(state['streak'])
        if not 0<=self.level<5 or not 0<=self.streak<2:raise ValueError('Invalid curriculum state')


def deadline_steps(reference_length):
    seconds=np.clip(MISSION_CFG.deadline_base_s+MISSION_CFG.deadline_seconds_per_m*reference_length,
                    MISSION_CFG.deadline_min_s,MISSION_CFG.deadline_max_s)
    return int(np.ceil(seconds/NAV_CFG.action_dt))


def evaluation_distance_range(values):
    """Optional test-only range; never changes the training curriculum."""
    if values is None:return None
    limits=np.asarray(values,dtype=float)
    if limits.shape!=(2,) or not np.isfinite(limits).all() or not 0<limits[0]<limits[1]:
        raise ValueError('Distance range must contain finite values 0 < MIN < MAX, in metres')
    return tuple(float(x) for x in limits)


class MissionSampler:
    def __init__(self,path=BANK):
        with np.load(path,allow_pickle=False) as bank:
            self.points=bank['points'].copy();self.evaluation=bank['evaluation'].copy()
            self.valid=bank['valid'].copy();self.references=bank['reference_lengths'].copy()
            self.groups={key:bank[key].copy() for key in bank.files if key.startswith(('train_','evaluation_'))}
        self.evaluation_width=128
        self.eval_suite='mixed'
        self.eval_distance_range=None

    def make(self,a,b,reference,stage,category,rng):
        d=self.points[b,:2]-self.points[a,:2];distance=float(np.linalg.norm(d))
        bearing=np.arctan2(d[1],d[0]);limit=np.deg2rad(MISSION_CFG.heading_limits_deg[stage])
        yaw=(bearing+rng.uniform(-limit,limit)+np.pi)%(2*np.pi)-np.pi
        return dict(start_id=int(a),goal_id=int(b),start=self.points[a].tolist(),goal=self.points[b].tolist(),
                    yaw=float(yaw),band=int(stage),category=category,reference_length_m=float(reference),
                    distance_m=distance,deadline_steps=deadline_steps(reference))

    def sample(self,stage,rng=np.random,evaluation=False,category=None):
        if category is None:
            category=str(rng.choice(MISSION_CFG.categories,p=MISSION_CFG.category_weights[stage]))
        split='evaluation' if evaluation else 'train'
        if category=='free':
            candidates=np.flatnonzero(self.valid&(self.evaluation==evaluation))
            override=evaluation_distance_range(self.eval_distance_range) if evaluation else None
            lo,hi=override or MISSION_CFG.distance_bands[stage]
            for _ in range(1000):
                # No conditioning on hazards, required detour, side or material cost.
                a=int(rng.choice(candidates));distance=np.linalg.norm(self.points[candidates,:2]-self.points[a,:2],axis=1)
                choices=candidates[(distance>=lo)&(distance<=hi)&np.isfinite(self.references[a,candidates])]
                if len(choices):
                    b=int(rng.choice(choices))
                    return self.make(a,b,self.references[a,b],stage,category,rng)
            raise RuntimeError('No reachable random pair in this split/stage')
        key=f'{split}_{stage}_{category}'
        if key not in self.groups or not len(self.groups[key]):raise ValueError('Missing required mission group: '+key)
        pair=self.groups[key][int(rng.integers(len(self.groups[key]))) if hasattr(rng,'integers') else int(rng.randint(len(self.groups[key])))]
        return self.make(int(pair[0]),int(pair[1]),pair[2],stage,category,rng)

    def training(self,level):return self.sample(level)

    def evaluation_mission(self,env_id,episode_index,seed,stage=0):
        index=int(env_id)+int(episode_index)*self.evaluation_width
        category='free' if self.eval_suite=='free' else category_schedule(stage)[index%128]
        rng=np.random.default_rng(np.random.SeedSequence([seed,stage,env_id,episode_index]))
        return self.sample(stage,rng,evaluation=True,category=category)
