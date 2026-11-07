"""Reproducible 15-episode evaluation, independent of the training curriculum."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from train_general_big_map_test13_SAC_Re.missions import MissionSampler, deadline_steps
from train_general_big_map_test13_SAC_Re.layouts import TRAIN_LAYOUTS, map_metadata


DEFAULT_SUITE=HERE/'artifacts/missions_9.json'


def build_suite(seed=231313, episodes=9):
    """Grey endpoints; last mission has ice on the straight segment, no forced route."""
    if episodes not in (9,15):
        raise ValueError('Expected 9 or 15 episodes')
    sampler=MissionSampler()
    rng=np.random.default_rng(seed)
    points=sampler.points
    grey=np.flatnonzero(sampler.evaluation & (sampler.materials==0))
    grid=np.array(map_metadata()['grid'])
    counts=(2,2,2,2) if episodes==9 else (4,4,3,3)
    groups=[('vicino',1.5,3.,counts[0],0),('2_celle',5.5,6.5,counts[1],1),
            ('3_celle',8.5,9.5,counts[2],2),('4_celle',11.5,12.5,counts[3],3),
            ('deviazione_ghiaccio',3.,6.,1,1)]
    used=set();cases=[]
    distances=np.linalg.norm(points[grey,None,:2]-points[grey,:2],axis=2)
    for label,lo,hi,count,band in groups:
        candidates=np.argwhere((distances>=lo)&(distances<=hi))
        candidates=candidates[rng.permutation(len(candidates))]
        accepted=0
        for ia,ib in candidates:
            a,b=int(grey[ia]),int(grey[ib])
            if a in used or b in used:
                continue
            line=np.linspace(points[a,:2],points[b,:2],241)
            indices=np.floor(line/3.).astype(int)
            fraction=float(np.mean(grid[indices[:,1],indices[:,0]]=='lab_blackice'))
            if label=='deviazione_ghiaccio' and fraction<.20:
                continue
            distance=float(distances[ia,ib])
            bearing=float(np.arctan2(points[b,1]-points[a,1],points[b,0]-points[a,0]))
            yaw=float((bearing+rng.uniform(-np.pi,np.pi)+np.pi)%(2*np.pi)-np.pi)
            mission=dict(start_id=a,goal_id=b,start=points[a].tolist(),goal=points[b].tolist(),
                         yaw=yaw,band=band,distance_m=distance,deadline_steps=deadline_steps(distance))
            cases.append(dict(episode=len(cases)+1,category=label,mission=mission,
                              straight_line_black_fraction=fraction))
            used.update((a,b)); accepted+=1
            if accepted==count:
                break
        if accepted!=count:
            raise ValueError(f'Not enough evaluation endpoint pairs for {label}')
    return dict(schema=1,name=f'{episodes} missioni grigio-grigio: vicino/6/9/12 m + deviazione ghiaccio',
                seed=seed,map_sha256=hashlib.sha256(TRAIN_LAYOUTS[0].usd.read_bytes()).hexdigest(),
                cells_interpretation='Euclidean XY distance: 2/3/4 cells approximately 6/9/12 m',
                endpoint_split='held-out evaluation candidates; black excluded from spawn and goal',cases=cases)


class SuiteSampler(MissionSampler):
    def __init__(self,path=DEFAULT_SUITE):
        super().__init__()
        self.document=json.loads(Path(path).read_text())
        self.cases=self.document['cases']
        if len(self.cases) not in (9,15):
            raise ValueError('Expected a 9- or 15-mission suite')
        if self.document['map_sha256']!=hashlib.sha256(TRAIN_LAYOUTS[0].usd.read_bytes()).hexdigest():
            raise ValueError('Mission suite was prepared for a different map')
        for case in self.cases:
            m=case['mission']
            a,b=m['start_id'],m['goal_id']
            if not self.spawn_allowed[a] or not self.goal_allowed[b]:
                raise ValueError('Black ice endpoints are forbidden')
            if not self.evaluation[a] or not self.evaluation[b]:
                raise ValueError('Suite endpoint is not in the evaluation split')
            if not np.array_equal(self.points[a],m['start']) or not np.array_equal(self.points[b],m['goal']):
                raise ValueError('Suite endpoint bank changed')

    def evaluation_mission(self,env_id,episode_index,seed,stage=0):
        if env_id!=0:
            raise ValueError('This suite runs sequentially with one robot')
        # DirectRLEnv auto-resets also after the final episode: wrap safely
        # without recording an additional episode.
        case=self.cases[episode_index%len(self.cases)]
        m=case['mission']
        print(f'[TEST{len(self.cases)}] Episodio {case["episode"]}/{len(self.cases)}: {case["category"]}; '
              f'distanza={m["distance_m"]:.2f} m; '
              f'A=({m["start"][0]:.2f},{m["start"][1]:.2f}) '
              f'B=({m["goal"][0]:.2f},{m["goal"][1]:.2f})',flush=True)
        return copy.deepcopy(m)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DEFAULT_SUITE)
    parser.add_argument('--episodes',type=int,choices=(9,15),default=9)
    args=parser.parse_args()
    document=build_suite(episodes=args.episodes)
    args.output.write_text(json.dumps(document,indent=2)+'\n')
    rows=[f'# Piano dei {args.episodes} episodi','',
          'Distanze in linea dritta; una cella misura 3 m. Start e goal sono sul grigio; episodio finale con ghiaccio sulla retta A–B.',
          '', '| Episodio | Gruppo | Distanza (m) | Start XY | Goal XY | Goal |',
          '|---:|---|---:|---|---|---|']
    for case in document['cases']:
        m=case['mission']
        rows.append(f'| {case["episode"]} | {case["category"]} | {m["distance_m"]:.2f} | '
                    f'({m["start"][0]:.2f}, {m["start"][1]:.2f}) | '
                    f'({m["goal"][0]:.2f}, {m["goal"][1]:.2f}) | '
                    f'{"Nero" if m["goal"][4]==1 else "Grigio"} |')
    args.output.with_suffix('.md').write_text('\n'.join(rows)+'\n')
    print(args.output)


if __name__=='__main__':main()
