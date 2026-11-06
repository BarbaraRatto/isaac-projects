"""Build immutable mixed mission banks offline, no simulator or policy training."""
import argparse
import json
from pathlib import Path
import sys
import tempfile
from dataclasses import asdict
import numpy as np
from scipy.sparse.csgraph import dijkstra

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from train_general_big_map_test10_SAC.config import MISSION_CFG
from train_general_big_map_test10_SAC.planning import RouteGrid,route_side


def build(output, limit=2048):
    if output.exists():raise FileExistsError('Frozen mission bank already exists: '+str(output))
    meta=json.loads((HERE/'artifacts/map_metadata.json').read_text())
    with np.load(HERE/'artifacts/spawn_bank.npz') as stored:
        points=stored['points'];evaluation=stored['evaluation']
    planner=RouteGrid(meta)
    nodes,offset,valid=planner.snap(points)
    unique,inverse=np.unique(nodes[valid],return_inverse=True)
    row_for=np.full(len(points),-1,dtype=int);row_for[valid]=inverse
    references=np.full((len(points),len(points)),np.inf,dtype=np.float32)
    energy=np.full_like(references,np.inf)
    rng=np.random.default_rng(MISSION_CFG.bank_seed)
    pools={};examples={};counts={}
    with tempfile.TemporaryDirectory(prefix='test10-routes-') as temporary:
        predecessors=np.memmap(Path(temporary)/'geo',dtype='int32',mode='w+',shape=(len(unique),len(planner.xy)))
        energy_predecessors=np.memmap(Path(temporary)/'energy',dtype='int32',mode='w+',shape=predecessors.shape)
        for start in range(0,len(unique),64):
            end=min(start+64,len(unique))
            for graph,target,pred in ((planner.graph,references,predecessors),
                                      (planner.energy_graph,energy,energy_predecessors)):
                dist,previous=dijkstra(graph,indices=unique[start:end],return_predecessors=True)
                pred[start:end]=previous
                for row in range(start,end):
                    ids=np.flatnonzero(row_for==row)
                    for a in ids:
                        target[a]=dist[row-start,nodes]+offset[a]+offset
                        target[a,~valid]=np.inf
            if start%512==0:print(f'[MISSIONS] routes {end}/{len(unique)}',flush=True)
        for split,is_eval in (('train',False),('evaluation',True)):
            candidates=np.flatnonzero(valid&(evaluation==is_eval))
            starts=rng.permutation(candidates)[:MISSION_CFG.bank_eval_starts if is_eval else MISSION_CFG.bank_train_starts]
            for stage in range(5):
                groups={key:[] for key in ('direct','right','left','energy') if stage>=2 or key!='energy'}
                max_pairs=limit if not is_eval else max(128,limit//4)
                for a in starts:
                    distance=np.linalg.norm(points[candidates,:2]-points[a,:2],axis=1)
                    lo=min(MISSION_CFG.distance_bands[stage][0],MISSION_CFG.detour_bands[stage][0])
                    hi=max(MISSION_CFG.distance_bands[stage][1],MISSION_CFG.detour_bands[stage][1])
                    ids=candidates[(distance>=lo)&(distance<=hi)&np.isfinite(references[a,candidates])]
                    ids=rng.permutation(ids)[:160]
                    if not len(ids):continue
                    clear=planner.straight_clear(points[a,:2],points[ids,:2])
                    raw_clear=planner.straight_clear(points[a,:2],points[ids,:2],inflated=False)
                    for j,b in enumerate(ids):
                        direct=float(np.linalg.norm(points[b,:2]-points[a,:2]));ref=float(references[a,b])
                        if ref>direct*2.5+.5:continue
                        category=None;path=None
                        if clear[j] and MISSION_CFG.distance_bands[stage][0]<=direct<=MISSION_CFG.distance_bands[stage][1]:
                            category='direct'
                            if stage>=2 and len(groups['energy'])<max_pairs:
                                sample=points[a,:2]+np.linspace(0,1,65)[:,None]*(points[b,:2]-points[a,:2])
                                proxy=np.asarray(MISSION_CFG.material_proxy)[planner.codes(sample)].mean()*direct
                                if energy[a,b]<proxy*.95:
                                    ep=planner.path(energy_predecessors[row_for[a]],nodes[a],nodes[b],points[a,:2],points[b,:2])
                                    if ep is not None:
                                        length=float(np.linalg.norm(np.diff(ep,axis=0),axis=1).sum())
                                        if length>direct+.4:
                                            groups['energy'].append([a,b,length]);examples.setdefault(f'{split}/{stage}/energy',ep.tolist())
                        elif not raw_clear[j] and MISSION_CFG.detour_bands[stage][0]<=direct<=MISSION_CFG.detour_bands[stage][1]:
                            path=planner.path(predecessors[row_for[a]],nodes[a],nodes[b],points[a,:2],points[b,:2])
                            if path is None:continue
                            category,deviation=route_side(path,points[a,:2],points[b,:2])
                            if deviation<.25:continue
                            if stage==0:
                                direction=(points[b,:2]-points[a,:2])/direct
                                fraction=np.linspace(0,1,100)
                                line=points[a,:2]+fraction[:,None]*(points[b,:2]-points[a,:2])
                                bad=np.isin(planner.codes(line),(1,6))
                                if not bad.any():continue
                                first,last=fraction[bad][0]*direct,fraction[bad][-1]*direct
                                forward=(path-points[a,:2])@direction
                                lateral=(path-points[a,:2])@np.array([-direction[1],direction[0]])
                                if not (.55<=first<=1.8 and last<=2.25 and np.max(np.abs(lateral[forward<=last+.3]))<=2.25):continue
                        if category and len(groups[category])<max_pairs:
                            groups[category].append([a,b,ref])
                            if path is None:path=planner.path(predecessors[row_for[a]],nodes[a],nodes[b],points[a,:2],points[b,:2])
                            examples.setdefault(f'{split}/{stage}/{category}',path.tolist())
                    if all(len(value)>=max_pairs for value in groups.values()):break
                print(f'[MISSIONS] candidates {split} stage {stage}: '+str({k:len(v) for k,v in groups.items()}),flush=True)
                for name,pairs in groups.items():
                    key=f'{split}_{stage}_{name}'
                    minimum=32 if is_eval else 128
                    if len(pairs)<minimum:raise RuntimeError(f'Insufficient {key}: {len(pairs)} < {minimum}; no silent category substitution')
                    pools[key]=np.asarray(pairs,dtype=np.float64)
                    counts[key]=len(pairs)
                print(f'[MISSIONS] {split} stage {stage}: '+str({k:len(v) for k,v in groups.items()}),flush=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as stream:
        np.savez_compressed(stream,points=points,evaluation=evaluation,valid=valid,
                            reference_lengths=references,**pools)
    report=dict(schema=1,build_contract=asdict(MISSION_CFG),counts=counts,valid_endpoints=int(valid.sum()),resolution_m=planner.resolution,
                clearance_m=planner.clearance,examples=examples,
                limitations='2D reference routes clear ice/stairs with footprint margin; other terrains are not guaranteed stable. Energy grouping uses a material proxy, never oracle Joules or policy inputs.')
    (output.with_suffix('.json')).write_text(json.dumps(report,indent=2)+'\n')
    print('[MISSIONS] Saved '+str(output),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=HERE/'artifacts/missions.npz')
    parser.add_argument('--pairs-per-group',type=int,default=2048)
    args=parser.parse_args();build(args.output,args.pairs_per_group)
