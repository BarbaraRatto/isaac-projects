"""Fixed random missions on the held-out seven-terrain map."""
import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
CASE = HERE.parents[1]
PROJECT = HERE.parents[3]
sys.path.insert(0, str(PROJECT))
TRAINING = PROJECT / 'train_general_big_map_test7_SAC'
from train_general_big_map_test7_SAC.config import NAV_CFG, MISSION_CFG, TERRAIN_NAMES
from train_general_big_map_test7_SAC.missions import deadline_steps
MAP = CASE / 'maps/real_terrains_v2_eval_01.usd'
METADATA = MAP.with_name(MAP.stem + '_metadata.json')
SUITE = HERE / 'exp_coordinates.json'
PER_CATEGORY = 100
TOTAL = 4 * PER_CATEGORY
CATEGORIES = ('regular','ice','stairs','multiple_choices')
REFERENCE = HERE / 'retained_reference.json'
CHECKPOINT = TRAINING / 'checkpoints_importati/policy_best_stage_3.pt'
SEED = 2026111501
HAZARDS = {'t2_slippery', 't7_stairs'}
ROUGH = {'t5_large_rocks', 't6_obstacles'}
CATEGORY_NAMES = {
    'regular': 'Simple route on traversable terrain',
    'ice': 'Ice on the direct segment, without stairs',
    'stairs': 'Stairs on the direct segment, without ice',
    'surface_choice': 'Alternative routes on different surfaces',
    'multiple_choices': 'Several surfaces and successive choices',
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def cell_at(xy):
    return int(math.floor((xy[1]+1.5)/3)), int(math.floor((xy[0]+3)/6))


def segment_cells(a, b, grid):
    a, b = np.asarray(a[:2]), np.asarray(b[:2])
    delta = b-a
    cuts = [0., 1.]
    for axis, lines in ((0, [6*i-3 for i in range(1,7)]), (1, [3*i-1.5 for i in range(1,7)])):
        if abs(delta[axis]) > 1e-12:
            cuts += [float((line-a[axis])/delta[axis]) for line in lines if 0 < (line-a[axis])/delta[axis] < 1]
    cuts = sorted(set(cuts))
    distance = float(np.linalg.norm(delta))
    result = []
    for lo, hi in zip(cuts, cuts[1:]):
        if hi-lo < 1e-10:
            continue
        row,col = cell_at(a+(lo+hi)/2*delta)
        if not (0 <= row < 7 and 0 <= col < 7):
            raise ValueError('Segment leaves the map')
        result.append(dict(row=row,col=col,terrain=grid[row][col],length_m=(hi-lo)*distance))
    return result


def lengths(crossed):
    result = {}
    for item in crossed:
        name=item['terrain']
        result[name]=result.get(name,0.)+item['length_m']
    return result


def endpoint_margin(xy):
    row,col=cell_at(xy)
    if not (0 <= row < 7 and 0 <= col < 7):
        return -1.
    x,y=xy[:2]
    return min(x-(6*col-3),6*col+3-x,y-(3*row-1.5),3*row+1.5-y)


def clear_segment(a, b, grid, margin=.55):
    """Reject segments touching expanded ice or stair rectangles."""
    a,b=np.asarray(a[:2]),np.asarray(b[:2])
    d=b-a
    for r,row in enumerate(grid):
        for c,name in enumerate(row):
            if name not in HAZARDS:
                continue
            lower=np.array([6*c-3-margin,3*r-1.5-margin])
            upper=np.array([6*c+3+margin,3*r+1.5+margin])
            lo,hi=0.,1.
            for axis in (0,1):
                if abs(d[axis]) < 1e-12:
                    if a[axis] < lower[axis] or a[axis] > upper[axis]:
                        lo,hi=1.,0.
                        break
                else:
                    x,y=sorted(((lower[axis]-a[axis])/d[axis],(upper[axis]-a[axis])/d[axis]))
                    lo,hi=max(lo,x),min(hi,y)
            if lo <= hi:
                return False
    return True


def alternative(a,b,grid,direct):
    """Find a geometrically different surface option, not an energy optimum."""
    direct_rough=sum(value for name,value in direct.items() if name in ROUGH)
    if direct_rough < 1.5 or len(direct)<2:
        return None
    distance=math.dist(a,b)
    candidates=[]
    for r,row in enumerate(grid):
        for c,name in enumerate(row):
            if name in HAZARDS:
                continue
            waypoint=[6.*c,3.*r]
            route_length=math.dist(a,waypoint)+math.dist(waypoint,b)
            if not 1.08*distance <= route_length <= 1.8*distance:
                continue
            if not clear_segment(a,waypoint,grid) or not clear_segment(waypoint,b,grid):
                continue
            parts=segment_cells(a,waypoint,grid)+segment_cells(waypoint,b,grid)
            terrain_lengths=lengths(parts)
            rough_length=sum(v for n,v in terrain_lengths.items() if n in ROUGH)
            if direct_rough-rough_length >= 1.:
                candidates.append(dict(waypoint_xy_m=waypoint,length_m=route_length,
                                       terrain_lengths_m=terrain_lengths,
                                       rough_length_reduction_m=direct_rough-rough_length))
    return min(candidates,key=lambda item:item['length_m']) if candidates else None


def classify(a,b,grid,distance):
    crossed=segment_cells(a,b,grid)
    totals=lengths(crossed)
    ice=totals.get('t2_slippery',0.)
    stairs=totals.get('t7_stairs',0.)
    mixed=len(crossed)>=5 and len(totals)>=4
    option=None
    if distance==18 and mixed:
        category='multiple_choices'
    elif ice>=1. and stairs<1e-8:
        category='ice'
    elif stairs>=1. and ice<1e-8:
        category='stairs'
    elif ice<1e-8 and stairs<1e-8:
        if distance==6:
            option=alternative(a,b,grid,totals)
            category='surface_choice' if option else 'regular' if len(totals)<=2 and not (set(totals)&ROUGH) else None
        else:
            category='regular'
    else:
        category=None
    return category,crossed,option


class SurfaceChecks:
    def __init__(self):
        from pxr import Usd, UsdGeom
        from train_general_big_map_test7_SAC.prepare_map import Surface
        self.stage=Usd.Stage.Open(str(MAP))
        cache=UsdGeom.XformCache()
        self.surfaces={}
        grid=json.loads(METADATA.read_text())['grid']
        for r,row in enumerate(grid):
            for c,name in enumerate(row):
                if name in HAZARDS:
                    continue
                mesh=UsdGeom.Mesh(self.stage.GetPrimAtPath(f'/World/Grid/Cell_{r}_{c}_{name}/terrain/mesh'))
                self.surfaces[r,c]=Surface(mesh,cache.GetLocalToWorldTransform(mesh.GetPrim()))
        v=np.linspace(-.55,.55,11)
        x,y=np.meshgrid(v,v)
        self.offsets=np.column_stack((x.ravel(),y.ravel()))
        self.grid=grid

    def endpoint(self,xy):
        r,c=cell_at(xy)
        if endpoint_margin(xy)<.65-1e-9 or (r,c) not in self.surfaces:
            return None
        heights=self.surfaces[r,c].height(np.asarray(xy)[None,:]+self.offsets)
        if not np.isfinite(heights).all() or np.ptp(heights)>.16:
            return None
        return [float(xy[0]),float(xy[1]),float(heights.min()),float(heights.max()),
                TERRAIN_NAMES.index(self.grid[r][c]),r,c]


def pair_key(a,b):
    return tuple(sorted((tuple(cell_at(a)),tuple(cell_at(b)))))


def pair_distance(a,b,c,d):
    """Endpoint RMS distance, treating forward and reverse routes as similar."""
    a,b,c,d=map(lambda p:np.asarray(p[:2]),(a,b,c,d))
    forward=np.sum((a-c)**2)+np.sum((b-d)**2)
    reverse=np.sum((a-d)**2)+np.sum((b-c)**2)
    return float(np.sqrt(min(forward,reverse)/2))


def bounds(cell):
    r,c=cell
    return np.array([6*c-2.35,3*r-.85]),np.array([6*c+2.35,3*r+.85])


def feasible_pair(a,b):
    lo,hi=bounds(a); other_lo,other_hi=bounds(b)
    gap=np.maximum(np.maximum(other_lo-hi,lo-other_hi),0.)
    far=np.maximum(np.abs(other_hi-lo),np.abs(hi-other_lo))
    return float(np.linalg.norm(gap)),float(np.linalg.norm(far))


def pair_points(a,b,rng,limit=3000):
    lo,hi=bounds(a); other_lo,other_hi=bounds(b)
    bearing=math.atan2(3*(b[0]-a[0]),6*(b[1]-a[1]))
    angles=[]
    for x in (other_lo[0]-hi[0],other_hi[0]-lo[0]):
        for y in (other_lo[1]-hi[1],other_hi[1]-lo[1]):
            angle=math.atan2(y,x)
            angles.append(bearing+(angle-bearing+np.pi)%(2*np.pi)-np.pi)
    for _ in range(limit):
        angle=rng.uniform(min(angles),max(angles))
        delta=18*np.array([math.cos(angle),math.sin(angle)])
        lower=np.maximum(lo,other_lo-delta)
        upper=np.minimum(hi,other_hi-delta)
        if np.all(upper-lower>1e-9):
            start=rng.uniform(lower,upper)
            yield start,start+delta


def novelty(a,b,reference):
    if getattr(novelty, '_reference', None) is not reference:
        novelty._reference=reference
        missions=[s['case']['mission'] for s in reference['slots']]
        novelty._starts=np.array([m['start'][:2] for m in missions])
        novelty._goals=np.array([m['goal'][:2] for m in missions])
        novelty._pairs={pair_key(m['start'],m['goal']) for m in missions}
    if pair_key(a,b) in novelty._pairs:
        return False
    c,d=novelty._starts,novelty._goals
    ac=np.sum((a-c)**2,axis=1);bd=np.sum((b-d)**2,axis=1)
    ad=np.sum((a-d)**2,axis=1);bc=np.sum((b-c)**2,axis=1)
    return bool(np.all(np.minimum(ac+bd,ad+bc)>=18.) and
                np.all(np.minimum(ac,ad)>=.64) and np.all(np.minimum(bc,bd)>=.64))


def diversity(cases):
    from collections import Counter
    starts=Counter(tuple(c['mission']['start'][5:7]) for c in cases)
    goals=Counter(tuple(c['mission']['goal'][5:7]) for c in cases)
    pairs=Counter(pair_key(c['mission']['start'],c['mission']['goal']) for c in cases)
    return dict(unique_start_cells=len(starts),unique_goal_cells=len(goals),unique_unordered_cell_pairs=len(pairs),
        max_missions_per_unordered_pair=max(pairs.values()),
        start_counts={f'{r},{c}':n for (r,c),n in sorted(starts.items())},
        goal_counts={f'{r},{c}':n for (r,c),n in sorted(goals.items())})


def build_suite():
    from collections import Counter
    meta=json.loads(METADATA.read_text()); grid=meta['grid']
    if sha(MAP)!=meta['usd_sha256']:
        raise ValueError('Map hash differs')
    reference=json.loads(REFERENCE.read_text())
    safe=[(r,c) for r,row in enumerate(grid) for c,name in enumerate(row) if name not in HAZARDS]
    requested=[]
    requested_report=[]
    for a,b in requested:
        low,high=feasible_pair(a,b)
        requested_report.append(dict(start_cell=list(a),goal_cell=list(b),min_distance_m=low,max_distance_m=high,
            distance_18m_possible=low<=18<=high,excluded_as_saved_pair=pair_key([6*a[1],3*a[0]],[6*b[1],3*b[0]]) in {
                pair_key(s['case']['mission']['start'],s['case']['mission']['goal']) for s in reference['slots']}))
    pairs=[(a,b) for a in safe for b in safe if a!=b and feasible_pair(a,b)[0]<=18<=feasible_pair(a,b)[1]]
    rng=np.random.default_rng(SEED)
    rng.shuffle(pairs)
    pool=[]
    for pair_index,(a,b) in enumerate(pairs):
        kept=0
        for start,goal in pair_points(a,b,rng):
            if not novelty(start,goal,reference):
                continue
            category,crossed,option=classify(start,goal,grid,18)
            if category not in CATEGORIES:
                continue
            pool.append(dict(a=start,b=goal,category=category,crossed=crossed,option=option,
                             start_cell=a,goal_cell=b,key=tuple(sorted((a,b))),tie=float(rng.random())))
            kept+=1
            if kept>=120:
                break
        if (pair_index+1)%100==0:
            print(f'[POOL] {pair_index+1}/{len(pairs)} cell pairs; {len(pool)} candidates',flush=True)
    # Enrich the scarce regular and stairs categories without changing geometric constraints.
    regular_pairs={ (c['start_cell'],c['goal_cell']) for c in pool if c['category'] in ('regular','stairs') }
    for a,b in sorted(regular_pairs):
        kept=0
        for start,goal in pair_points(a,b,rng,limit=12000):
            if not novelty(start,goal,reference):
                continue
            category,crossed,option=classify(start,goal,grid,18)
            if category not in ('regular','stairs'):
                continue
            pool.append(dict(a=start,b=goal,category=category,crossed=crossed,option=option,
                             start_cell=a,goal_cell=b,key=tuple(sorted((a,b))),tie=float(rng.random())))
            kept+=1
            if kept>=600:
                break
    pool_counts=Counter(c['category'] for c in pool)
    print('[POOL COUNTS]',dict(pool_counts),flush=True)
    print('[PAIR COUNTS]',{cat:len({c['key'] for c in pool if c['category']==cat}) for cat in CATEGORIES},flush=True)
    checker=SurfaceChecks()
    cases=[]; starts=[]; goals=[]
    counts=Counter(); sc=Counter(); gc=Counter(); pc=Counter(); category_sc=Counter(); category_gc=Counter()
    used=set(); bad=set()

    def add(index):
        candidate=pool[index]
        a,b=candidate['a'],candidate['b']
        category=candidate['category']
        if counts[category]>=PER_CATEGORY or pc[candidate['key']]>=100:
            return False
        if any(math.dist(a,p)<.25 for p in starts) or any(math.dist(b,p)<.25 for p in goals):
            return False
        if any(pair_distance(a,b,old['mission']['start'],old['mission']['goal'])<.4 for old in cases):
            return False
        start,goal=checker.endpoint(a),checker.endpoint(b)
        if start is None or goal is None:
            bad.add(index)
            return False
        episode=len(cases)+1
        mission=dict(start_id=2*(episode-1),goal_id=2*(episode-1)+1,start=start,goal=goal,
            yaw=float(rng.uniform(-np.pi,np.pi)),band=4,distance_m=math.dist(a,b),deadline_steps=deadline_steps(18.))
        minimum=min(pair_distance(a,b,s['case']['mission']['start'],s['case']['mission']['goal']) for s in reference['slots'])
        cases.append(dict(episode=episode,distance_group_m=18,category=category,category_name=CATEGORY_NAMES[category],
            mission=mission,direct_cells=candidate['crossed'],direct_terrain_lengths_m=lengths(candidate['crossed']),
            alternative_geometry=candidate['option'],minimum_saved_endpoint_rms_m=minimum,pool_index=index))
        starts.append(a);goals.append(b);counts[category]+=1
        sc[candidate['start_cell']]+=1;gc[candidate['goal_cell']]+=1;pc[candidate['key']]+=1
        category_sc[category,candidate['start_cell']]+=1;category_gc[category,candidate['goal_cell']]+=1
        used.add(index)
        return True

    # Include feasible user examples when they also pass novelty and placement checks.
    for a,b in requested:
        for index,candidate in enumerate(pool):
            if candidate['start_cell']==a and candidate['goal_cell']==b and add(index):
                break
    for _ in range(TOTAL):
        if len(cases)==TOTAL:
            break
        available=[i for i,c in enumerate(pool) if i not in used and i not in bad and counts[c['category']]<PER_CATEGORY and pc[c['key']]<100]
        if not available:
            break
        # Prefer uncovered cells and less repeated category/cell combinations.
        def score(i):
            c=pool[i];a,b=c['start_cell'],c['goal_cell'];cat=c['category']
            return (2000*(sc[a]>0)+2000*(gc[b]>0)+20*(sc[a]+gc[b])+200*pc[c['key']]
                    +12*(category_sc[cat,a]+category_gc[cat,b])+2*counts[cat]+c['tie'])
        category_order=['regular','stairs','ice','multiple_choices']
        available.sort(key=lambda i:(category_order.index(pool[i]['category']),score(i)))
        found=False
        for index in available:
            if add(index):
                found=True
                break
            bad.add(index)
        if not found:
            break
        if len(cases)%25==0:
            print('[SELECT]',len(cases),dict(counts),flush=True)
    if len(cases)!=TOTAL or any(counts[c]!=PER_CATEGORY for c in CATEGORIES):
        raise RuntimeError(f'Cannot fill panel with fixed diversity constraints: {len(cases)} {dict(counts)}')
    cases.sort(key=lambda c:(CATEGORIES.index(c['category']),c['episode']))
    for index,c in enumerate(cases):
        c['episode']=index+1;c['mission']['start_id']=2*index;c['mission']['goal_id']=2*index+1
    for report in requested_report:
        report['selected_episodes']=[c['episode'] for c in cases if c['mission']['start'][5:7]==report['start_cell'] and c['mission']['goal'][5:7]==report['goal_cell']]
    write_json(HERE/'requested_cell_pairs.json',requested_report)
    stats=diversity(cases)
    stats['by_category']={cat:diversity([c for c in cases if c['category']==cat]) for cat in CATEGORIES}
    write_json(HERE/'coverage_summary.json',stats)
    print('[COVERAGE]',stats['unique_start_cells'],stats['unique_goal_cells'],stats['unique_unordered_cell_pairs'],flush=True)
    return dict(schema=2,name='Try3: 400 diverse eighteen-metre missions with test7 stage3',map_sha256=sha(MAP),
        metadata_sha256=sha(METADATA),reference_sha256=sha(REFERENCE),
        selection=dict(seed=SEED,rng='numpy.random.Generator(PCG64)',numpy_version=np.__version__,per_category=PER_CATEGORY,
            categories=list(CATEGORIES),distance_m=18.,endpoint_margin_m=.65,footprint_half_width_m=.55,
            footprint_samples_per_axis=11,max_endpoint_relief_m=.16,endpoint_excluded_terrains=sorted(HAZARDS),
            min_start_separation_m=.25,min_goal_separation_m=.25,min_new_pair_endpoint_rms_m=.4,
            min_saved_pair_endpoint_rms_m=3.,min_endpoint_to_any_saved_endpoint_m=.8,
            saved_cell_pairs_excluded_in_both_directions=True,max_repetitions_per_unordered_cell_pair=100,
            candidate_sampling='Enumerate feasible ordered cell pairs; random bearing within the corner-defined angular span; uniform start inside source/shifted-goal rectangle intersection; goal=start+18m unit bearing',
            candidate_limit_per_ordered_pair=120,candidate_bearing_attempt_limit_per_pair=3000,
            scarce_category_supplement=dict(categories=['regular','stairs'],max_candidates_per_ordered_pair=600,bearing_attempt_limit=12000),
            balancing='Fill regular, stairs, ice, multiple_choices in this order, then greedy cell coverage and category/cell diversity with seeded random tie breaks; repeated cell pairs penalized',
            geometric_category_rules='Same as try1; multiple_choices has priority when >=5 cells and >=4 terrain types',
            initial_yaw='Independent uniform [-pi, pi), fixed in mission file',candidate_count=len(pool),feasible_ordered_pairs=len(pairs),
            reference_note='Frozen snapshot of all 32 selected episodes, including tentative cases; later collection edits do not change this panel',coverage=stats),
        path_note='Geometric diversity constraints describe assigned segments, not predicted policy outcomes. All 400 outcomes must be retained.',cases=cases)


def load_suite(path=SUITE,distance='18'):
    if distance not in ('18','all'):
        raise ValueError('This panel contains eighteen-metre missions only')
    document=json.loads(Path(path).read_text())
    if document['map_sha256']!=sha(MAP) or document['metadata_sha256']!=sha(METADATA):
        raise ValueError('Mission file does not match the evaluation map')
    if document.get('reference_sha256')!=sha(REFERENCE):
        raise ValueError('Retained reference snapshot changed')
    grid=json.loads(METADATA.read_text())['grid']
    if [c['episode'] for c in document['cases']] != list(range(1,TOTAL+1)):
        raise ValueError('Invalid episode IDs')
    counts={}
    for case in document['cases']:
        m=case['mission']
        d=case['distance_group_m']
        index=case['episode']-1
        if not math.isfinite(m['yaw']) or [m['start_id'],m['goal_id']] != [2*index,2*index+1]:
            raise ValueError('Invalid heading or endpoint IDs')
        if not math.isclose(math.dist(m['start'][:2],m['goal'][:2]),d,abs_tol=1e-8):
            raise ValueError('Mission distance differs from its assigned distance')
        for key in ('start','goal'):
            p=m[key]
            r,c=cell_at(p)
            if endpoint_margin(p)<.65-1e-8 or grid[r][c] in HAZARDS or p[4:]!=[TERRAIN_NAMES.index(grid[r][c]),r,c]:
                raise ValueError('Invalid endpoint')
        category,crossed,option=classify(m['start'][:2],m['goal'][:2],grid,d)
        if category!=case['category'] or crossed!=case['direct_cells'] or option!=case['alternative_geometry']:
            raise ValueError('Mission category evidence differs')
        if m['deadline_steps']!=deadline_steps(float(d)):
            raise ValueError('Invalid deadline')
        counts[d,category]=counts.get((d,category),0)+1
    if len(document['cases'])!=TOTAL or len(counts)!=4 or set(counts.values())!={PER_CATEGORY}:
        raise ValueError('Expected one hundred missions per category')
    reference=json.loads(REFERENCE.read_text())
    for case in document['cases']:
        m=case['mission']
        if not novelty(m['start'][:2],m['goal'][:2],reference):
            raise ValueError('Mission violates saved-case novelty criteria')
    if distance!='all':
        document['cases']=[c for c in document['cases'] if c['distance_group_m']==int(distance)]
    return document


class FixedSampler:
    def __init__(self,document):
        self.document=copy.deepcopy(document)
        self.points=np.array([c['mission'][key] for c in document['cases'] for key in ('start','goal')])
        if not (self.points[:,4]==0).any():
            raise ValueError('At least one asphalt endpoint is needed for standing calibration')

    def evaluation_mission(self,env_id,episode_index,seed,stage=0):
        if env_id!=0:
            raise ValueError('Use one evaluation robot')
        case=self.document['cases'][min(episode_index,len(self.document['cases'])-1)]
        print(f"[MISSION] Ep {case['episode']}: {case['distance_group_m']} m, {case['category']}",flush=True)
        return copy.deepcopy(case['mission'])

    def training(self,level):
        raise RuntimeError('Evaluation only')


def export_suite(document,path):
    write_json(path,document)
    fields=['episode','distance_group_m','category','start_x_m','start_y_m','goal_x_m','goal_y_m',
            'start_row','start_col','goal_row','goal_col','start_terrain','goal_terrain','initial_yaw_rad',
            'direct_distance_m','deadline_s']
    for group in ('all',18):
        name=path.with_suffix('.csv') if group=='all' else path.with_name(f'exp_coordinates_{group}m.csv')
        with name.open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields)
            writer.writeheader()
            for c in document['cases']:
                if group!='all' and c['distance_group_m']!=group:
                    continue
                m=c['mission']; a,b=m['start'],m['goal']
                writer.writerow(dict(episode=c['episode'],distance_group_m=c['distance_group_m'],category=c['category'],
                  start_x_m=a[0],start_y_m=a[1],goal_x_m=b[0],goal_y_m=b[1],start_row=a[5],start_col=a[6],
                  goal_row=b[5],goal_col=b[6],start_terrain=TERRAIN_NAMES[a[4]],goal_terrain=TERRAIN_NAMES[b[4]],
                  initial_yaw_rad=m['yaw'],direct_distance_m=m['distance_m'],deadline_s=m['deadline_steps']*NAV_CFG.action_dt))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=SUITE)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Mission file already exists; use a new output path')
    export_suite(build_suite(),args.output)
    load_suite(args.output)
    print(args.output)
