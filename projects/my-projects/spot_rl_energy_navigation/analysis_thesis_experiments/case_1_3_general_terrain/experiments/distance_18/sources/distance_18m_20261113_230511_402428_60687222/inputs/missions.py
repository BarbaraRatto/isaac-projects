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
CHECKPOINT = TRAINING / 'checkpoints_importati/policy_best_stage_3.pt'
SEED = 2026111319
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


def build_suite():
    meta=json.loads(METADATA.read_text())
    assert sha(MAP)==meta['usd_sha256']
    surfaces=SurfaceChecks()
    rng=np.random.default_rng(SEED)
    cases=[]
    sampling=[]
    for distance in (18,):
        categories=('regular','ice','stairs','surface_choice' if distance==6 else 'multiple_choices')
        counts={name:0 for name in categories}
        starts=[]
        goals=[]
        attempts=0
        while min(counts.values())<10 and attempts<500000:
            attempts+=1
            a=rng.uniform([-2.35,-.85],[38.35,18.85])
            angle=rng.uniform(-np.pi,np.pi)
            b=a+distance*np.array([np.cos(angle),np.sin(angle)])
            if endpoint_margin(a)<.65 or endpoint_margin(b)<.65:
                continue
            if any(meta['grid'][cell_at(p)[0]][cell_at(p)[1]] in HAZARDS for p in (a,b)):
                continue
            # Separation prevents near-duplicate endpoints within each distance group.
            if any(math.dist(a,old)<.6 for old in starts) or any(math.dist(b,old)<.6 for old in goals):
                continue
            category,crossed,option=classify(a,b,meta['grid'],distance)
            if category not in counts or counts[category]>=10:
                continue
            start,goal=surfaces.endpoint(a),surfaces.endpoint(b)
            if start is None or goal is None:
                continue
            episode=len(cases)+1
            mission=dict(start_id=2*(episode-1),goal_id=2*(episode-1)+1,
                         start=start,goal=goal,yaw=float(rng.uniform(-np.pi,np.pi)),
                         band=2 if distance==6 else 4,distance_m=math.dist(a,b),
                         deadline_steps=deadline_steps(float(distance)))
            cases.append(dict(episode=episode,distance_group_m=distance,category=category,
                              category_name=CATEGORY_NAMES[category],mission=mission,
                              direct_cells=crossed,direct_terrain_lengths_m=lengths(crossed),
                              alternative_geometry=option,draw_attempt=attempts))
            counts[category]+=1
            starts.append(a)
            goals.append(b)
        if min(counts.values())<10:
            raise RuntimeError(f'Insufficient geometric candidates at {distance} m: {counts}')
        sampling.append(dict(distance_m=distance,attempts=attempts,accepted=40,counts=counts))
        print(sampling[-1],flush=True)
    # Keep each category together; accepted coordinates remain unchanged.
    cases.sort(key=lambda c:(c['distance_group_m'],('regular','ice','stairs','surface_choice','multiple_choices').index(c['category']),c['episode']))
    for index,case in enumerate(cases):
        case['episode']=index+1
        case['mission']['start_id']=2*index
        case['mission']['goal_id']=2*index+1
    return dict(schema=1,name='40 new eighteen-metre missions with test7 policy_best_stage_3',
                map_sha256=sha(MAP),metadata_sha256=sha(METADATA),
                selection=dict(design_note='New fixed random panel with test7 policy_best_stage_3; previous results remain separate',seed=SEED,rng='numpy.random.Generator(PCG64)',numpy_version=np.__version__,
                  start_sampling='Uniform XY over inset map bounding box, followed by rejection',
                  start_bounds_xy_m=[[-2.35,-.85],[38.35,18.85]],
                  goal_sampling='Uniform bearing in [-pi, pi); goal = start + distance * unit bearing',
                  distances_m=[18],distance_tolerance_m=1e-8,endpoint_margin_m=.65,
                  endpoint_excluded_terrains=sorted(HAZARDS),footprint_half_width_m=.55,
                  footprint_grid_shape=[11,11],max_endpoint_relief_m=.16,
                  min_start_separation_m=.6,min_goal_separation_m=.6,
                  initial_yaw='Independent uniform [-pi, pi); frozen in the mission list',
                  hazard_min_direct_intersection_m=1.,multiple_min_direct_cells=5,
                  multiple_min_distinct_terrains=4,
                  category_priority='18 m: multiple_choices first; 6 m: surface_choice before regular',
                  regular_6m='Asphalt, ramp or gravel only; at most two terrain types; no qualifying alternative',
                  regular_18m='No ice/stairs and fewer than five cells or four terrain types',
                  surface_choice='No ice/stairs; >=2 direct terrain types and >=1.5 m rocks/obstacles; one-bend alternative <=1.8x direct length, >=1.08x direct length and >=1 m less rocks/obstacles',
                  alternative_clearance_m=.55,
                  energy_note='Surface contrast is a geometric selection criterion, not a measured energy ranking',
                  sampling=sampling),
                path_note='Only endpoints are assigned to the policy. Alternative waypoints are selection evidence, never policy commands. No geodesic distance is computed.',
                cases=cases)


def load_suite(path=SUITE,distance='18'):
    if distance not in ('18','all'):
        raise ValueError('This panel contains eighteen-metre missions only')
    document=json.loads(Path(path).read_text())
    if document['map_sha256']!=sha(MAP) or document['metadata_sha256']!=sha(METADATA):
        raise ValueError('Mission file does not match the evaluation map')
    grid=json.loads(METADATA.read_text())['grid']
    if [c['episode'] for c in document['cases']] != list(range(1,41)):
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
    if len(document['cases'])!=40 or len(counts)!=4 or set(counts.values())!={10}:
        raise ValueError('Expected ten missions per distance/category')
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
