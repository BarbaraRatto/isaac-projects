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
CASE = HERE
PROJECT = HERE.parents[3]
sys.path.insert(0, str(PROJECT))
TRAINING = PROJECT / 'train_general_big_map_test7_SAC'
from train_general_big_map_test7_SAC.config import NAV_CFG, MISSION_CFG, TERRAIN_NAMES
from train_general_big_map_test7_SAC.missions import deadline_steps
MAP = CASE / 'maps/real_terrains_v2_eval_01.usd'
METADATA = MAP.with_name(MAP.stem + '_metadata.json')
SUITE = HERE / 'exp_coordinates.json'
CHECKPOINT = TRAINING / 'runs/general_map_test7_sac_20261030_003829_628341/policy_best.pt'
SEED = 2026111311
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


def corner_crossing(a,b,crossed,grid):
    ice=[item for item in crossed if item['terrain']=='t2_slippery']
    if len(ice)!=1 or not .4 <= ice[0]['length_m'] <= 1.6:
        return None
    item=ice[0]
    r,c=item['row'],item['col']
    distance=math.dist(a,b)
    index=crossed.index(item)
    entry_t=sum(x['length_m'] for x in crossed[:index])/distance
    exit_t=entry_t+item['length_m']/distance
    start=np.asarray(a[:2]); delta=np.asarray(b[:2])-start
    entry=start+entry_t*delta; exit=start+exit_t*delta
    xmin,xmax,ymin,ymax=6*c-3,6*c+3,3*r-1.5,3*r+1.5
    def edge(p):
        values=[abs(p[0]-xmin),abs(p[0]-xmax),abs(p[1]-ymin),abs(p[1]-ymax)]
        return int(np.argmin(values))
    edges={edge(entry),edge(exit)}
    if len(edges)!=2 or not (edges & {0,1}) or not (edges & {2,3}):
        return None
    left=0 in edges; bottom=2 in edges
    corner=[xmin if left else xmax,ymin if bottom else ymax]
    waypoint=[corner[0]+(-.75 if left else .75),corner[1]+(-.75 if bottom else .75)]
    if not (-2.45<=waypoint[0]<=38.45 and -.95<=waypoint[1]<=18.95):
        return None
    route_length=math.dist(a[:2],waypoint)+math.dist(waypoint,b[:2])
    if route_length>1.45*distance:
        return None
    if not clear_segment(a,waypoint,grid) or not clear_segment(waypoint,b,grid):
        return None
    return dict(ice_cell=[r,c],ice_chord_m=item['length_m'],
                entry_xy_m=entry.tolist(),exit_xy_m=exit.tolist(),corner_xy_m=corner,
                waypoint_xy_m=waypoint,length_m=route_length,clearance_m=.55)


def classify(a,b,grid,distance):
    if distance!=6:
        raise ValueError('This panel contains six-metre missions only')
    crossed=segment_cells(a,b,grid)
    totals=lengths(crossed)
    ice=totals.get('t2_slippery',0.)
    stairs=totals.get('t7_stairs',0.)
    option=None
    category=None
    if ice>0 and stairs<1e-8:
        option=corner_crossing(a,b,crossed,grid)
        if option:
            category='ice'
    elif stairs>=1. and ice<1e-8:
        category='stairs'
    elif ice<1e-8 and stairs<1e-8:
        option=alternative(a,b,grid,totals)
        if option and totals.get('t5_large_rocks',0.)>=1.5 and option['terrain_lengths_m'].get('t1_asphalt',0.)>=.75:
            category='surface_choice'
        else:
            option=None
            if len(totals)<=3 and clear_segment(a,b,grid):
                category='regular'
    return category,crossed,option


def zone(a,b):
    middle=(np.asarray(a[:2])+np.asarray(b[:2]))/2
    return ('west' if middle[0]<18 else 'east')+'_'+('south' if middle[1]<9 else 'north')


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
    if sha(MAP)!=meta['usd_sha256']:
        raise ValueError('Map hash differs')
    checker=SurfaceChecks()
    rng=np.random.default_rng(SEED)
    categories=('regular','ice','stairs','surface_choice')
    quotas={'west_south':3,'west_north':2,'east_south':2,'east_north':3}
    zones={name:0 for name in quotas}
    counts={name:0 for name in categories}
    starts=[]; goals=[]; cases=[]
    attempts=0
    while min(counts.values())<10 and attempts<1000000:
        attempts+=1
        a=rng.uniform([-2.35,-.85],[38.35,18.85])
        bearing=rng.uniform(-np.pi,np.pi)
        b=a+6*np.array([np.cos(bearing),np.sin(bearing)])
        if endpoint_margin(a)<.65 or endpoint_margin(b)<.65:
            continue
        if any(meta['grid'][cell_at(p)[0]][cell_at(p)[1]] in HAZARDS for p in (a,b)):
            continue
        if any(math.dist(a,old)<.6 for old in starts) or any(math.dist(b,old)<.6 for old in goals):
            continue
        category,crossed,option=classify(a,b,meta['grid'],6)
        if category not in counts or counts[category]>=10:
            continue
        area=zone(a,b)
        if category=='regular' and zones[area]>=quotas[area]:
            continue
        start,goal=checker.endpoint(a),checker.endpoint(b)
        if start is None or goal is None:
            continue
        episode=len(cases)+1
        mission=dict(start_id=2*(episode-1),goal_id=2*(episode-1)+1,start=start,goal=goal,
                     yaw=float(rng.uniform(-np.pi,np.pi)),band=2,distance_m=math.dist(a,b),
                     deadline_steps=deadline_steps(6.))
        cases.append(dict(episode=episode,distance_group_m=6,category=category,
                          category_name=CATEGORY_NAMES[category],mission=mission,
                          direct_cells=crossed,direct_terrain_lengths_m=lengths(crossed),
                          alternative_geometry=option,midpoint_zone=area,draw_attempt=attempts))
        counts[category]+=1
        if category=='regular':
            zones[area]+=1
        starts.append(a); goals.append(b)
    if min(counts.values())<10:
        raise RuntimeError(f'Cannot fill all categories: {counts}; regular zones: {zones}')
    cases.sort(key=lambda c:(categories.index(c['category']),c['episode']))
    for i,c in enumerate(cases):
        c['episode']=i+1
        c['mission']['start_id']=2*i
        c['mission']['goal_id']=2*i+1
    selection=dict(seed=SEED,rng='numpy.random.Generator(PCG64)',numpy_version=np.__version__,
        start_sampling='Uniform XY over inset map bounds, followed by category and zone rejection',
        start_bounds_xy_m=[[-2.35,-.85],[38.35,18.85]],
        goal_sampling='Uniform bearing [-pi, pi); goal = start + 6 m unit bearing',
        distances_m=[6],distance_tolerance_m=1e-8,endpoint_margin_m=.65,
        endpoint_excluded_terrains=sorted(HAZARDS),footprint_half_width_m=.55,
        footprint_grid_shape=[11,11],max_endpoint_relief_m=.16,
        min_start_separation_m=.6,min_goal_separation_m=.6,
        initial_yaw='Independent uniform [-pi, pi); fixed before execution',
        regular='No ice/stairs, <=3 traversable terrain types; 0.55 m clearance from hazard cells. Rocks and asphalt allowed.',
        regular_zone_definition='Start-goal midpoint; west/east split X=18 m; south/north split Y=9 m',
        regular_zone_quotas=quotas,
        ice='One ice cell crossed through adjacent edges; 0.4 to 1.6 m chord; no stairs; local outside-corner alternative <=1.45x direct length',
        ice_waypoint_corner_offset_m=.75,alternative_clearance_m=.55,
        stairs='At least 1 m stairs on the direct segment, no ice',
        surface_choice='At least 1.5 m rocks on direct segment; one-bend alternative contains >=0.75 m asphalt and >=1 m less rocks/obstacles; original length ratio 1.08 to 1.8',
        energy_note='Alternative surface exposure is geometric evidence, not a measured energy ranking',
        category_priority='Ice and stairs separated; surface_choice before regular',
        total_draw_attempts=attempts,accepted_counts=counts,regular_zone_counts=zones,
        previous_panel='../exp_coordinates.json',
        design_note='New prospective panel after preliminary tests, with changed geometric criteria and selected policy_best checkpoint')
    print(json.dumps(dict(attempts=attempts,counts=counts,regular_zones=zones)),flush=True)
    return dict(schema=2,name='40 new six-metre missions, policy_best, 2026-11-13',
        map_sha256=sha(MAP),metadata_sha256=sha(METADATA),selection=selection,
        path_note='Assigned start and goal only. Selection waypoints never constrain the policy. Previous missions and results stay separate.',cases=cases)


def load_suite(path=SUITE,distance='6'):
    document=json.loads(Path(path).read_text())
    if distance not in ('6','all'):
        raise ValueError('This panel has six-metre missions only')
    if document['map_sha256']!=sha(MAP) or document['metadata_sha256']!=sha(METADATA):
        raise ValueError('Mission file does not match this map')
    if [c['episode'] for c in document['cases']]!=list(range(1,41)):
        raise ValueError('Expected forty unique ordered episodes')
    grid=json.loads(METADATA.read_text())['grid']
    counts={}; zones={}
    for case in document['cases']:
        m=case['mission']; i=case['episode']-1
        if case['distance_group_m']!=6 or not math.isfinite(m['yaw']):
            raise ValueError('Invalid distance group or yaw')
        if not math.isclose(math.dist(m['start'][:2],m['goal'][:2]),6.,abs_tol=1e-8):
            raise ValueError('Invalid direct distance')
        if [m['start_id'],m['goal_id']]!=[2*i,2*i+1] or m['deadline_steps']!=deadline_steps(6.):
            raise ValueError('Invalid endpoint IDs or deadline')
        for key in ('start','goal'):
            p=m[key]; r,c=cell_at(p)
            if endpoint_margin(p)<.65-1e-8 or grid[r][c] in HAZARDS or p[4:]!=[TERRAIN_NAMES.index(grid[r][c]),r,c]:
                raise ValueError('Invalid endpoint')
        category,crossed,option=classify(m['start'][:2],m['goal'][:2],grid,6)
        if category!=case['category'] or crossed!=case['direct_cells'] or option!=case['alternative_geometry']:
            raise ValueError('Invalid geometric category evidence')
        if zone(m['start'],m['goal'])!=case['midpoint_zone']:
            raise ValueError('Invalid spatial zone')
        counts[category]=counts.get(category,0)+1
        if category=='regular':
            zones[case['midpoint_zone']]=zones.get(case['midpoint_zone'],0)+1
    if len(counts)!=4 or set(counts.values())!={10} or zones!=document['selection']['regular_zone_quotas']:
        raise ValueError('Category or spatial quotas differ')
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
    for group in ('all',6):
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
