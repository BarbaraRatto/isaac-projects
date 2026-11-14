"""Fixed missions on the new map; no simulator imports."""

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import numpy as np

HERE = Path(__file__).resolve().parent
TRAINING = HERE.parent
PROJECT = TRAINING.parent
sys.path.insert(0, str(PROJECT))
from train_general_big_map_test13_SAC_Re.config import MISSION_CFG, NAV_CFG
from train_general_big_map_test13_SAC_Re.missions import deadline_steps

MAP = TRAINING / 'maps/lab_15x15_eval_01.usda'
METADATA = MAP.with_name(MAP.stem + '_metadata.json')
DRAG = MAP.with_name(MAP.stem + '_drag_manifest.json')
SUITE = HERE / 'missions_20.json'
CHECKPOINT = TRAINING / 'runs/general_map_test13_sac_re_20261110_104043_247976/policy_best.pt'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def segment_cells(start, goal, grid):
    """Exact positive-length cell intersections of the direct segment."""
    a, b = np.asarray(start[:2]), np.asarray(goal[:2])
    delta = b - a
    cuts = [0., 1.]
    for axis in (0, 1):
        if abs(delta[axis]) > 1e-12:
            cuts.extend((line - a[axis]) / delta[axis] for line in (3, 6, 9, 12)
                        if 0 < (line - a[axis]) / delta[axis] < 1)
    cuts = sorted(set(cuts))
    distance = float(np.linalg.norm(delta))
    result = []
    for lo, hi in zip(cuts, cuts[1:]):
        if hi - lo < 1e-10:
            continue
        x, y = a + (lo + hi) / 2 * delta
        row, col = int(y // 3), int(x // 3)
        result.append(dict(row=row, col=col, terrain=grid[row][col], length_m=(hi - lo) * distance))
    return result


def black_groups(grid):
    remaining = {(r, c) for r in range(5) for c in range(5) if grid[r][c] == 'lab_blackice'}
    groups = {}
    index = 0
    while remaining:
        first = min(remaining)
        remaining.remove(first)
        queue, cells = [first], [first]
        while queue:
            r, c = queue.pop()
            for cell in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if cell in remaining:
                    remaining.remove(cell)
                    cells.append(cell)
                    queue.append(cell)
        for cell in cells:
            groups[cell] = (index, len(cells))
        index += 1
    return groups


def endpoint(point, grid):
    x, y = map(float, point[:2])
    if not all(math.isfinite(value) and 0 <= value < 15 for value in (x, y)):
        raise ValueError('Endpoint outside the map')
    row, col = int(y // 3), int(x // 3)
    if min(x % 3, y % 3, 3 - x % 3, 3 - y % 3) < MISSION_CFG.cell_margin_m - 1e-7:
        raise ValueError('Endpoint too close to a cell edge')
    code = int(grid[row][col] == 'lab_blackice')
    return [x, y, 0., 0., code, row, col]


def build_suite():
    metadata = json.loads(METADATA.read_text())
    grid = metadata['grid']
    reference_file = PROJECT / 'train_general_big_map_test12_SAC_Re/new_map_eval/missions_20.json'
    reference = json.loads(reference_file.read_text())
    source_meta = json.loads((reference_file.parent.parent / 'maps/lab_15x15_eval_01_metadata.json').read_text())
    expected = [['lab_blackice' if cell == 'lab_green' else cell for cell in row] for row in source_meta['grid']]
    if grid != expected:
        raise ValueError('Expected the same special-cell layout as test12')
    if len(reference['cases']) != 20:
        raise ValueError('Expected twenty reference missions')
    grey = [(r, c) for r in range(5) for c in range(5) if grid[r][c] == 't1_asphalt']
    rng = np.random.default_rng(13132026)
    cases, used_goals = [], []
    for source_case in reference['cases']:
        case = copy.deepcopy(source_case)
        mission = case['mission']
        replaced = mission['goal'][4] != 0
        if replaced:
            lo, hi = case['distance_range_m']
            for _ in range(100000):
                row, col = grey[int(rng.integers(len(grey)))]
                goal = np.round(np.array([3 * col, 3 * row]) + rng.uniform(.65, 2.35, 2), 4)
                distance = math.dist(mission['start'][:2], goal)
                if not lo <= distance <= hi:
                    continue
                if any(np.linalg.norm(goal - old) < .6 for old in used_goals):
                    continue
                crossed = segment_cells(mission['start'], goal, grid)
                length = sum(c['length_m'] for c in crossed if c['terrain'] == 'lab_blackice')
                if length < 1.2:
                    continue
                mission['goal'] = endpoint(goal, grid)
                mission['distance_m'] = distance
                mission['band'] = 0 if distance < 3 else 1 if distance < 6 else 2 if distance < 10 else 3
                mission['deadline_steps'] = deadline_steps(distance)
                break
            else:
                raise RuntimeError('Cannot find a grey replacement goal')
            case['category'] = 'black_detour_short' if hi <= 6 else 'black_detour_medium'
        else:
            case['category'] = case['category'].replace('green', 'black')
        crossed = segment_cells(mission['start'], mission['goal'], grid)
        case.pop('direct_green_length_m', None)
        case.pop('direct_green_cells', None)
        case['direct_black_length_m'] = sum(c['length_m'] for c in crossed if c['terrain'] == 'lab_blackice')
        case['direct_black_cells'] = [[c['row'], c['col']] for c in crossed if c['terrain'] == 'lab_blackice']
        case['source_episode'] = source_case['episode']
        case['reference_goal_replaced'] = replaced
        if mission['start'][4] or mission['goal'][4]:
            raise ValueError('Test13 endpoints must both be grey')
        used_goals.append(np.array(mission['goal'][:2]))
        cases.append(case)
    return dict(schema=1, name='20 fixed missions on unseen test13 map 01',
                map_file=MAP.name, map_sha256=sha(MAP), metadata_sha256=sha(METADATA), drag_sha256=sha(DRAG),
                selection='Sixteen matching test12 missions; four green goals replaced by random grey goals in the same distance band',
                reference_suite=str(reference_file.relative_to(PROJECT)), reference_suite_sha256=sha(reference_file),
                unchanged_reference_episodes=list(range(1, 17)), replaced_goal_episodes=list(range(17, 21)),
                endpoint_margin_m=.65, all_starts_grey=True, all_goals_grey=True, goal_black_count=0,
                initial_heading='Same fixed headings as test12; original automatic alignment is retained',
                path_note='Intersections describe only the direct line; the policy chooses its actual route freely',
                cases=cases)


def load_suite(path=SUITE):
    document = json.loads(Path(path).read_text())
    for key, file in (('map_sha256', MAP), ('metadata_sha256', METADATA), ('drag_sha256', DRAG)):
        if document[key] != sha(file):
            raise ValueError(f'Mission suite does not match {file.name}')
    grid = json.loads(METADATA.read_text())['grid']
    cases = document['cases']
    if len(cases) != 20 or [case['episode'] for case in cases] != list(range(1, 21)):
        raise ValueError('Expected twenty ordered missions')
    grey_only = 0
    for index, case in enumerate(cases):
        mission = case['mission']
        for key in ('start', 'goal'):
            if endpoint(mission[key], grid) != mission[key]:
                raise ValueError('Endpoint material or coordinates mismatch')
        if mission['start'][4] != 0 or mission['goal'][4] != 0 or not math.isfinite(mission['yaw']):
            raise ValueError('Invalid endpoint material or heading')
        distance = math.dist(mission['start'][:2], mission['goal'][:2])
        if not math.isclose(distance, mission['distance_m'], abs_tol=1e-8):
            raise ValueError('Invalid mission distance')
        if mission['deadline_steps'] != deadline_steps(distance):
            raise ValueError('Invalid mission deadline')
        if [mission['start_id'], mission['goal_id']] != [2 * index, 2 * index + 1]:
            raise ValueError('Invalid endpoint IDs')
        crossed = segment_cells(mission['start'], mission['goal'], grid)
        cells = [[cell['row'], cell['col']] for cell in crossed if cell['terrain'] == 'lab_blackice']
        length = sum(cell['length_m'] for cell in crossed if cell['terrain'] == 'lab_blackice')
        if cells != case['direct_black_cells'] or not math.isclose(length, case['direct_black_length_m'], abs_tol=1e-8):
            raise ValueError('Invalid direct-line classification')
        grey_only += not cells
    if grey_only != 1 or any(c['mission']['goal'][4] for c in cases):
        raise ValueError('Expected one grey-only line and all endpoints on grey')
    return document


class FixedSampler:
    def __init__(self, document):
        self.document = copy.deepcopy(document)
        self.points = np.asarray([case['mission'][key] for case in document['cases']
                                  for key in ('start', 'goal')])

    def evaluation_mission(self, env_id, episode_index, seed, stage=0):
        if env_id != 0:
            raise ValueError('Run this suite with one robot')
        # The final auto-reset is not executed as another episode.
        case = self.document['cases'][min(episode_index, 19)]
        print(f"[MISSION] Ep {case['episode']}/20: {case['category']}", flush=True)
        return copy.deepcopy(case['mission'])

    def training(self, level):
        raise RuntimeError('This sampler is evaluation-only')


def export_suite(document, path):
    write_json(path, document)
    fields = ['episode', 'category', 'start_x_m', 'start_y_m', 'goal_x_m', 'goal_y_m',
              'goal_material', 'initial_yaw_deg', 'direct_distance_m', 'direct_black_length_m', 'deadline_s']
    with path.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in document['cases']:
            m = case['mission']
            writer.writerow(dict(episode=case['episode'], category=case['category'],
                                 start_x_m=m['start'][0], start_y_m=m['start'][1],
                                 goal_x_m=m['goal'][0], goal_y_m=m['goal'][1],
                                 goal_material='black' if m['goal'][4] else 'grey',
                                 initial_yaw_deg=math.degrees(m['yaw']), direct_distance_m=m['distance_m'],
                                 direct_black_length_m=case['direct_black_length_m'],
                                 deadline_s=m['deadline_steps'] * NAV_CFG.action_dt))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=SUITE)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Keep the agreed mission suite fixed; use a new output path')
    document = build_suite()
    export_suite(document, args.output)
    load_suite(args.output)
    print(args.output)
