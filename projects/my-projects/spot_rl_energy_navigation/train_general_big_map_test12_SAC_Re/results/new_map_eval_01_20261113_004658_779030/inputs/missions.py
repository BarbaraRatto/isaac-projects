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
from train_general_big_map_test12_SAC_Re.config import MISSION_CFG, NAV_CFG
from train_general_big_map_test12_SAC_Re.missions import deadline_steps

MAP = TRAINING / 'maps/lab_15x15_eval_01.usda'
METADATA = MAP.with_name(MAP.stem + '_metadata.json')
DRAG = MAP.with_name(MAP.stem + '_drag_manifest.json')
SUITE = HERE / 'missions_20.json'
CHECKPOINT = TRAINING / 'runs/general_map_test12_sac_re_20261109_203415_65592/policy_best.pt'


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


def green_groups(grid):
    remaining = {(r, c) for r in range(5) for c in range(5) if grid[r][c] == 'lab_green'}
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
    code = int(grid[row][col] == 'lab_green')
    return [x, y, 0., 0., code, row, col]


def build_suite():
    metadata = json.loads(METADATA.read_text())
    grid = metadata['grid']
    groups = green_groups(grid)
    grey = [(r, c) for r in range(5) for c in range(5) if grid[r][c] == 't1_asphalt']
    green = sorted(groups)
    rng = np.random.default_rng(13112026)
    specifications = [
        ('grey_only', 1.5, 3., 1, False),
        ('isolated_green_short', 3., 6., 3, False),
        ('isolated_green_medium', 6., 10., 3, False),
        ('green_cluster_medium', 6., 10., 3, False),
        ('multiple_green_regions_long', 10., 14., 3, False),
        ('green_cluster_long', 10., 14., 3, False),
        ('green_goal_short', 3., 6., 2, True),
        ('green_goal_medium', 6., 10., 2, True),
    ]
    cases, used_starts, used_goals = [], [], []

    def sample_point(cells):
        row, col = cells[int(rng.integers(len(cells)))]
        xy = np.array([3 * col, 3 * row]) + rng.uniform(.65, 2.35, 2)
        return np.round(xy, 4)

    for category, lo, hi, count, green_goal in specifications:
        accepted = 0
        for _ in range(300000):
            a, b = sample_point(grey), sample_point(green if green_goal else grey)
            distance = float(np.linalg.norm(b - a))
            if not lo <= distance <= hi:
                continue
            if any(np.linalg.norm(a - old) < .6 for old in used_starts):
                continue
            if any(np.linalg.norm(b - old) < .6 for old in used_goals):
                continue
            crossed = segment_cells(a, b, grid)
            green_cells = [(cell['row'], cell['col']) for cell in crossed if cell['terrain'] == 'lab_green']
            green_length = sum(cell['length_m'] for cell in crossed if cell['terrain'] == 'lab_green')
            if category == 'grey_only':
                allowed = not green_cells
            elif category.startswith('isolated_green'):
                allowed = len(green_cells) == 1 and groups[green_cells[0]][1] == 1 and green_length >= 1.
            elif category.startswith('green_cluster'):
                allowed = len(green_cells) >= 2 and len({groups[cell][0] for cell in green_cells}) == 1
            elif category.startswith('multiple_green'):
                allowed = len({groups[cell][0] for cell in green_cells}) >= 2 and green_length >= 2.
            else:
                allowed = green_length >= 1.2
            if not allowed:
                continue
            band = 0 if distance < 3 else 1 if distance < 6 else 2 if distance < 10 else 3
            mission = dict(start_id=2 * len(cases), goal_id=2 * len(cases) + 1,
                           start=endpoint(a, grid), goal=endpoint(b, grid),
                           yaw=float(rng.uniform(-np.pi, np.pi)), band=band,
                           distance_m=distance, deadline_steps=deadline_steps(distance))
            cases.append(dict(episode=len(cases) + 1, category=category, mission=mission,
                              distance_range_m=[lo, hi], direct_green_length_m=green_length,
                              direct_green_cells=[list(cell) for cell in green_cells]))
            used_starts.append(a)
            used_goals.append(b)
            accepted += 1
            if accepted == count:
                break
        if accepted != count:
            raise RuntimeError(f'Cannot fill mission category: {category}')
    return dict(schema=1, name='20 fixed missions on unseen test12 map 01',
                map_file=MAP.name, map_sha256=sha(MAP), metadata_sha256=sha(METADATA),
                drag_sha256=sha(DRAG), selection='Random endpoints filtered by distance and direct-line green intersections',
                endpoint_margin_m=.65, all_starts_grey=True, goal_green_count=4,
                initial_heading='Fixed random yaw; original automatic alignment is retained',
                path_note='Intersections describe the direct line only; the policy chooses its actual route freely',
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
        if mission['start'][4] != 0 or not math.isfinite(mission['yaw']):
            raise ValueError('Invalid start material or heading')
        distance = math.dist(mission['start'][:2], mission['goal'][:2])
        if not math.isclose(distance, mission['distance_m'], abs_tol=1e-8):
            raise ValueError('Invalid mission distance')
        if mission['deadline_steps'] != deadline_steps(distance):
            raise ValueError('Invalid mission deadline')
        if [mission['start_id'], mission['goal_id']] != [2 * index, 2 * index + 1]:
            raise ValueError('Invalid endpoint IDs')
        crossed = segment_cells(mission['start'], mission['goal'], grid)
        cells = [[cell['row'], cell['col']] for cell in crossed if cell['terrain'] == 'lab_green']
        length = sum(cell['length_m'] for cell in crossed if cell['terrain'] == 'lab_green')
        if cells != case['direct_green_cells'] or not math.isclose(length, case['direct_green_length_m'], abs_tol=1e-8):
            raise ValueError('Invalid direct-line classification')
        grey_only += not cells
    if grey_only != 1 or sum(c['mission']['goal'][4] == 1 for c in cases) != 4:
        raise ValueError('Expected one grey-only line and four green goals')
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
              'goal_material', 'initial_yaw_deg', 'direct_distance_m', 'direct_green_length_m', 'deadline_s']
    with path.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in document['cases']:
            m = case['mission']
            writer.writerow(dict(episode=case['episode'], category=case['category'],
                                 start_x_m=m['start'][0], start_y_m=m['start'][1],
                                 goal_x_m=m['goal'][0], goal_y_m=m['goal'][1],
                                 goal_material='green' if m['goal'][4] else 'grey',
                                 initial_yaw_deg=math.degrees(m['yaw']), direct_distance_m=m['distance_m'],
                                 direct_green_length_m=case['direct_green_length_m'],
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
