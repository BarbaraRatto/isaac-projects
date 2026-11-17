"""Validated raw logging shared with the policy evaluation runner."""
import csv
import math

class CSVLog:
    def __init__(self, path, fields):
        self.stream = path.open('w', newline='', buffering=65536)
        self.writer = csv.DictWriter(self.stream, fieldnames=fields)
        self.writer.writeheader()

    def write(self, row):
        self.writer.writerow(row)

    def flush(self):
        self.stream.flush()

    def close(self):
        self.stream.close()


def validate_logs(output, document, dt, decimation):
    with (output / 'episodes.csv').open(newline='') as stream:
        episodes = list(csv.DictReader(stream))
    if [int(row['episode']) for row in episodes] != [c['episode'] for c in document['cases']]:
        raise ValueError('Missing or unordered episode results')
    with (output / 'trajectories.csv').open(newline='') as stream:
        trajectories = list(csv.DictReader(stream))
    counts, integrals, previous = {}, {}, {}
    with (output / 'power_samples.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            episode, tick, power = int(row['episode']), int(row['physics_step']), float(row['power_w'])
            if tick != counts.get(episode, 0) or not math.isfinite(power) or power < 0:
                raise ValueError('Invalid physics power log')
            if not math.isclose(float(row['time_s']), tick * dt, abs_tol=1e-8):
                raise ValueError('Invalid physics timestamp')
            if episode in previous:
                integrals[episode] += .5 * (previous[episode] + power) * dt
            else:
                integrals[episode] = 0.
            previous[episode] = power
            counts[episode] = tick + 1
    for result, case in zip(episodes, document['cases']):
        episode, steps = int(result['episode']), int(result['steps'])
        data = [row for row in trajectories if int(row['episode']) == episode]
        if [int(row['step']) for row in data] != list(range(steps + 1)):
            raise ValueError(f'Missing trajectory samples: Ep {episode}')
        if [int(row['terminal']) for row in data] != [0] * steps + [1]:
            raise ValueError('Missing terminal position')
        for row in data:
            if not math.isclose(float(row['time_s']), int(row['step']) * dt * decimation, abs_tol=1e-8):
                raise ValueError('Invalid trajectory timestamp')
        start = case['mission']['start']
        if math.dist([float(data[0]['x_m']), float(data[0]['y_m'])], start[:2]) > 1e-4:
            raise ValueError('Reset does not match the fixed start')
        path = sum(math.hypot(float(b['x_m']) - float(a['x_m']), float(b['y_m']) - float(a['y_m']))
                   for a, b in zip(data, data[1:]))
        if not math.isclose(path, float(result['path_m']), rel_tol=1e-4, abs_tol=1e-4):
            raise ValueError('Trajectory and traveled distance differ')
        if counts[episode] != steps * decimation + 1:
            raise ValueError('Missing power samples')
        if not math.isclose(integrals[episode], float(result['energy_j']), rel_tol=1e-4, abs_tol=.02):
            raise ValueError('Power integral and recorded energy differ')
        if not math.isclose(float(data[-1]['energy_j']), float(result['energy_j']), rel_tol=1e-6, abs_tol=.001):
            raise ValueError('Terminal energy differs')
    return dict(logging_checks='passed', completed_missions=len(document['cases']))


