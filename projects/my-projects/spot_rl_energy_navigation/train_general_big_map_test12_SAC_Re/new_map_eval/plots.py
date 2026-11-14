"""Offline mission plans and result plots; never starts the simulator."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Rectangle

from missions import HERE, METADATA, DRAG, load_suite

BACKGROUND = '#F8FAFC'
INK = '#263142'
COLORS = ['#166CA6', '#D47716', '#20815A', '#95469F', '#C34B59',
          '#167C86', '#8E5A3A', '#3C4B99', '#B44384', '#6D7825',
          '#227FBA', '#A35318', '#306B3B', '#6D589F', '#9C3550',
          '#31716B', '#77533B', '#62579B', '#A63569', '#7D6A1D']


def style():
    plt.rcParams.update({'font.family': 'Liberation Serif', 'pdf.fonttype': 42})


def draw_map(axis, metadata, materials, scale=1):
    axis.set_facecolor(BACKGROUND)
    for row, cells in enumerate(metadata['grid']):
        for col, name in enumerate(cells):
            key = 'green' if name == 'lab_green' else 'grey'
            axis.add_patch(Rectangle((3 * col, 3 * row), 3, 3,
                                    facecolor=materials[key]['color_srgb'], edgecolor='white', lw=.9))
            axis.text(3 * col + .12, 3 * row + 2.76, f'{row},{col}',
                      fontsize=12 * scale, color='#526071', va='center', zorder=2)
            axis.text(3 * col + 1.5, 3 * row + 1.5, key.capitalize(),
                      fontsize=22 * scale, color='#465262', alpha=.73, ha='center', va='center')
    axis.set(xlim=(0, 15), ylim=(0, 15), aspect='equal', xticks=range(0, 16, 3), yticks=range(0, 16, 3))
    axis.tick_params(labelsize=15 * scale, colors=INK)
    axis.set_xlabel('X [m]', fontsize=17 * scale, color=INK)
    axis.set_ylabel('Y [m]', fontsize=17 * scale, color=INK)
    axis.spines[['top', 'right']].set_visible(False)
    axis.spines[['bottom', 'left']].set_color('#7B8490')


def draw_mission(axis, case, color, trajectory=None, radius=False):
    mission = case['mission']
    start, goal = mission['start'], mission['goal']
    if trajectory is None:
        axis.plot([start[0], goal[0]], [start[1], goal[1]], ls='--', color=color, lw=1.8, zorder=4)
    else:
        axis.plot([float(row['x_m']) for row in trajectory], [float(row['y_m']) for row in trajectory],
                  color=color, lw=2.1, zorder=4)
        axis.scatter(float(trajectory[-1]['x_m']), float(trajectory[-1]['y_m']),
                     color=color, marker='x', s=36, lw=1.6, zorder=6)
    axis.scatter(start[0], start[1], marker='o', s=33, color=color, edgecolor='white', lw=.5, zorder=5)
    axis.scatter(goal[0], goal[1], marker='*', s=150, color=color, edgecolor='white', lw=.5, zorder=5)
    if radius:
        axis.add_patch(Circle(goal[:2], .55, fill=False, ls='--', color='#536276', lw=.9, zorder=3))


def plan(document):
    style()
    metadata = json.loads(METADATA.read_text())
    materials = json.loads(DRAG.read_text())['materials']
    with PdfPages(HERE / 'missions_plan.pdf') as pdf:
        for offset in range(0, 20, 4):
            figure, axes = plt.subplots(2, 2, figsize=(12, 13), facecolor=BACKGROUND)
            for axis, case in zip(axes.flat, document['cases'][offset:offset + 4]):
                draw_map(axis, metadata, materials, scale=.62)
                draw_mission(axis, case, COLORS[case['episode'] - 1])
                axis.set_title(f"Ep {case['episode']} · {case['category'].replace('_', ' ')}\n"
                               f"Direct distance: {case['mission']['distance_m']:.2f} m", fontsize=12, color=INK)
            figure.suptitle('New map 01 · planned missions', fontsize=21, fontweight='bold', color=INK)
            figure.text(.5, .02, 'Dashed lines show assigned start–goal segments, not robot trajectories. '
                        'Circle: start · star: goal', ha='center', fontsize=11, color='#536276')
            figure.tight_layout(rect=(0, .04, 1, .96), h_pad=4, w_pad=2)
            pdf.savefig(figure, facecolor=BACKGROUND)
            if offset == 0:
                figure.savefig(HERE / 'missions_plan_preview.png', dpi=140, facecolor=BACKGROUND)
            plt.close(figure)


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def plot_results(output, document):
    style()
    metadata = json.loads((output / 'inputs' / METADATA.name).read_text())
    materials = json.loads((output / 'inputs' / DRAG.name).read_text())['materials']
    trajectories = read_csv(output / 'trajectories.csv')
    episodes = read_csv(output / 'episodes.csv')
    by_episode = {int(row['episode']): row for row in episodes}
    paths = {case['episode']: [row for row in trajectories if int(row['episode']) == case['episode']]
             for case in document['cases']}
    figure, axis = plt.subplots(figsize=(11, 12), facecolor=BACKGROUND)
    draw_map(axis, metadata, materials)
    handles = []
    for case in document['cases']:
        episode = case['episode']
        if episode not in by_episode:
            continue
        color = COLORS[episode - 1]
        draw_mission(axis, case, color, paths[episode])
        handles.append(Line2D([], [], color=color, lw=2.2, label=f'Ep {episode}'))
    count = sum(int(row['success']) for row in episodes)
    figure.suptitle(f'Test 12 SAC Re · new map 01 · {count}/{len(episodes)} goals reached',
                   fontsize=22, fontweight='bold', color=INK, y=.97)
    figure.legend(handles=handles, loc='lower center', ncol=5, bbox_to_anchor=(.5, .035), frameon=False, fontsize=13)
    figure.text(.5, .012, 'Circle: start · star: goal · cross: final position', ha='center', fontsize=12, color='#536276')
    figure.subplots_adjust(left=.10, right=.97, top=.91, bottom=.20)
    for suffix in ('.png', '.pdf'):
        figure.savefig(output / f'routes_overview{suffix}', dpi=180, facecolor=BACKGROUND)
    plt.close(figure)
    with PdfPages(output / 'routes_per_episode.pdf') as pdf:
        for offset in range(0, 20, 4):
            figure, axes = plt.subplots(2, 2, figsize=(12, 13), facecolor=BACKGROUND)
            for axis, case in zip(axes.flat, document['cases'][offset:offset + 4]):
                episode = case['episode']
                draw_map(axis, metadata, materials, scale=.62)
                if episode in by_episode:
                    row = by_episode[episode]
                    draw_mission(axis, case, COLORS[episode - 1], paths[episode], radius=True)
                    axis.set_title(f"Ep {episode} · {row['reason']}\n{float(row['energy_j']):.0f} J · "
                                   f"{float(row['path_m']):.2f} m · {float(row['time_s']):.1f} s", fontsize=13)
            figure.suptitle('Recorded trajectories · new map 01', fontsize=21, fontweight='bold', color=INK)
            figure.tight_layout(rect=(0, .02, 1, .96), h_pad=4, w_pad=2)
            pdf.savefig(figure, facecolor=BACKGROUND)
            plt.close(figure)
    figure, axes = plt.subplots(3, 1, figsize=(13, 10), facecolor=BACKGROUND)
    numbers = [int(row['episode']) for row in episodes]
    for axis, key, title in zip(axes, ('path_m', 'time_s', 'energy_j'),
                                ('Traveled distance [m]', 'Simulated time [s]', 'Absolute joint work [J]')):
        axis.set_facecolor(BACKGROUND)
        axis.bar(numbers, [float(row[key]) for row in episodes], color=[COLORS[n - 1] for n in numbers])
        axis.set_ylabel(title, fontsize=14)
        axis.set_xticks(range(1, 21))
        axis.tick_params(labelsize=13)
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='y', alpha=.15)
    axes[-1].set_xlabel('Episode', fontsize=15)
    figure.suptitle('Mission measurements · all outcomes retained', fontsize=22, fontweight='bold', color=INK)
    figure.tight_layout(rect=(0, .01, 1, .95), h_pad=2)
    for suffix in ('.png', '.pdf'):
        figure.savefig(output / f'mission_metrics{suffix}', dpi=180, facecolor=BACKGROUND)
    plt.close(figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    if args.plan:
        plan(load_suite())
    elif args.run:
        config = json.loads((args.run / 'evaluation_config.json').read_text())
        document = json.loads((args.run / 'inputs' / config.get('mission_file', 'missions_20.json')).read_text())
        plot_results(args.run, document)
    else:
        parser.error('Choose --plan or --run PATH')
