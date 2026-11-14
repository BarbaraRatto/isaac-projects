"""Draw separate route maps from the saved binary-terrain missions."""

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import font_manager
from matplotlib.patches import Circle, Polygon, Rectangle
from matplotlib.text import Text
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))
from multi_terrain_test13.layouts import ROCKS, MIRRORED

BACKGROUND = '#F8FAFC'
INK = '#263142'
EPISODE_COLORS = ('#166CA6', '#D47716', '#20815A', '#95469F', '#C34B59')


def configure_style(font_file=None):
    if font_file:
        font_manager.fontManager.addfont(str(font_file))
        family = font_manager.FontProperties(fname=str(font_file)).get_name()
    else:
        families = {entry.name for entry in font_manager.fontManager.ttflist}
        family = 'Times New Roman' if 'Times New Roman' in families else 'Liberation Serif'
    plt.rcParams.update({'font.family': family, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    return family


def project(points):
    # Keep X horizontal and Y vertical.
    return np.asarray(points)


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def enlarge_text(figure):
    # Increase every label by one point, including ticks and legends.
    figure.canvas.draw()
    for text in figure.findobj(match=Text):
        text.set_fontsize(text.get_fontsize() + 1)


def draw_world_axes(axis, scale=1):
    axis.set_xticks((0, 2, 4))
    axis.set_yticks((0, 2, 4))
    axis.tick_params(colors=INK, labelsize=15 * scale)
    axis.spines[['top', 'right']].set_visible(False)
    axis.spines[['bottom', 'left']].set_color('#7B8490')
    axis.set_xlabel('X [m]', fontsize=16 * scale, color=INK)
    axis.set_ylabel('Y [m]', fontsize=16 * scale, color=INK)


def draw_map_panel(axis, missions, samples, layout, palette, scale=1):
    if len(missions) != 5:
        raise ValueError('Expected five missions per map')
    axis.set_facecolor(BACKGROUND)
    for row_index, row in enumerate(layout.grid):
        for col_index, terrain in enumerate(row):
            x, y = col_index * 2, row_index * 2
            corners = project(((x - 1, y - 1), (x + 1, y - 1),
                               (x + 1, y + 1), (x - 1, y + 1)))
            axis.add_patch(Polygon(corners, facecolor=palette[terrain]['color'],
                                   edgecolor='#F8FAFC', linewidth=1.5, zorder=0))
            center = project((x, y))
            axis.text(*(center + (-.87, .82)), f'{row_index},{col_index}',
                      fontsize=12 * scale, color='#526071', ha='left', va='center', zorder=8,
                      bbox=dict(facecolor=palette[terrain]['color'], edgecolor='none',
                                alpha=.85, pad=.8))
            axis.text(*center, palette[terrain]['label'].capitalize(),
                      fontsize=27.5 * scale, fontweight='normal', color='#465262',
                      alpha=.73, ha='center', va='center', zorder=2)
    episode_handles = []
    for index, (color, mission) in enumerate(zip(EPISODE_COLORS, missions), 1):
        rows = [row for row in samples if row['mission_id'] == mission['mission_id']]
        if not rows or int(rows[-1]['terminal']) != 1:
            raise ValueError(f"Missing final sample: {mission['mission_id']}")
        xy = project([(float(row['x_m']), float(row['y_m'])) for row in rows])
        start = project((float(mission['start_x_m']), float(mission['start_y_m'])))
        goal = project((float(mission['goal_x_m']), float(mission['goal_y_m'])))
        axis.plot(xy[:, 0], xy[:, 1], color=color, lw=2.45 * scale, alpha=.98, zorder=4)
        axis.scatter(*start, s=30 * scale ** 2, color=color, edgecolor='white', lw=.5, zorder=6)
        axis.scatter(*goal, marker='*', s=185 * scale ** 2, color=color, edgecolor='white', lw=.6, zorder=6)
        axis.scatter(*xy[-1], marker='x', s=45 * scale ** 2, color=color, lw=1.8 * scale, zorder=7)
        midpoint = len(xy) // 2
        axis.annotate('', xy=xy[min(midpoint + 2, len(xy) - 1)], xytext=xy[midpoint],
                      arrowprops=dict(arrowstyle='->', color=color, lw=1.6 * scale), zorder=5)
        episode_handles.append(Line2D([], [], color=color, lw=2.45, label=f'Ep {index}'))
    for point, text, offset in (((0, 0), 'Start', (0, -31 * scale)),
                                 ((4, 4), 'Goal', (0, 35 * scale))):
        axis.annotate(text, project(point), xytext=offset, textcoords='offset points',
                      ha='center', va='center', fontsize=16 * scale, fontweight='bold', color=INK)
    axis.add_patch(Circle(project((4, 4)), .55, fill=False, ls='--',
                          edgecolor='#536276', lw=1.1, alpha=.75, zorder=3))
    draw_world_axes(axis, scale)
    axis.set(xlim=(-1, 5), ylim=(-1, 5), aspect='equal')
    return episode_handles


def marker_handles():
    return [Line2D([], [], color=INK, marker=marker, ls='', markersize=size, label=label)
            for marker, size, label in (('o', 6, 'Start'), ('*', 13, 'Goal'), ('x', 7, 'Final position'))]


def draw_map(run, map_name, layout, palette, output, preview):
    missions = read_csv(run / map_name / 'missions.csv')
    samples = read_csv(run / map_name / 'trajectories.csv')
    figure, axis = plt.subplots(figsize=(9, 10), facecolor=BACKGROUND)
    episode_handles = draw_map_panel(axis, missions, samples, layout, palette)
    suffix = 'A — rocks on the right' if map_name == 'rocks_right' else 'B — rocks on the left'
    figure.text(.5, .955, f'Binary terrain · Map {suffix}', ha='center',
                fontsize=21, fontweight='bold', color='#1B2735')
    figure.text(.5, .920, 'Five recorded episodes · 2 × 2 m cells', ha='center',
                fontsize=14, color='#536276')
    figure.legend(handles=episode_handles, loc='lower center', bbox_to_anchor=(.5, .099),
                  ncol=5, frameon=False, fontsize=13, handlelength=1.8, columnspacing=1.1)
    figure.legend(handles=marker_handles(), loc='lower center', bbox_to_anchor=(.5, .059),
                  ncol=3, frameon=False, fontsize=12)
    figure.text(.5, .031, 'Cell labels: row,column · dashed circle: 0.55 m goal radius',
                ha='center', fontsize=11, color='#536276')
    figure.subplots_adjust(left=.105, right=.965, top=.88, bottom=.21)
    enlarge_text(figure)
    for suffix in ('.svg', '.pdf', '.png'):
        figure.savefig(output.with_suffix(suffix), dpi=200, facecolor=BACKGROUND)
    if preview is not None:
        figure.savefig(preview, dpi=150, facecolor=BACKGROUND)
    plt.close(figure)


def draw_mission_plots(output, layout, missions, palette=None, font_file=None):
    configure_style(font_file)
    palette = palette or json.loads((HERE / 'terrain_palette.json').read_text())
    samples = read_csv(output / 'trajectories.csv')
    figure, axes = plt.subplots(1, 3, figsize=(17, 6.8), facecolor=BACKGROUND,
                                gridspec_kw={'width_ratios': (1.25, 1, 1)})
    handles = draw_map_panel(axes[0], missions, samples, layout, palette, scale=.68)
    axes[0].set_title('Recorded routes', fontsize=16, color=INK, pad=12)
    for index, axis in enumerate(axes[1:]):
        axis.set_facecolor(BACKGROUND)
        axis.spines[['top', 'right']].set_visible(False)
        axis.spines[['bottom', 'left']].set_color('#7B8490')
        axis.tick_params(colors=INK, labelsize=13)
        axis.grid(alpha=.16, color='#7B8490')
        field = 'energy_j' if index == 0 else 'goal_distance_m'
        for color, mission in zip(EPISODE_COLORS, missions):
            rows = [row for row in samples if row['mission_id'] == mission['mission_id']]
            axis.plot([float(row['time_s']) for row in rows],
                      [float(row[field]) for row in rows], color=color, lw=2.1)
        axis.set_xlabel('Time [s]', fontsize=13, color=INK)
        axis.set_ylabel('Absolute joint work [J]' if index == 0 else 'Distance to goal [m]',
                        fontsize=13, color=INK)
        axis.set_title('Cumulative mechanical work' if index == 0 else 'Distance to goal',
                       fontsize=16, color=INK, pad=12)
        axis.set_xlim(left=0)
        axis.set_ylim(bottom=0)
    axes[2].axhline(.55, color='#536276', ls='--', lw=1.1)
    axes[2].text(.98, .55, ' 0.55 m arrival threshold', transform=axes[2].get_yaxis_transform(),
                 ha='right', va='bottom', fontsize=10, color='#536276')
    name = 'A' if layout == ROCKS else 'B'
    figure.suptitle(f'Binary terrain · Map {name} · Mission diagnostics',
                   fontsize=21, fontweight='bold', color=INK, y=.96)
    figure.legend(handles=handles + marker_handles(), loc='lower center',
                  bbox_to_anchor=(.5, .075), ncol=8, frameon=False, fontsize=11)
    figure.text(.5, .028, 'Recorded data · simulated time · energy is absolute joint mechanical work',
                ha='center', fontsize=11, color='#536276')
    figure.subplots_adjust(left=.035, right=.98, bottom=.24, top=.82, wspace=.36)
    enlarge_text(figure)
    for suffix in ('.png', '.pdf'):
        figure.savefig(output / f'mission_plots{suffix}', dpi=200, facecolor=BACKGROUND)
    plt.close(figure)


def draw_palette(palette, path):
    figure, axis = plt.subplots(figsize=(12, 2.1), facecolor=BACKGROUND)
    axis.set(xlim=(0, 7), ylim=(0, 1))
    axis.axis('off')
    for index, entry in enumerate(palette.values()):
        axis.add_patch(Rectangle((index + .07, .41), .86, .45,
                                 facecolor=entry['color'], edgecolor='#7B8490', lw=.7))
        axis.text(index + .5, .27, entry['label'].capitalize(), ha='center', fontsize=15, color=INK)
        axis.text(index + .5, .08, entry['color'], ha='center', fontsize=11, color='#536276')
    figure.tight_layout(pad=1)
    enlarge_text(figure)
    figure.savefig(path, dpi=150, facecolor=BACKGROUND)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--preview-dir', type=Path)
    parser.add_argument('--font-file', type=Path, help='Optional Times New Roman font file')
    args = parser.parse_args()
    family = configure_style(args.font_file)
    print(f'Font: {family}')
    palette = json.loads((HERE / 'terrain_palette.json').read_text())
    output = args.run / 'figures'
    output.mkdir(exist_ok=True)
    if args.preview_dir:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
        draw_palette(palette, args.preview_dir / 'terrain_palette.png')
    for name, layout, label in (('rocks_right', ROCKS, 'map_a_routes'),
                                ('rocks_left', MIRRORED, 'map_b_routes')):
        preview = args.preview_dir / f'{label}.png' if args.preview_dir else None
        draw_map(args.run, name, layout, palette, output / f'{label}.svg', preview)
        draw_mission_plots(args.run / name, layout,
                           read_csv(args.run / name / 'missions.csv'), palette, args.font_file)
        configure_style(args.font_file)
        print(output / f'{label}.pdf')


if __name__ == '__main__':
    main()
